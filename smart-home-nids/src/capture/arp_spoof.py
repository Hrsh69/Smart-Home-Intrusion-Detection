"""ARP Spoofing engine for Smart Home NIDS.

Routes traffic from target devices through this laptop so packets can be
sniffed and classified, while IP forwarding keeps the targets online.

Usage (CLI):
    sudo python -m src.capture.arp_spoof \\
        --targets 192.168.1.10 192.168.1.20 \\
        --gateway 192.168.1.1 \\
        --iface en0

Usage (library):
    spoofer = ArpSpoofer(
        targets=["192.168.1.10"],
        gateway="192.168.1.1",
        iface="en0",
    )
    spoofer.start()
    # ... capture traffic ...
    spoofer.stop()   # always call; also registered as atexit/signal handler

Safety:
    --targets is REQUIRED. The tool refuses to run without an explicit list
    to prevent accidentally spoofing unintended hosts.

Requirements:
    - scapy (pip install scapy)
    - Root / sudo on macOS; or CAP_NET_RAW + CAP_NET_ADMIN on Linux (Phase 5).
    - IP forwarding enabled (done automatically at start, restored at stop).
"""

from __future__ import annotations

import atexit
import logging
import os
import platform
import signal
import subprocess
import threading
import time
from typing import Optional

logger = logging.getLogger("nids.capture.arp_spoof")


# ── IP forwarding helpers ─────────────────────────────────────────────────────

def _enable_ip_forwarding() -> None:
    """Enable kernel IP forwarding so spoofed hosts stay online."""
    system = platform.system()
    if system == "Linux":
        _write_proc("/proc/sys/net/ipv4/ip_forward", "1")
        logger.info("Linux: IP forwarding enabled (/proc/sys/net/ipv4/ip_forward = 1)")
    elif system == "Darwin":
        _run_sysctl("net.inet.ip.forwarding", "1")
        logger.info("macOS: IP forwarding enabled (net.inet.ip.forwarding = 1)")
    else:
        logger.warning("Unknown OS '%s' — please enable IP forwarding manually.", system)


def _disable_ip_forwarding() -> None:
    """Restore IP forwarding to off after capture ends."""
    system = platform.system()
    if system == "Linux":
        _write_proc("/proc/sys/net/ipv4/ip_forward", "0")
        logger.info("Linux: IP forwarding disabled")
    elif system == "Darwin":
        _run_sysctl("net.inet.ip.forwarding", "0")
        logger.info("macOS: IP forwarding disabled")


def _write_proc(path: str, value: str) -> None:
    try:
        with open(path, "w") as f:
            f.write(value)
    except PermissionError:
        logger.error(
            "Cannot write %s — run with sudo or grant CAP_NET_ADMIN.", path
        )
        raise


def _run_sysctl(key: str, value: str) -> None:
    try:
        subprocess.run(
            ["sysctl", "-w", f"{key}={value}"],
            check=True, capture_output=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        logger.error("sysctl failed: %s — try running with sudo.", exc)
        raise


# ── ARP packet helpers ────────────────────────────────────────────────────────

def _get_mac(ip: str, iface: str) -> Optional[str]:
    """Resolve MAC address for an IP using an ARP request (returns None on timeout)."""
    try:
        from scapy.layers.l2 import ARP, Ether
        from scapy.sendrecv import srp

        ans, _ = srp(
            Ether(dst="ff:ff:ff:ff:ff:ff") / ARP(pdst=ip),
            iface=iface,
            timeout=3,
            verbose=False,
        )
        if ans:
            return ans[0][1].hwsrc
        logger.warning("No ARP reply from %s — is the host online?", ip)
        return None
    except ImportError:
        raise RuntimeError(
            "scapy is not installed. Run: pip install scapy"
        )


def _send_poison(
    target_ip: str,
    target_mac: str,
    spoof_ip: str,
    iface: str,
) -> None:
    """Send a single ARP reply telling target_ip that spoof_ip is at OUR MAC."""
    from scapy.layers.l2 import ARP
    from scapy.sendrecv import send

    pkt = ARP(
        op=2,                   # ARP reply
        pdst=target_ip,
        hwdst=target_mac,
        psrc=spoof_ip,          # lie: pretend we are the gateway (or target)
    )
    send(pkt, iface=iface, verbose=False)


def _send_restore(
    target_ip: str,
    target_mac: str,
    real_ip: str,
    real_mac: str,
    iface: str,
    count: int = 5,
) -> None:
    """Send correct ARP replies to un-poison a target."""
    from scapy.layers.l2 import ARP
    from scapy.sendrecv import send

    pkt = ARP(
        op=2,
        pdst=target_ip,
        hwdst=target_mac,
        psrc=real_ip,
        hwsrc=real_mac,
    )
    send(pkt, iface=iface, count=count, verbose=False)
    logger.info("Restored ARP for %s → %s (%s)", target_ip, real_ip, real_mac)


# ── Main class ────────────────────────────────────────────────────────────────

class ArpSpoofer:
    """Man-in-the-middle ARP spoofer for home LAN traffic capture.

    Poisons ARP caches on *targets* so their traffic routes through this
    machine (MITM), while simultaneously poisoning the *gateway* so return
    traffic also flows through here.  IP forwarding is enabled so targets
    retain internet access.

    Always call stop() when done.  It is also registered with atexit and
    SIGINT/SIGTERM so it runs even if the process is killed.

    Args:
        targets:    List of target IP addresses to MITM.
        gateway:    LAN gateway IP (router).
        iface:      Network interface to use (e.g. "en0", "eth0").
        interval:   Seconds between ARP poison bursts (default 2).
    """

    def __init__(
        self,
        targets: list[str],
        gateway: str,
        iface: str,
        interval: float = 2.0,
    ) -> None:
        if not targets:
            raise ValueError(
                "--targets is required. Provide at least one IP address. "
                "This safeguard prevents accidentally spoofing unintended hosts."
            )

        self.targets = list(targets)
        self.gateway = gateway
        self.iface = iface
        self.interval = interval

        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

        # MACs resolved at start()
        self._target_macs: dict[str, str] = {}
        self._gateway_mac: Optional[str] = None

        # Register cleanup handlers — runs on normal exit AND signals
        atexit.register(self.stop)
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                signal.signal(sig, self._signal_handler)
            except (OSError, ValueError):
                pass  # Running in non-main thread — skip signal registration

    # ── Public API ───────────────────────────────────────────────────────────

    def start(self) -> None:
        """Resolve MACs, enable IP forwarding, start poisoning thread.

        Raises:
            RuntimeError: if scapy is not installed.
            ValueError: if targets list is empty.
            PermissionError: if IP forwarding cannot be enabled (needs root).
        """
        logger.info(
            "ArpSpoofer starting — targets=%s gateway=%s iface=%s",
            self.targets, self.gateway, self.iface,
        )

        # Resolve gateway MAC
        logger.info("Resolving gateway MAC for %s …", self.gateway)
        self._gateway_mac = _get_mac(self.gateway, self.iface)
        if not self._gateway_mac:
            raise RuntimeError(
                f"Could not resolve MAC for gateway {self.gateway}. "
                f"Is it reachable on {self.iface}?"
            )

        # Resolve target MACs
        for ip in self.targets:
            mac = _get_mac(ip, self.iface)
            if mac:
                self._target_macs[ip] = mac
                logger.info("Resolved %s → %s", ip, mac)
            else:
                logger.warning("Skipping %s — could not resolve MAC.", ip)

        if not self._target_macs:
            raise RuntimeError("No target MACs could be resolved. Aborting.")

        # Enable IP forwarding so targets stay online
        _enable_ip_forwarding()

        # Start background poison thread
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._poison_loop, daemon=True, name="arp-spoof"
        )
        self._thread.start()
        logger.info(
            "ARP poisoning active — %d target(s): %s",
            len(self._target_macs),
            list(self._target_macs.keys()),
        )

    def stop(self) -> None:
        """Stop poisoning and restore correct ARP entries on all targets.

        Safe to call multiple times.
        """
        if not self._stop_event.is_set():
            logger.info("ArpSpoofer stopping — restoring ARP on %d target(s)…",
                        len(self._target_macs))
            self._stop_event.set()

            if self._thread and self._thread.is_alive():
                self._thread.join(timeout=self.interval * 2 + 1)

            self._restore_all()
            _disable_ip_forwarding()
            logger.info("ArpSpoofer stopped. Network restored.")

    # ── Internal ─────────────────────────────────────────────────────────────

    def _poison_loop(self) -> None:
        """Background thread: send ARP poisons every `interval` seconds."""
        while not self._stop_event.wait(timeout=self.interval):
            for target_ip, target_mac in self._target_macs.items():
                try:
                    # Tell target: "the gateway is at MY MAC"
                    _send_poison(target_ip, target_mac, self.gateway, self.iface)
                    # Tell gateway: "target IP is at MY MAC"
                    _send_poison(self.gateway, self._gateway_mac, target_ip, self.iface)
                except Exception as exc:
                    logger.error("Poison send failed for %s: %s", target_ip, exc)

    def _restore_all(self) -> None:
        """Send correct ARP replies to un-poison all targets and the gateway."""
        if not self._gateway_mac:
            return
        for target_ip, target_mac in self._target_macs.items():
            try:
                # Restore target's ARP cache entry for gateway
                _send_restore(
                    target_ip=target_ip,
                    target_mac=target_mac,
                    real_ip=self.gateway,
                    real_mac=self._gateway_mac,
                    iface=self.iface,
                )
                # Restore gateway's ARP cache entry for target
                _send_restore(
                    target_ip=self.gateway,
                    target_mac=self._gateway_mac,
                    real_ip=target_ip,
                    real_mac=target_mac,
                    iface=self.iface,
                )
            except Exception as exc:
                logger.error("Restore failed for %s: %s", target_ip, exc)

    def _signal_handler(self, signum, frame) -> None:
        logger.info("Signal %d received — shutting down ARP spoofer.", signum)
        self.stop()


# ── CLI entry point ───────────────────────────────────────────────────────────

def _parse_args():
    import argparse

    parser = argparse.ArgumentParser(
        description="ARP spoofer for Smart Home NIDS live capture.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  sudo python -m src.capture.arp_spoof \\
      --targets 192.168.1.10 192.168.1.20 \\
      --gateway 192.168.1.1 \\
      --iface en0

  # Dry-run (no actual ARP packets sent):
  python -m src.capture.arp_spoof \\
      --targets 192.168.1.10 \\
      --gateway 192.168.1.1 \\
      --dry-run
""",
    )
    parser.add_argument(
        "--targets", nargs="+", required=True, metavar="IP",
        help="Target IP address(es) to MITM. REQUIRED — no default.",
    )
    parser.add_argument(
        "--gateway", required=True, metavar="IP",
        help="Gateway/router IP address.",
    )
    parser.add_argument(
        "--iface", default=None, metavar="INTERFACE",
        help="Network interface (e.g. en0, eth0). Auto-detected if omitted.",
    )
    parser.add_argument(
        "--interval", type=float, default=2.0, metavar="SECONDS",
        help="Seconds between ARP poison bursts (default: 2).",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print what would happen without sending any packets or changing routing.",
    )
    return parser.parse_args()


def _auto_detect_iface() -> str:
    """Return the first non-loopback interface with an IP address."""
    try:
        import socket
        import netifaces
        for iface in netifaces.interfaces():
            addrs = netifaces.ifaddresses(iface)
            if netifaces.AF_INET in addrs and not iface.startswith("lo"):
                return iface
    except ImportError:
        pass
    # Fallback: try common names
    for candidate in ("en0", "eth0", "wlan0", "wlp2s0"):
        if os.path.exists(f"/sys/class/net/{candidate}") or candidate == "en0":
            return candidate
    return "en0"


if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    args = _parse_args()
    iface = args.iface or _auto_detect_iface()

    if args.dry_run:
        print(f"[DRY RUN] Would spoof:")
        print(f"  Targets : {args.targets}")
        print(f"  Gateway : {args.gateway}")
        print(f"  Interface: {iface}")
        print(f"  Interval : {args.interval}s")
        print("[DRY RUN] No packets sent. No IP forwarding changed.")
        sys.exit(0)

    spoofer = ArpSpoofer(
        targets=args.targets,
        gateway=args.gateway,
        iface=iface,
        interval=args.interval,
    )

    try:
        spoofer.start()
        print(f"ARP spoofing active. Press Ctrl+C to stop and restore ARP.")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass  # atexit/signal handler calls stop()
