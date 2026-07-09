"""Packet sniffer for Smart Home NIDS.

Captures IP packets on a network interface using Scapy and puts them into a
queue for the flow assembler to consume.  Designed to run in a daemon thread
alongside the ARP spoofer so intercepted traffic is processed immediately.

Usage (library):
    import queue
    from src.capture.sniffer import PacketSniffer

    pkt_queue = queue.Queue(maxsize=10_000)
    sniffer = PacketSniffer(iface="en0", packet_queue=pkt_queue)
    sniffer.start()
    # ... read from pkt_queue in another thread ...
    sniffer.stop()

Requirements:
    - scapy
    - Root / sudo on macOS; or CAP_NET_RAW on Linux (see Phase 5 hardening).
"""

from __future__ import annotations

import logging
import queue
import threading
from typing import Optional

logger = logging.getLogger("nids.capture.sniffer")

# BPF filter: capture only IP traffic (TCP + UDP + ICMP).
# This avoids processing non-IP frames (ARP, etc.) in the flow assembler.
_BPF_FILTER = "ip and (tcp or udp or icmp)"


class PacketSniffer:
    """Scapy-based packet sniffer that feeds a queue consumed by FlowAssembler.

    Args:
        iface:        Network interface to sniff on (e.g. "en0", "eth0").
        packet_queue: Thread-safe queue to push captured packets into.
        max_queue_size: Drop packets (warn) when queue exceeds this size.
                        Set to 0 (default) to never drop.
    """

    def __init__(
        self,
        iface: str,
        packet_queue: queue.Queue,
        max_queue_size: int = 0,
    ) -> None:
        self.iface = iface
        self.packet_queue = packet_queue
        self.max_queue_size = max_queue_size

        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._packets_captured: int = 0
        self._packets_dropped: int = 0

    # ── Public API ────────────────────────────────────────────────────────────

    def start(self) -> None:
        """Start sniffing in a daemon background thread.

        Returns immediately. Packets are enqueued as they arrive.

        Raises:
            RuntimeError: if scapy is not installed.
            PermissionError: if the process lacks raw-socket privileges.
        """
        try:
            import scapy.all  # noqa: F401 — validate install before spawning thread
        except ImportError:
            raise RuntimeError(
                "scapy is not installed. Run: pip install scapy"
            )

        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._sniff_loop,
            daemon=True,
            name=f"nids-sniffer-{self.iface}",
        )
        self._thread.start()
        logger.info(
            "Sniffer started on interface '%s' (filter: %s)", self.iface, _BPF_FILTER
        )

    def stop(self) -> None:
        """Signal the sniffer to stop and wait for the thread to exit.

        Safe to call multiple times.
        """
        if not self._stop_event.is_set():
            logger.info(
                "Sniffer stopping — captured=%d dropped=%d",
                self._packets_captured,
                self._packets_dropped,
            )
            self._stop_event.set()
            # Scapy's sniff() checks the stop_filter on each packet; we may
            # need to wait for the next packet to arrive before it exits.
            if self._thread and self._thread.is_alive():
                self._thread.join(timeout=5.0)

    @property
    def packets_captured(self) -> int:
        return self._packets_captured

    @property
    def packets_dropped(self) -> int:
        return self._packets_dropped

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # ── Internal ──────────────────────────────────────────────────────────────

    def _sniff_loop(self) -> None:
        """Run scapy.sniff() in this thread, routing packets to the queue."""
        from scapy.sendrecv import sniff

        try:
            sniff(
                iface=self.iface,
                filter=_BPF_FILTER,
                prn=self._on_packet,
                store=False,               # do NOT accumulate in memory
                stop_filter=self._should_stop,
            )
        except PermissionError as exc:
            logger.error(
                "Permission denied on %s: %s — "
                "Run with sudo or grant CAP_NET_RAW to the Python binary.",
                self.iface, exc,
            )
        except Exception as exc:
            if not self._stop_event.is_set():
                logger.error("Sniffer error on %s: %s", self.iface, exc)

    def _on_packet(self, pkt) -> None:
        """Called by Scapy for each captured packet."""
        self._packets_captured += 1

        if self.max_queue_size and self.packet_queue.qsize() >= self.max_queue_size:
            self._packets_dropped += 1
            if self._packets_dropped % 1000 == 1:
                logger.warning(
                    "Packet queue full (%d) — dropped %d packets. "
                    "Consider increasing max_queue_size or reducing capture rate.",
                    self.max_queue_size, self._packets_dropped,
                )
            return

        try:
            self.packet_queue.put_nowait(pkt)
        except queue.Full:
            self._packets_dropped += 1

    def _should_stop(self, pkt) -> bool:
        """Scapy stop_filter: returns True when stop() has been called."""
        return self._stop_event.is_set()


# ── Interface utilities ───────────────────────────────────────────────────────

def list_interfaces() -> list[dict[str, str]]:
    """Return available network interfaces with their IP addresses.

    Returns a list of dicts: [{"name": "en0", "ip": "192.168.1.5"}, ...]
    Falls back to a basic OS-level list if netifaces is not installed.
    """
    interfaces: list[dict[str, str]] = []

    try:
        import netifaces
        for iface in netifaces.interfaces():
            addrs = netifaces.ifaddresses(iface)
            ip = ""
            if netifaces.AF_INET in addrs:
                ip = addrs[netifaces.AF_INET][0].get("addr", "")
            if not iface.startswith("lo"):
                interfaces.append({"name": iface, "ip": ip})
        return interfaces
    except ImportError:
        pass

    # Minimal fallback using scapy
    try:
        from scapy.interfaces import get_if_list
        return [{"name": iface, "ip": ""} for iface in get_if_list()
                if not iface.startswith("lo")]
    except ImportError:
        pass

    return [{"name": "en0", "ip": ""}]
