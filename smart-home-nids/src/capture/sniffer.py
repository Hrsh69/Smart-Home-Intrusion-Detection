"""Packet sniffer for Smart Home NIDS.

Captures IP packets on a network interface using Scapy and puts them into a
queue for the flow assembler to consume.  Also maintains a lightweight
packet_log (deque) of per-packet metadata for the Wireshark-style UI.

Usage (library):
    import queue, collections
    from src.capture.sniffer import PacketSniffer

    pkt_queue = queue.Queue(maxsize=10_000)
    pkt_log = collections.deque(maxlen=1000)
    sniffer = PacketSniffer(iface="en0", packet_queue=pkt_queue, packet_log=pkt_log)
    sniffer.start()
    # ... read from pkt_queue in another thread ...
    # ... read pkt_log snapshot for the UI ...
    sniffer.stop()

Requirements:
    - scapy
    - Root / sudo on macOS; or CAP_NET_RAW on Linux.
"""

from __future__ import annotations

import collections
import logging
import queue
import threading
import time
from datetime import datetime
from typing import Optional

logger = logging.getLogger("nids.capture.sniffer")

# BPF filter: capture only IP traffic (TCP + UDP + ICMP).
_BPF_FILTER = "ip and (tcp or udp or icmp)"

# Protocol number → display name
_PROTO_NAMES = {1: "ICMP", 6: "TCP", 17: "UDP"}


class PacketSniffer:
    """Scapy-based packet sniffer with a per-packet metadata log for the UI.

    Args:
        iface:          Network interface to sniff on (e.g. "en0", "eth0").
        packet_queue:   Thread-safe queue to push raw Scapy packets into
                        (consumed by FlowAssembler).
        packet_log:     Optional deque that receives lightweight per-packet
                        metadata dicts for the dashboard's packet table.
                        Append is O(1) and thread-safe; old entries auto-evict.
        max_queue_size: Drop packets (warn) when queue exceeds this size.
    """

    def __init__(
        self,
        iface: str,
        packet_queue: queue.Queue,
        packet_log: Optional[collections.deque] = None,
        max_queue_size: int = 0,
    ) -> None:
        self.iface = iface
        self.packet_queue = packet_queue
        self.packet_log = packet_log
        self.max_queue_size = max_queue_size

        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._packets_captured: int = 0
        self._packets_dropped: int = 0
        self._bytes_total: int = 0
        self._start_time: float = 0.0

        # Surfaced to the dashboard if a capture error occurs
        self.error_message: Optional[str] = None

    # ── Public API ────────────────────────────────────────────────────────────

    def start(self) -> None:
        """Start sniffing in a daemon background thread."""
        try:
            import scapy.all  # noqa: F401
        except ImportError:
            raise RuntimeError("scapy is not installed. Run: pip install scapy")

        self._stop_event.clear()
        self.error_message = None
        self._start_time = time.monotonic()
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
        """Signal the sniffer to stop and wait for the thread to exit."""
        if not self._stop_event.is_set():
            logger.info(
                "Sniffer stopping — captured=%d dropped=%d",
                self._packets_captured,
                self._packets_dropped,
            )
            self._stop_event.set()
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

    @property
    def packets_per_second(self) -> float:
        elapsed = time.monotonic() - self._start_time if self._start_time else 0
        if elapsed <= 0:
            return 0.0
        return self._packets_captured / elapsed

    @property
    def bytes_per_second(self) -> float:
        elapsed = time.monotonic() - self._start_time if self._start_time else 0
        if elapsed <= 0:
            return 0.0
        return self._bytes_total / elapsed

    @property
    def bytes_total(self) -> int:
        return self._bytes_total

    # ── Privilege check ───────────────────────────────────────────────────────

    @staticmethod
    def check_capture_permission(iface: str = "en0") -> bool:
        """Test whether we can open a raw socket on the given interface.

        Attempts a 0.5 s sniff. Returns True if it succeeds (even with
        zero packets), False on PermissionError or OSError.
        """
        try:
            from scapy.sendrecv import sniff
            sniff(iface=iface, filter=_BPF_FILTER, count=1, timeout=0.5, store=False)
            return True
        except (PermissionError, OSError):
            return False
        except Exception:
            return False

    # ── Internal ──────────────────────────────────────────────────────────────

    def _sniff_loop(self) -> None:
        """Run scapy.sniff() in this thread, routing packets to the queue."""
        from scapy.sendrecv import sniff

        try:
            sniff(
                iface=self.iface,
                filter=_BPF_FILTER,
                prn=self._on_packet,
                store=False,
                stop_filter=self._should_stop,
            )
        except PermissionError as exc:
            self.error_message = (
                f"Permission denied on {self.iface}: {exc} — "
                "Run with sudo or grant CAP_NET_RAW."
            )
            logger.error(self.error_message)
        except Exception as exc:
            if not self._stop_event.is_set():
                self.error_message = f"Sniffer error on {self.iface}: {exc}"
                logger.error(self.error_message)

    def _on_packet(self, pkt) -> None:
        """Called by Scapy for each captured packet."""
        from scapy.layers.inet import IP, TCP, UDP, ICMP

        self._packets_captured += 1

        # Track total bytes
        pkt_len = len(pkt)
        self._bytes_total += pkt_len

        # ── Build lightweight metadata for the UI packet log ──────────
        if self.packet_log is not None and pkt.haslayer(IP):
            try:
                ip = pkt[IP]
                proto_num = ip.proto
                proto_name = _PROTO_NAMES.get(proto_num, f"IP/{proto_num}")
                sport, dport = 0, 0
                info = ""

                if proto_num == 6 and pkt.haslayer(TCP):
                    tcp = pkt[TCP]
                    sport, dport = tcp.sport, tcp.dport
                    flags = str(tcp.flags)
                    info = f"[{flags}] Seq={tcp.seq & 0xFFFF} Win={tcp.window}"
                    # Detect common higher-layer protocols
                    if dport == 443 or sport == 443:
                        proto_name = "TLS"
                    elif dport == 80 or sport == 80:
                        proto_name = "HTTP"
                    elif dport == 22 or sport == 22:
                        proto_name = "SSH"
                elif proto_num == 17 and pkt.haslayer(UDP):
                    udp = pkt[UDP]
                    sport, dport = udp.sport, udp.dport
                    if dport == 53 or sport == 53:
                        proto_name = "DNS"
                        # Try to extract DNS query name
                        try:
                            from scapy.layers.dns import DNS, DNSQR
                            if pkt.haslayer(DNS) and pkt.haslayer(DNSQR):
                                qname = pkt[DNSQR].qname
                                if isinstance(qname, bytes):
                                    qname = qname.decode("utf-8", errors="replace").rstrip(".")
                                info = f"Query: {qname}"
                            elif pkt.haslayer(DNS):
                                dns = pkt[DNS]
                                info = f"{'Response' if dns.qr else 'Query'} id={dns.id}"
                        except Exception:
                            info = f"DNS port {sport}→{dport}"
                    elif dport == 5353 or sport == 5353:
                        proto_name = "mDNS"
                    elif dport == 67 or dport == 68:
                        proto_name = "DHCP"
                    else:
                        info = f"Len={len(udp.payload)}"
                elif proto_num == 1 and pkt.haslayer(ICMP):
                    icmp = pkt[ICMP]
                    icmp_types = {0: "Echo Reply", 8: "Echo Request", 3: "Dest Unreachable",
                                  11: "Time Exceeded"}
                    info = icmp_types.get(icmp.type, f"Type={icmp.type} Code={icmp.code}")

                src_str = f"{ip.src}:{sport}" if sport else ip.src
                dst_str = f"{ip.dst}:{dport}" if dport else ip.dst

                self.packet_log.append({
                    "no": self._packets_captured,
                    "time": datetime.now().strftime("%H:%M:%S.%f")[:12],
                    "src": src_str,
                    "dst": dst_str,
                    "proto": proto_name,
                    "length": pkt_len,
                    "info": info[:80],  # cap info string length
                })
            except Exception:
                pass  # never let UI logging crash the capture

        # ── Enqueue raw packet for flow assembler ─────────────────────
        if self.max_queue_size and self.packet_queue.qsize() >= self.max_queue_size:
            self._packets_dropped += 1
            if self._packets_dropped % 1000 == 1:
                logger.warning(
                    "Packet queue full (%d) — dropped %d packets.",
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
