"""Live flow assembler — converts raw Scapy packets into CICFlowMeter-style feature vectors.

Aggregates packets into bidirectional 5-tuple flows and computes the 18 features
that exactly match the training data schema (selected_features.pkl):

    iat, avg, header_length, header_bytes_per_packet, rst_count, tot_size, max,
    tot_sum, flow_duration, urg_count, rate, bytes_per_packet, variance,
    packets_per_second, protocol_type, min, rst_ratio, urg_ratio

Flow timeout:
    A flow is considered complete (emitted) when it has been idle for
    FLOW_IDLE_TIMEOUT seconds.  CICFlowMeter uses 120 s by default; this
    module matches that.  Adjust via FlowAssembler(idle_timeout=...).

Thread safety:
    FlowAssembler.process_packet() is called from the sniffer thread.
    Completed flows are placed in ``flow_queue`` which is consumed by the
    prediction pipeline thread.  The internal flow table is protected by a
    threading.Lock.

Usage:
    import queue, threading
    from src.capture.flow_assembler import FlowAssembler

    pkt_queue  = queue.Queue()   # fed by PacketSniffer
    flow_queue = queue.Queue()   # consumed by CapturePipeline / predict

    assembler = FlowAssembler(
        packet_queue=pkt_queue,
        flow_queue=flow_queue,
    )
    assembler.start()
    # ... sniffer puts packets in pkt_queue ...
    # ... flow_queue yields dict[str, float] feature vectors ...
    assembler.stop()
"""

from __future__ import annotations

import logging
import math
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger("nids.capture.flow_assembler")

# ── Constants ────────────────────────────────────────────────────────────────

#: Flow idle timeout in seconds — matches CICFlowMeter default.
FLOW_IDLE_TIMEOUT: float = 120.0

#: Maximum packets per flow before forcing emission (memory guard).
MAX_PACKETS_PER_FLOW: int = 50_000

#: How often (seconds) the flush thread checks for timed-out flows.
FLUSH_INTERVAL: float = 5.0


# ── Data structures ──────────────────────────────────────────────────────────

@dataclass
class FlowKey:
    """Bidirectional 5-tuple flow identifier.

    Flows are bidirectional: (A→B, TCP) and (B→A, TCP) belong to the same
    flow.  The key is normalised so (src,dst,sp,dp,proto) == (dst,src,dp,sp,proto).
    """
    src_ip: str
    dst_ip: str
    src_port: int
    dst_port: int
    proto: int  # IP protocol number: 6=TCP, 17=UDP, 1=ICMP

    @classmethod
    def from_packet(cls, ip_src, ip_dst, sport, dport, proto) -> "FlowKey":
        # Normalise so the smaller (ip, port) tuple is always "src"
        a = (ip_src, sport)
        b = (ip_dst, dport)
        if a <= b:
            return cls(ip_src, ip_dst, sport, dport, proto)
        else:
            return cls(ip_dst, ip_src, dport, sport, proto)

    def __hash__(self):
        return hash((self.src_ip, self.dst_ip, self.src_port, self.dst_port, self.proto))

    def __eq__(self, other):
        return (
            self.src_ip == other.src_ip
            and self.dst_ip == other.dst_ip
            and self.src_port == other.src_port
            and self.dst_port == other.dst_port
            and self.proto == other.proto
        )


@dataclass
class FlowRecord:
    """Accumulates per-packet statistics for one network flow."""

    key: FlowKey
    start_time: float = field(default_factory=time.monotonic)
    last_time: float = field(default_factory=time.monotonic)

    # Packet arrival timestamps (for IAT calculation)
    timestamps: list[float] = field(default_factory=list)

    # Payload sizes (bytes, not including IP/TCP headers)
    packet_sizes: list[int] = field(default_factory=list)

    # Total header bytes (IP header + transport header)
    total_header_bytes: int = 0

    # TCP flag counters
    rst_count: int = 0
    urg_count: int = 0
    syn_count: int = 0
    ack_count: int = 0
    fin_count: int = 0

    def update(
        self,
        now: float,
        payload_size: int,
        header_bytes: int,
        tcp_flags: Optional[int],
    ) -> None:
        """Add one packet to this flow record."""
        self.timestamps.append(now)
        self.last_time = now
        self.packet_sizes.append(payload_size)
        self.total_header_bytes += header_bytes

        if tcp_flags is not None:
            if tcp_flags & 0x04:  # RST
                self.rst_count += 1
            if tcp_flags & 0x20:  # URG
                self.urg_count += 1
            if tcp_flags & 0x02:  # SYN
                self.syn_count += 1
            if tcp_flags & 0x10:  # ACK
                self.ack_count += 1
            if tcp_flags & 0x01:  # FIN
                self.fin_count += 1

    @property
    def packet_count(self) -> int:
        return len(self.packet_sizes)

    @property
    def idle_time(self) -> float:
        return time.monotonic() - self.last_time

    @property
    def flow_duration(self) -> float:
        if len(self.timestamps) < 2:
            return 0.0
        return self.timestamps[-1] - self.timestamps[0]


# ── Feature extraction ────────────────────────────────────────────────────────

def extract_features(flow: FlowRecord) -> dict[str, float]:
    """Compute the 18 training-schema features from a completed FlowRecord.

    Feature order matches selected_features.pkl exactly:
        iat, avg, header_length, header_bytes_per_packet, rst_count, tot_size,
        max, tot_sum, flow_duration, urg_count, rate, bytes_per_packet, variance,
        packets_per_second, protocol_type, min, rst_ratio, urg_ratio

    All divisions are guarded against zero denominators.
    """
    n = flow.packet_count
    sizes = flow.packet_sizes
    dur = flow.flow_duration
    eps = 1e-9  # zero-guard

    tot_size = float(sum(sizes))

    # Inter-arrival times (IAT)
    if len(flow.timestamps) >= 2:
        iats = [
            flow.timestamps[i] - flow.timestamps[i - 1]
            for i in range(1, len(flow.timestamps))
        ]
        mean_iat = sum(iats) / len(iats)
    else:
        mean_iat = 0.0

    avg_size = tot_size / max(n, 1)
    max_size = float(max(sizes)) if sizes else 0.0
    min_size = float(min(sizes)) if sizes else 0.0

    # Variance of packet sizes
    if n > 1:
        variance = sum((s - avg_size) ** 2 for s in sizes) / (n - 1)
    else:
        variance = 0.0

    header_bytes_per_pkt = flow.total_header_bytes / max(n, 1)
    rate = n / max(dur, eps)
    bytes_per_pkt = tot_size / max(n, 1)
    pps = n / max(dur, eps)
    rst_ratio = flow.rst_count / max(n, 1)
    urg_ratio = flow.urg_count / max(n, 1)

    return {
        "iat":                    round(mean_iat, 6),
        "avg":                    round(avg_size, 4),
        "header_length":          float(flow.total_header_bytes),
        "header_bytes_per_packet": round(header_bytes_per_pkt, 4),
        "rst_count":              float(flow.rst_count),
        "tot_size":               round(tot_size, 2),
        "max":                    max_size,
        "tot_sum":                round(tot_size, 2),   # alias for tot_size
        "flow_duration":          round(dur, 6),
        "urg_count":              float(flow.urg_count),
        "rate":                   round(rate, 4),
        "bytes_per_packet":       round(bytes_per_pkt, 4),
        "variance":               round(variance, 4),
        "packets_per_second":     round(pps, 4),
        "protocol_type":          float(flow.key.proto),
        "min":                    min_size,
        "rst_ratio":              round(rst_ratio, 6),
        "urg_ratio":              round(urg_ratio, 6),
    }


# ── Flow assembler ─────────────────────────────────────────────────────────────

class FlowAssembler:
    """Consumes raw Scapy packets and emits completed flows as feature dicts.

    Args:
        packet_queue:   Queue of Scapy IP packets (from PacketSniffer).
        flow_queue:     Output queue for completed flow feature dicts.
        idle_timeout:   Seconds of inactivity before a flow is emitted.
        max_pkt:        Max packets per flow before forcing emission.
        flush_interval: How often the housekeeping thread checks for timeouts.
    """

    def __init__(
        self,
        packet_queue: queue.Queue,
        flow_queue: queue.Queue,
        idle_timeout: float = FLOW_IDLE_TIMEOUT,
        max_pkt: int = MAX_PACKETS_PER_FLOW,
        flush_interval: float = FLUSH_INTERVAL,
    ) -> None:
        self.packet_queue = packet_queue
        self.flow_queue = flow_queue
        self.idle_timeout = idle_timeout
        self.max_pkt = max_pkt
        self.flush_interval = flush_interval

        self._flows: dict[FlowKey, FlowRecord] = {}
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._pkt_thread: Optional[threading.Thread] = None
        self._flush_thread: Optional[threading.Thread] = None

        self._flows_completed: int = 0
        self._packets_processed: int = 0

    # ── Public API ────────────────────────────────────────────────────────────

    def start(self) -> None:
        """Start packet processing and flush threads."""
        self._stop_event.clear()
        self._pkt_thread = threading.Thread(
            target=self._packet_loop, daemon=True, name="flow-assembler-pkt"
        )
        self._flush_thread = threading.Thread(
            target=self._flush_loop, daemon=True, name="flow-assembler-flush"
        )
        self._pkt_thread.start()
        self._flush_thread.start()
        logger.info(
            "FlowAssembler started (idle_timeout=%.0fs, max_pkt=%d)",
            self.idle_timeout, self.max_pkt,
        )

    def stop(self) -> None:
        """Stop threads and flush all remaining active flows."""
        if not self._stop_event.is_set():
            self._stop_event.set()
            for t in (self._pkt_thread, self._flush_thread):
                if t and t.is_alive():
                    t.join(timeout=self.flush_interval + 2)
            self._flush_all()
            logger.info(
                "FlowAssembler stopped — flows_completed=%d packets=%d",
                self._flows_completed,
                self._packets_processed,
            )

    @property
    def active_flow_count(self) -> int:
        with self._lock:
            return len(self._flows)

    @property
    def flows_completed(self) -> int:
        return self._flows_completed

    # ── Packet processing ─────────────────────────────────────────────────────

    def _packet_loop(self) -> None:
        """Drain the packet queue and update flow records."""
        while not self._stop_event.is_set():
            try:
                pkt = self.packet_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            try:
                self._process_packet(pkt)
                self._packets_processed += 1
            except Exception as exc:
                logger.debug("Packet processing error: %s", exc)

    def _process_packet(self, pkt) -> None:
        """Extract flow key + stats from a Scapy IP packet and update table."""
        from scapy.layers.inet import IP, TCP, UDP, ICMP

        if not pkt.haslayer(IP):
            return

        ip = pkt[IP]
        proto = ip.proto  # 6=TCP, 17=UDP, 1=ICMP

        # Extract ports
        sport, dport = 0, 0
        tcp_flags: Optional[int] = None
        payload_size = len(ip.payload)
        header_bytes = len(ip) - payload_size  # IP header length

        if proto == 6 and pkt.haslayer(TCP):
            tcp = pkt[TCP]
            sport, dport = tcp.sport, tcp.dport
            tcp_flags = int(tcp.flags)
            header_bytes += len(tcp)  # include TCP header
            payload_size = max(0, len(tcp.payload))
        elif proto == 17 and pkt.haslayer(UDP):
            udp = pkt[UDP]
            sport, dport = udp.sport, udp.dport
            header_bytes += 8  # UDP header is always 8 bytes
            payload_size = max(0, len(udp.payload))
        elif proto == 1 and pkt.haslayer(ICMP):
            header_bytes += 8  # ICMP header is 8 bytes

        key = FlowKey.from_packet(
            ip_src=ip.src, ip_dst=ip.dst,
            sport=sport, dport=dport, proto=proto,
        )
        now = time.monotonic()

        with self._lock:
            if key not in self._flows:
                self._flows[key] = FlowRecord(key=key, start_time=now, last_time=now)

            flow = self._flows[key]
            flow.update(now, payload_size, header_bytes, tcp_flags)

            # Emit if flow has too many packets
            if flow.packet_count >= self.max_pkt:
                self._emit_flow(key, flow)

    # ── Flush logic ───────────────────────────────────────────────────────────

    def _flush_loop(self) -> None:
        """Periodically emit flows that have been idle past the timeout."""
        while not self._stop_event.wait(timeout=self.flush_interval):
            self._flush_timed_out()

    def _flush_timed_out(self) -> None:
        timed_out: list[FlowKey] = []
        with self._lock:
            for key, flow in self._flows.items():
                if flow.idle_time >= self.idle_timeout:
                    timed_out.append(key)

        with self._lock:
            for key in timed_out:
                if key in self._flows:
                    self._emit_flow(key, self._flows[key])

        if timed_out:
            logger.debug("Flushed %d timed-out flows.", len(timed_out))

    def _flush_all(self) -> None:
        """Emit all remaining active flows (called at shutdown)."""
        with self._lock:
            keys = list(self._flows.keys())
            for key in keys:
                if key in self._flows:
                    self._emit_flow(key, self._flows[key])
        logger.debug("Flushed all remaining flows at shutdown.")

    def _emit_flow(self, key: FlowKey, flow: FlowRecord) -> None:
        """Compute features for a completed flow and put them in flow_queue.

        Must be called with self._lock held.
        """
        if flow.packet_count == 0:
            del self._flows[key]
            return

        features = extract_features(flow)
        del self._flows[key]
        self._flows_completed += 1

        try:
            self.flow_queue.put_nowait(features)
        except queue.Full:
            logger.warning("Flow queue full — discarding completed flow.")

        logger.debug(
            "Flow emitted: %s:%d→%s:%d proto=%d pkts=%d dur=%.2fs",
            key.src_ip, key.src_port, key.dst_ip, key.dst_port,
            key.proto, flow.packet_count, flow.flow_duration,
        )
