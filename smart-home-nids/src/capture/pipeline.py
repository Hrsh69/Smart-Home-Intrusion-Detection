"""Capture pipeline — orchestrates ARP spoof + sniffer + flow assembler + predictor.

Wires together:
  1. ArpSpoofer  — optional MITM (can be skipped if traffic already flows through)
  2. PacketSniffer — Scapy packet capture into packet_queue
  3. FlowAssembler — aggregates packets into CICFlowMeter-style feature dicts
  4. NIDSPredictor — classifies each completed flow
  5. NIDSDatabase + AlertManager — log every result

The pipeline exposes a result_queue that the dashboard polls for real-time updates.

Usage:
    from src.capture.pipeline import CapturePipeline
    from src.predict import NIDSPredictor
    from src.database import NIDSDatabase
    from src.alerts import AlertManager

    pipeline = CapturePipeline(
        iface="en0",
        predictor=predictor,
        db=db,
        alert_mgr=alert_mgr,
        targets=["192.168.1.10"],   # optional — skip ARP spoofing if empty
        gateway="192.168.1.1",      # required if targets provided
    )
    pipeline.start()
    # result_queue yields dicts for the dashboard
    # pipeline.stop() on shutdown
"""

from __future__ import annotations

import collections
import logging
import queue
import threading
from typing import Optional

from src.capture.flow_assembler import FlowAssembler
from src.capture.sniffer import PacketSniffer

logger = logging.getLogger("nids.capture.pipeline")


class CapturePipeline:
    """End-to-end capture → classify → log pipeline.

    Args:
        iface:       Network interface to sniff on.
        predictor:   Loaded NIDSPredictor instance.
        db:          NIDSDatabase for logging detections.
        alert_mgr:   AlertManager for notifications.
        targets:     Optional list of IPs to ARP-spoof (requires root/sudo).
        gateway:     Gateway IP (required when targets is non-empty).
        idle_timeout: Flow idle timeout in seconds (default 120, matches CICFlowMeter).
        result_queue: Optional external queue to receive classification results.
                      If None, an internal queue is created.
    """

    def __init__(
        self,
        iface: str,
        predictor,
        db,
        alert_mgr,
        targets: Optional[list[str]] = None,
        gateway: Optional[str] = None,
        idle_timeout: float = 120.0,
        result_queue: Optional[queue.Queue] = None,
    ) -> None:
        self.iface = iface
        self.predictor = predictor
        self.db = db
        self.alert_mgr = alert_mgr
        self.targets = targets or []
        self.gateway = gateway
        self.idle_timeout = idle_timeout

        # Internal queues
        self._packet_queue: queue.Queue = queue.Queue(maxsize=20_000)
        self._flow_queue: queue.Queue = queue.Queue(maxsize=5_000)
        self.result_queue: queue.Queue = result_queue or queue.Queue(maxsize=1_000)
        self.packet_log: collections.deque = collections.deque(maxlen=1000)

        # Component instances
        self._spoofer = None
        self._sniffer: Optional[PacketSniffer] = None
        self._assembler: Optional[FlowAssembler] = None
        self._predict_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        # Stats
        self.flows_classified: int = 0
        self.threats_detected: int = 0

    # ── Public API ────────────────────────────────────────────────────────────

    def start(self) -> None:
        """Start all pipeline components in the correct order.

        Component start order:
            1. ARP spoofer (if targets provided) — begin MITM
            2. FlowAssembler — ready to consume packets before sniffer starts
            3. PacketSniffer — begins filling packet_queue
            4. Prediction thread — consumes flow_queue
        """
        logger.info(
            "CapturePipeline starting — iface=%s targets=%s",
            self.iface, self.targets,
        )
        self._stop_event.clear()

        # 1. Optional ARP spoofing
        if self.targets:
            if not self.gateway:
                raise ValueError("gateway is required when targets are specified.")
            from src.capture.arp_spoof import ArpSpoofer
            self._spoofer = ArpSpoofer(
                targets=self.targets,
                gateway=self.gateway,
                iface=self.iface,
            )
            self._spoofer.start()

        # 2. Flow assembler
        self._assembler = FlowAssembler(
            packet_queue=self._packet_queue,
            flow_queue=self._flow_queue,
            idle_timeout=self.idle_timeout,
        )
        self._assembler.start()

        # 3. Packet sniffer
        self._sniffer = PacketSniffer(
            iface=self.iface,
            packet_queue=self._packet_queue,
            packet_log=self.packet_log,
            max_queue_size=20_000,
        )
        self._sniffer.start()

        # 4. Prediction consumer thread
        self._predict_thread = threading.Thread(
            target=self._predict_loop,
            daemon=True,
            name="nids-predict",
        )
        self._predict_thread.start()

        logger.info("CapturePipeline started successfully.")

    def stop(self) -> None:
        """Stop all pipeline components in reverse order. Safe to call multiple times."""
        if not self._stop_event.is_set():
            logger.info("CapturePipeline stopping…")
            self._stop_event.set()

            # Stop in reverse order: sniffer → assembler → predict → spoofer
            if self._sniffer:
                self._sniffer.stop()
            if self._assembler:
                self._assembler.stop()
            if self._predict_thread and self._predict_thread.is_alive():
                self._predict_thread.join(timeout=5.0)
            if self._spoofer:
                self._spoofer.stop()  # ARP restore — always last

            logger.info(
                "CapturePipeline stopped — classified=%d threats=%d",
                self.flows_classified, self.threats_detected,
            )

    @property
    def is_running(self) -> bool:
        return not self._stop_event.is_set()

    @property
    def sniffer_error(self) -> Optional[str]:
        return self._sniffer.error_message if self._sniffer else None

    @property
    def packets_captured(self) -> int:
        return self._sniffer.packets_captured if self._sniffer else 0

    @property
    def packets_per_second(self) -> float:
        return self._sniffer.packets_per_second if self._sniffer else 0.0

    @property
    def bytes_per_second(self) -> float:
        return self._sniffer.bytes_per_second if self._sniffer else 0.0

    # ── Prediction loop ───────────────────────────────────────────────────────

    def _predict_loop(self) -> None:
        """Consume flow_queue → classify → log → push to result_queue."""
        while not self._stop_event.is_set():
            try:
                features: dict = self._flow_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            try:
                result = self.predictor.predict_single(features)
                self.flows_classified += 1
                if result.label != "BENIGN":
                    self.threats_detected += 1

                # Log to database
                det_id = self.db.insert_detection(
                    prediction=result.label,
                    confidence=result.confidence,
                    severity=result.severity,
                    features=features,
                    source_ip=None,   # populated by flow if src_ip tracked
                    protocol=str(int(features.get("protocol_type", 0))),
                    model_version=self.predictor.model_version,
                )

                # Send alerts
                self.alert_mgr.process_alert(
                    det_id,
                    result.label,
                    result.confidence,
                    result.severity,
                    source_ip=None,
                )

                # Push to dashboard result queue
                result_record = {
                    "prediction": result.label,
                    "confidence": result.confidence,
                    "severity": result.severity,
                    "probabilities": result.probabilities,
                    "top_3": result.top_3,
                    "features": features,
                    "det_id": det_id,
                }
                try:
                    self.result_queue.put_nowait(result_record)
                except queue.Full:
                    # Dashboard is not polling fast enough — discard oldest
                    try:
                        self.result_queue.get_nowait()
                        self.result_queue.put_nowait(result_record)
                    except queue.Empty:
                        pass

                logger.info(
                    "Flow classified: %s (%.1f%%) [%s]",
                    result.label, result.confidence * 100, result.severity,
                )

            except Exception as exc:
                logger.error("Prediction error: %s", exc, exc_info=True)
