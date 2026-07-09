"""src/capture — Live packet capture subsystem.

Modules:
    arp_spoof      — ARP spoofing engine (poison + restore)
    sniffer        — Scapy packet sniffer → queue
    flow_assembler — 5-tuple flow tracker → 18 CICFlowMeter features (Phase 3)
    pipeline       — Orchestrates all above + predictor + DB (Phase 4)
"""
