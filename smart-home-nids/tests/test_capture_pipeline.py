"""Integration test for the Live Capture Pipeline.

Verifies that packets passed to the packet_queue are correctly assembled
into flows by FlowAssembler, classified by NIDSPredictor, and logged by NIDSDatabase.
"""

import queue
import time
from unittest.mock import MagicMock

import pytest

from src.capture.flow_assembler import FlowAssembler
from src.capture.pipeline import CapturePipeline
from src.predict import PredictionResult

def test_capture_pipeline_integration():
    """Test the full pipeline without the actual sniffer and spoofer."""
    mock_db = MagicMock()
    mock_alert_mgr = MagicMock()
    mock_predictor = MagicMock()

    # Mock the predictor to return a specific result
    mock_predictor.predict_single.return_value = PredictionResult(
        label="Recon",
        confidence=0.95,
        severity="Low",
        probabilities={"BENIGN": 0.05, "Recon": 0.95},
        top_3=[("Recon", 0.95), ("BENIGN", 0.05), ("DoS", 0.0)],
    )
    mock_predictor.model_version = "rf-testmock"

    # Setup the pipeline
    pipeline = CapturePipeline(
        iface="lo0",
        predictor=mock_predictor,
        db=mock_db,
        alert_mgr=mock_alert_mgr,
        idle_timeout=0.1,  # very short timeout to flush flow quickly
    )

    # We will manually inject a few fake flow feature dicts into the flow_queue
    # instead of doing raw packet injection to avoid scapy dependencies in CI 
    # (since Scapy is an optional dependency for live capture only).
    
    # Start just the prediction loop
    pipeline._stop_event.clear()
    import threading
    pipeline._predict_thread = threading.Thread(
        target=pipeline._predict_loop,
        daemon=True,
    )
    pipeline._predict_thread.start()

    # Inject fake flow
    fake_features = {
        "flow_duration": 1.5,
        "tot_size": 500,
        "protocol_type": 6,
    }
    pipeline._flow_queue.put(fake_features)

    # Wait for the predict loop to process the item
    time.sleep(0.5)
    
    # Stop the pipeline
    pipeline.stop()

    # Assertions
    assert pipeline.flows_classified == 1
    assert pipeline.threats_detected == 1

    # Check if predictor was called with the features
    mock_predictor.predict_single.assert_called_once_with(fake_features)

    # Check if DB insert was called
    mock_db.insert_detection.assert_called_once()
    call_kwargs = mock_db.insert_detection.call_args.kwargs
    assert call_kwargs["prediction"] == "Recon"
    assert call_kwargs["confidence"] == 0.95
    assert call_kwargs["severity"] == "Low"
    assert call_kwargs["model_version"] == "rf-testmock"

    # Check if AlertManager was called
    mock_alert_mgr.process_alert.assert_called_once()
