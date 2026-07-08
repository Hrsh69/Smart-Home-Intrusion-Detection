"""Tests for the alert system."""

from __future__ import annotations

import sys
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.alerts import AlertManager


@pytest.fixture
def alert_mgr():
    """Create an AlertManager with a mocked database."""
    mock_db = MagicMock()
    mock_db.insert_alert.return_value = 1
    mgr = AlertManager(db=mock_db)
    mgr._cooldown = 0  # Disable rate limiting for tests
    return mgr


class TestSeverityRouting:
    """Test that alerts route correctly based on severity."""

    def test_info_no_alert(self, alert_mgr):
        sent = alert_mgr.process_alert(1, "BENIGN", 0.99, "Info")
        assert sent is False

    def test_low_no_alert(self, alert_mgr):
        sent = alert_mgr.process_alert(1, "Recon", 0.7, "Low")
        assert sent is False

    @patch("src.alerts.SETTINGS")
    def test_medium_desktop_only(self, mock_settings, alert_mgr):
        mock_settings.DESKTOP_ALERTS_ENABLED = True
        mock_settings.EMAIL_ALERTS_ENABLED = False
        mock_settings.TELEGRAM_ALERTS_ENABLED = False

        with patch.object(alert_mgr, "_send_desktop") as mock_desktop:
            sent = alert_mgr.process_alert(1, "Spoofing", 0.6, "Medium")
            assert sent is True
            mock_desktop.assert_called_once()

    @patch("src.alerts.SETTINGS")
    def test_high_desktop_and_email(self, mock_settings, alert_mgr):
        mock_settings.DESKTOP_ALERTS_ENABLED = True
        mock_settings.EMAIL_ALERTS_ENABLED = True
        mock_settings.SMTP_USER = "test@test.com"
        mock_settings.TELEGRAM_ALERTS_ENABLED = False

        with patch.object(alert_mgr, "_send_desktop") as mock_desktop, \
             patch.object(alert_mgr, "_send_email") as mock_email:
            sent = alert_mgr.process_alert(1, "BruteForce", 0.8, "High")
            assert sent is True
            mock_desktop.assert_called_once()
            mock_email.assert_called_once()

    @patch("src.alerts.SETTINGS")
    def test_critical_all_channels(self, mock_settings, alert_mgr):
        mock_settings.DESKTOP_ALERTS_ENABLED = True
        mock_settings.EMAIL_ALERTS_ENABLED = True
        mock_settings.SMTP_USER = "test@test.com"
        mock_settings.TELEGRAM_ALERTS_ENABLED = True
        mock_settings.TELEGRAM_BOT_TOKEN = "fake_token"

        with patch.object(alert_mgr, "_send_desktop") as mock_desktop, \
             patch.object(alert_mgr, "_send_email") as mock_email, \
             patch.object(alert_mgr, "_send_telegram") as mock_telegram:
            sent = alert_mgr.process_alert(1, "DDoS", 0.95, "Critical")
            assert sent is True
            mock_desktop.assert_called_once()
            mock_email.assert_called_once()
            mock_telegram.assert_called_once()


class TestRateLimiting:
    """Test alert rate limiting."""

    def test_rate_limit_blocks_repeated(self):
        mock_db = MagicMock()
        mock_db.insert_alert.return_value = 1
        mgr = AlertManager(db=mock_db)
        mgr._cooldown = 60  # 60 second cooldown

        with patch("src.alerts.SETTINGS") as mock_settings:
            mock_settings.DESKTOP_ALERTS_ENABLED = True
            mock_settings.EMAIL_ALERTS_ENABLED = False
            mock_settings.TELEGRAM_ALERTS_ENABLED = False

            with patch.object(mgr, "_send_desktop"):
                # First alert should go through
                result1 = mgr.process_alert(1, "DDoS", 0.9, "Critical")

                # Second immediate alert for same category should be blocked
                result2 = mgr.process_alert(2, "DDoS", 0.85, "Critical")

                assert result1 is True
                assert result2 is False

    def test_rate_limit_allows_different_categories(self):
        mock_db = MagicMock()
        mock_db.insert_alert.return_value = 1
        mgr = AlertManager(db=mock_db)
        mgr._cooldown = 60

        with patch("src.alerts.SETTINGS") as mock_settings:
            mock_settings.DESKTOP_ALERTS_ENABLED = True
            mock_settings.EMAIL_ALERTS_ENABLED = False
            mock_settings.TELEGRAM_ALERTS_ENABLED = False

            with patch.object(mgr, "_send_desktop"):
                result1 = mgr.process_alert(1, "DDoS", 0.9, "Critical")
                result2 = mgr.process_alert(2, "Mirai", 0.85, "Critical")

                assert result1 is True
                assert result2 is True


class TestMessageBuilding:
    """Test alert message formatting."""

    def test_message_contains_prediction(self, alert_mgr):
        msg = alert_mgr._build_message("DDoS", 0.95, "Critical", "192.168.1.100")
        assert "DDoS" in msg
        assert "Critical" in msg
        assert "95.0%" in msg
        assert "192.168.1.100" in msg

    def test_message_without_ip(self, alert_mgr):
        msg = alert_mgr._build_message("Mirai", 0.8, "Critical", None)
        assert "Mirai" in msg
        assert "Source IP" not in msg


class TestAlertLogging:
    """Test that alerts are logged to the database."""

    @patch("src.alerts.SETTINGS")
    def test_alert_logged_to_db(self, mock_settings):
        mock_db = MagicMock()
        mock_db.insert_alert.return_value = 1
        mgr = AlertManager(db=mock_db)
        mgr._cooldown = 0

        mock_settings.DESKTOP_ALERTS_ENABLED = True
        mock_settings.EMAIL_ALERTS_ENABLED = False
        mock_settings.TELEGRAM_ALERTS_ENABLED = False

        with patch.object(mgr, "_send_desktop"):
            mgr.process_alert(1, "DDoS", 0.9, "Critical")

        mock_db.insert_alert.assert_called_once()
