"""Multi-channel alert system for Smart Home NIDS.

Supports:
- Desktop notifications (via plyer)
- Email alerts (SMTP)
- Telegram webhook
- Rate limiting to prevent alert storms.

Usage:
    from src.alerts import AlertManager
    mgr = AlertManager(db)
    mgr.process_alert(detection_id, prediction, confidence, severity)
"""

from __future__ import annotations

import logging
import smtplib
import time
from email.mime.text import MIMEText
from typing import Optional

from config.settings import SETTINGS

logger = logging.getLogger("smart_home_nids.alerts")


class AlertManager:
    """Manages multi-channel alerting with rate limiting."""

    def __init__(self, db=None) -> None:
        self.db = db
        self._last_alert_time: dict[str, float] = {}  # category -> timestamp
        self._cooldown = SETTINGS.ALERT_COOLDOWN_SECONDS

    def process_alert(
        self,
        detection_id: int,
        prediction: str,
        confidence: float,
        severity: str,
        source_ip: Optional[str] = None,
    ) -> bool:
        """Process a detection and send alerts based on severity.

        Severity routing:
        - Critical → Desktop + Email + Telegram
        - High     → Desktop + Email
        - Medium   → Desktop only
        - Low/Info → Log only (no alert)

        Returns True if any alert was sent.
        """
        if severity in ("Info", "Low"):
            return False

        # Rate limiting
        if self._is_rate_limited(prediction):
            logger.debug("Alert rate-limited for category: %s", prediction)
            return False

        message = self._build_message(prediction, confidence, severity, source_ip)
        sent = False

        # Desktop notification
        if severity in ("Critical", "High", "Medium"):
            if SETTINGS.DESKTOP_ALERTS_ENABLED:
                self._send_desktop(prediction, message, severity)
                sent = True

        # Email
        if severity in ("Critical", "High"):
            if SETTINGS.EMAIL_ALERTS_ENABLED and SETTINGS.SMTP_USER:
                self._send_email(prediction, message, severity)
                sent = True

        # Telegram
        if severity == "Critical":
            if SETTINGS.TELEGRAM_ALERTS_ENABLED and SETTINGS.TELEGRAM_BOT_TOKEN:
                self._send_telegram(message)
                sent = True

        # Log alert to database
        if sent and self.db is not None:
            try:
                alert_type = "desktop"
                if severity == "Critical":
                    alert_type = "all_channels"
                elif severity == "High":
                    alert_type = "desktop+email"

                self.db.insert_alert(
                    detection_id=detection_id,
                    alert_type=alert_type,
                    severity=severity,
                    message=message,
                )
            except Exception as exc:
                logger.error("Failed to log alert to DB: %s", exc)

        self._last_alert_time[prediction] = time.time()
        return sent

    def _is_rate_limited(self, category: str) -> bool:
        """Check if alerts for this category are rate-limited."""
        last = self._last_alert_time.get(category, 0)
        return (time.time() - last) < self._cooldown

    def _build_message(
        self,
        prediction: str,
        confidence: float,
        severity: str,
        source_ip: Optional[str],
    ) -> str:
        """Build a human-readable alert message."""
        lines = [
            f"⚠️ NIDS ALERT — {severity} Severity",
            f"Attack Type: {prediction}",
            f"Confidence: {confidence:.1%}",
        ]
        if source_ip:
            lines.append(f"Source IP: {source_ip}")
        lines.append(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        return "\n".join(lines)

    # ── Channel Implementations ─────────────────────────────────────────────

    def _send_desktop(self, title: str, message: str, severity: str) -> None:
        """Send a desktop notification via plyer."""
        try:
            from plyer import notification as plyer_notification

            plyer_notification.notify(
                title=f"🛡️ NIDS: {title} ({severity})",
                message=message[:256],
                app_name="Smart Home NIDS",
                timeout=10,
            )
            logger.info("Desktop notification sent: %s", title)
        except ImportError:
            logger.debug("plyer not installed — skipping desktop notification.")
        except Exception as exc:
            logger.error("Desktop notification failed: %s", exc)

    def _send_email(self, subject: str, body: str, severity: str) -> None:
        """Send an email alert via SMTP."""
        try:
            msg = MIMEText(body, "plain")
            msg["Subject"] = f"[NIDS {severity}] {subject}"
            msg["From"] = SETTINGS.SMTP_USER
            msg["To"] = SETTINGS.ALERT_EMAIL_TO

            with smtplib.SMTP(SETTINGS.SMTP_HOST, SETTINGS.SMTP_PORT) as server:
                server.starttls()
                server.login(SETTINGS.SMTP_USER, SETTINGS.SMTP_PASS)
                server.sendmail(SETTINGS.SMTP_USER, SETTINGS.ALERT_EMAIL_TO, msg.as_string())

            logger.info("Email alert sent to %s", SETTINGS.ALERT_EMAIL_TO)
        except Exception as exc:
            logger.error("Email alert failed: %s", exc)

    def _send_telegram(self, message: str) -> None:
        """Send a Telegram message via Bot API."""
        try:
            import requests

            url = f"https://api.telegram.org/bot{SETTINGS.TELEGRAM_BOT_TOKEN}/sendMessage"
            payload = {
                "chat_id": SETTINGS.TELEGRAM_CHAT_ID,
                "text": message,
                "parse_mode": "Markdown",
            }
            resp = requests.post(url, json=payload, timeout=10)
            if resp.status_code == 200:
                logger.info("Telegram alert sent.")
            else:
                logger.warning("Telegram API returned %d: %s", resp.status_code, resp.text)
        except ImportError:
            logger.debug("requests not installed — skipping Telegram notification.")
        except Exception as exc:
            logger.error("Telegram alert failed: %s", exc)
