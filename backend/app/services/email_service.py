"""
Email service for verification and notifications.

Uses SMTP when configured, falls back to logging tokens in dev mode.
"""

from __future__ import annotations

import logging
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

from app.config import SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASSWORD, SMTP_FROM, SMTP_ENABLED, FRONTEND_URL

logger = logging.getLogger("meowllm.services.email")


def send_email(to: str, subject: str, html_body: str) -> bool:
    """Send an email. Returns True on success, False if SMTP not configured or failed."""
    if not SMTP_ENABLED:
        logger.info(
            "[DEV] Email not sent (SMTP not configured). To: %s, Subject: %s",
            to, subject,
        )
        return False

    try:
        msg = MIMEMultipart("alternative")
        msg["From"] = SMTP_FROM
        msg["To"] = to
        msg["Subject"] = subject
        msg.attach(MIMEText(html_body, "html"))

        with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=10) as server:
            server.starttls()
            server.login(SMTP_USER, SMTP_PASSWORD)
            server.send_message(msg)

        logger.info("Email sent to %s: %s", to, subject)
        return True
    except Exception as e:
        logger.error("Failed to send email to %s: %s", to, e)
        return False


def send_verification_email(to: str, token: str) -> bool:
    """Send email verification link."""
    verify_url = f"{FRONTEND_URL}/verify-email?token={token}"
    html = f"""
    <h2>Welcome to MeowTrain! 🐱</h2>
    <p>Please verify your email address by clicking the link below:</p>
    <p><a href="{verify_url}" style="
        display: inline-block;
        padding: 12px 24px;
        background-color: #6366f1;
        color: white;
        text-decoration: none;
        border-radius: 8px;
        font-weight: bold;
    ">Verify Email</a></p>
    <p>Or copy this link: {verify_url}</p>
    <p>This link expires in 24 hours.</p>
    <p style="color: #666; font-size: 12px;">
        If you did not create this account, you can safely ignore this email.
    </p>
    """
    return send_email(to, "Verify your MeowTrain email", html)


def send_password_reset_email(to: str, token: str) -> bool:
    """Send password reset link."""
    reset_url = f"{FRONTEND_URL}/reset-password?token={token}"
    html = f"""
    <h2>Password Reset 🔒</h2>
    <p>A password reset was requested for your MeowTrain account.</p>
    <p><a href="{reset_url}" style="
        display: inline-block;
        padding: 12px 24px;
        background-color: #6366f1;
        color: white;
        text-decoration: none;
        border-radius: 8px;
        font-weight: bold;
    ">Reset Password</a></p>
    <p>Or copy this link: {reset_url}</p>
    <p>This link expires in 1 hour.</p>
    <p style="color: #666; font-size: 12px;">
        If you did not request this reset, you can safely ignore this email.
    </p>
    """
    return send_email(to, "MeowTrain Password Reset", html)
