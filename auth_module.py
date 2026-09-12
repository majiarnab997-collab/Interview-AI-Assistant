"""
Interview AI Assistant — Authentication Module (Reference Implementation)
====================================================================
This module demonstrates a complete authentication system using PostgreSQL
(NeonDB), bcrypt password hashing, and email-based OTP verification.

⚠️ Note: This module is intentionally NOT wired into the live public demo
(app.py), so visitors can test the app with zero friction. It is kept here
to demonstrate production-grade authentication capability.

To use it: import this module in app.py, add Login/Register UI components
to the Gradio Blocks layout, and set the required secrets (DATABASE_URL,
SENDER_EMAIL_ID, APP_PASSWORD_ID) in your environment.
"""

import os
import random
import smtplib
import string
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

import bcrypt
import psycopg2
import gradio as gr

DATABASE_URL = os.getenv("DATABASE_URL")
SENDER_EMAIL = os.getenv("SENDER_EMAIL_ID")
APP_PASSWORD = os.getenv("APP_PASSWORD_ID")

OTP_VALIDITY_SECONDS = 300  # 5 minutes


def get_db_connection():
    """Creates a new PostgreSQL connection."""
    return psycopg2.connect(DATABASE_URL)


def is_strong_password(password: str) -> bool:
    """Checks that the password is at least 8 characters and contains a digit and a special character."""
    has_digit = any(ch.isdigit() for ch in password)
    has_special = any(ch in string.punctuation for ch in password)
    has_length = len(password) >= 8
    return has_digit and has_special and has_length


def otp_generate(new_email: str, new_username: str):
    """Generates a 6-digit OTP and emails it to the user. Leak-free via try/finally connection handling."""
    OTP_GENERATED = random.randint(100000, 999999)

    if not new_email or "@" not in new_email or "." not in new_email:
        return None, None, gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), \
               gr.update(value="❌ Please provide a valid email id.", visible=True)

    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT username, email FROM users WHERE username = %s OR email = %s;",
            (new_username, new_email)
        )
        if cursor.fetchone():
            return None, None, gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), \
                   gr.update(value="⚠️ You are already registered.", visible=True)

        try:
            msg = MIMEMultipart()
            msg["From"] = SENDER_EMAIL
            msg["To"] = new_email
            msg["Subject"] = "CareerForge AI - Your Verification OTP"
            msg.attach(MIMEText(
                f"Your verification code is: {OTP_GENERATED}\nThis code is valid for 5 minutes.",
                "plain"
            ))
            with smtplib.SMTP("smtp.gmail.com", 587) as server:
                server.starttls()
                server.login(SENDER_EMAIL, APP_PASSWORD)
                server.send_message(msg)

            return OTP_GENERATED, datetime.now(), gr.update(visible=True), gr.update(visible=True), \
                   gr.update(visible=True), gr.update(value="✅ An OTP has been sent to your email!", visible=True)
        except Exception as e:
            return None, None, gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), \
                   gr.update(value=f"❌ Error sending OTP: {str(e)}", visible=True)
    finally:
        cursor.close()
        conn.close()


def otp_verification(otp_box, saved_otp_state, otp_timestamp_state, new_username, new_password, new_email):
    """Verifies the OTP (with expiry check) and, if valid, registers the user with a bcrypt-hashed password."""
    if not otp_box or not otp_box.isdigit():
        return "No", "❌ Please enter a valid numeric OTP.", gr.update(visible=False), gr.update(visible=True)

    if otp_timestamp_state is None or (datetime.now() - otp_timestamp_state).total_seconds() > OTP_VALIDITY_SECONDS:
        return "No", "❌ OTP has expired. Please request a new one.", gr.update(visible=False), gr.update(visible=True)

    if int(otp_box) != saved_otp_state:
        return "No", "❌ Your OTP did not match.", gr.update(visible=False), gr.update(visible=True)

    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        password_hash = bcrypt.hashpw(new_password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")
        cursor.execute(
            "INSERT INTO users (username, email, password_hash) VALUES (%s, %s, %s);",
            (new_username, new_email, password_hash)
        )
        conn.commit()
        return "Yes", "✅ Your account has been created successfully!", gr.update(visible=True), gr.update(visible=False)
    finally:
        cursor.close()
        conn.close()


def checking_register_user(username: str, password: str):
    """Securely verifies login credentials using bcrypt. Leak-free connection handling."""
    if not password:
        return "❌ Please enter password", gr.update(visible=True), gr.update(visible=False)

    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT password_hash FROM users WHERE username = %s;", (username,))
        result = cursor.fetchone()

        if result and bcrypt.checkpw(password.encode("utf-8"), result[0].encode("utf-8")):
            return "✅ Login Successful!", gr.update(visible=False), gr.update(visible=True)
        return "❌ Invalid username or password", gr.update(visible=True), gr.update(visible=False)
    finally:
        cursor.close()
        conn.close()
