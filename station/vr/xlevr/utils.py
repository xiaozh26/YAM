"""Utility functions for XLeVR."""

import os
import subprocess
import logging
from pathlib import Path
from typing import Tuple

logger = logging.getLogger(__name__)


def generate_ssl_certificates(cert_path: str = "cert.pem", key_path: str = "key.pem") -> bool:
    """Generate self-signed SSL certs if they don't already exist."""
    if os.path.exists(cert_path) and os.path.exists(key_path):
        logger.info(f"SSL certificates already exist: {cert_path}, {key_path}")
        return True

    logger.info("Generating self-signed SSL certificates...")

    try:
        cmd = [
            "openssl", "req", "-x509", "-newkey", "rsa:2048",
            "-keyout", key_path,
            "-out", cert_path,
            "-sha256", "-days", "365", "-nodes",
            "-subj", "/C=US/ST=Test/L=Test/O=Test/OU=Test/CN=localhost"
        ]
        subprocess.run(cmd, capture_output=True, text=True, check=True)
        os.chmod(key_path, 0o600)
        os.chmod(cert_path, 0o644)
        logger.info(f"SSL certificates generated: {cert_path}, {key_path}")
        return True

    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to generate SSL certificates: {e}\n{e.stderr}")
        return False
    except FileNotFoundError:
        logger.error(
            "OpenSSL not found. Install it first:\n"
            "  Ubuntu/Debian: sudo apt-get install openssl\n"
            "  macOS: brew install openssl"
        )
        return False
    except Exception as e:
        logger.error(f"Unexpected error generating SSL certificates: {e}")
        return False


def ensure_ssl_certificates(cert_path: str = "cert.pem", key_path: str = "key.pem") -> bool:
    """Call generate_ssl_certificates and log the manual fallback command if it fails."""
    if not generate_ssl_certificates(cert_path, key_path):
        logger.error(
            "Could not generate SSL certificates automatically. Run manually:\n"
            'openssl req -x509 -newkey rsa:2048 -keyout key.pem -out cert.pem '
            '-sha256 -days 365 -nodes -subj "/C=US/ST=Test/L=Test/O=Test/OU=Test/CN=localhost"'
        )
        return False
    return True
