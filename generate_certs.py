"""
Generate a self-signed TLS certificate and private key for local testing.

Creates:
    certs/server.crt  — public certificate (safe to share / commit)
    certs/server.key  — private key (NEVER commit this)

The certificate is valid for 'localhost' and '127.0.0.1' for 365 days.
"""

import datetime
import ipaddress
import pathlib

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID


def generate(output_dir: str = "certs") -> tuple[pathlib.Path, pathlib.Path]:
    """Generate a self-signed RSA-2048 certificate for localhost."""
    out = pathlib.Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    cert_path = out / "server.crt"
    key_path = out / "server.key"

    # Generate RSA-2048 private key
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)

    # Build X.509 certificate
    subject = issuer = x509.Name([
        x509.NameAttribute(NameOID.COMMON_NAME, "localhost"),
        x509.NameAttribute(NameOID.ORGANIZATION_NAME, "MCP Traffic Lab"),
    ])

    now = datetime.datetime.now(datetime.timezone.utc)
    cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(issuer)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now)
        .not_valid_after(now + datetime.timedelta(days=365))
        .add_extension(
            x509.SubjectAlternativeName([
                x509.DNSName("localhost"),
                x509.IPAddress(ipaddress.IPv4Address("127.0.0.1")),
            ]),
            critical=False,
        )
        .sign(key, hashes.SHA256())
    )

    # Write private key (PEM, no password)
    key_path.write_bytes(
        key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.TraditionalOpenSSL,
            encryption_algorithm=serialization.NoEncryption(),
        )
    )

    # Write certificate (PEM)
    cert_path.write_bytes(cert.public_bytes(serialization.Encoding.PEM))

    print(f"[+] Certificate : {cert_path.resolve()}")
    print(f"[+] Private key : {key_path.resolve()}")
    print(f"[+] Valid for   : localhost, 127.0.0.1  (365 days)")

    return cert_path, key_path


if __name__ == "__main__":
    generate()
