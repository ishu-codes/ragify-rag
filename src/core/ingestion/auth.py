from os import getenv
from typing import Any

import requests
from grobid_client.grobid_client import GrobidClient, ServerUnavailableException

from src.core.utils.logger import get_logger

logger = get_logger("ragify.grobid")


def _authorized_session(audience: str):
    """requests session that signs every call with a Cloud Run ID token."""
    from google.auth.transport.requests import AuthorizedSession, Request
    from google.oauth2 import id_token

    credentials = id_token.fetch_id_token_credentials(
        audience.rstrip("/"), request=Request()
    )
    return AuthorizedSession(credentials)


class AuthorizedGrobidClient(GrobidClient):
    """GrobidClient that authenticates with a Cloud Run ID token.

    The deployed Grobid runs with ``--no-allow-unauthenticated``, so requests
    have to carry a Google-signed ID token whose audience is the service URL.
    ``AuthorizedSession`` attaches the token and refreshes it on expiry.
    """

    def __init__(self, *args: Any, audience: str, **kwargs: Any) -> None:
        self._session = _authorized_session(audience)
        super().__init__(*args, **kwargs)

    def call_api(
        self,
        method: str,
        url: str,
        headers: dict[str, str] | None = None,
        params: dict[str, Any] | None = None,
        data: dict[str, Any] | None = None,
        files: dict[str, Any] | None = None,
        timeout: int | None = None,
    ):
        # Mirrors grobid_client.ApiClient.call_api, but through the authorized
        # session so the ID token rides along on each request.
        request_headers = dict(headers or {})
        request_headers["Accept"] = self.accept_type
        response = self._session.request(
            method,
            url,
            headers=request_headers,
            params=params or {},
            data=data or {},
            files=files or {},
            timeout=timeout,
        )
        return response, response.status_code

    def _test_server_connection(self) -> tuple[bool, int]:
        """Authenticated twin of GrobidClient._test_server_connection."""
        the_url = self.get_server_url("isalive")
        try:
            response = self._session.get(the_url, timeout=10)
        except requests.exceptions.RequestException as exc:
            error_msg = (
                f"GROBID server {self.config['grobid_server']} does not appear up"
                f" and running, connection failed: {exc}"
            )
            logger.error("grobid_unavailable", extra={"error": error_msg})
            raise ServerUnavailableException(error_msg) from exc

        if response.status_code != 200:
            error_msg = (
                f"GROBID server {self.config['grobid_server']} does not appear up"
                f" and running (status: {response.status_code})"
            )
            logger.error("grobid_unhealthy", extra={"error": error_msg})
            return False, response.status_code

        return True, response.status_code


def build_grobid_client(grobid_url: str) -> GrobidClient:
    """Grobid client for ``grobid_url``, authenticated when that requires it.

    Authentication is driven by ``GROBID_AUDIENCE`` (defaulting to the HTTPS
    Grobid URL, which is what Cloud Run validates against). Plain ``http://``
    URLs — local development against a local Grobid — stay unauthenticated.
    """
    audience = getenv("GROBID_AUDIENCE") or (
        grobid_url if grobid_url.startswith("https://") else ""
    )
    if not audience:
        return GrobidClient(grobid_server=grobid_url, check_server=False)

    try:
        return AuthorizedGrobidClient(
            grobid_server=grobid_url,
            # Don't ping on init: on Cloud Run the Grobid instance may still be
            # cold-starting when this client is constructed.
            check_server=False,
            audience=audience,
        )
    except Exception as exc:  # no metadata server (local runs) or missing dep
        logger.warning(
            "grobid_id_token_unavailable",
            extra={"audience": audience, "error": str(exc)},
        )
        return GrobidClient(grobid_server=grobid_url, check_server=False)
