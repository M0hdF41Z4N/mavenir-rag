import asyncio
import base64
import json
import logging
from typing import Any

import requests
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)
bearer_scheme = HTTPBearer(auto_error=False, scheme_name="BearerAuth")


class KeycloakSettings(BaseSettings):
    server_url: str
    realm: str
    client_id: str
    client_secret: str | None = None
    verify_ssl: bool = True
    request_timeout_seconds: float = 10.0

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="KEYCLOAK__",
        extra="ignore",
    )


class KeycloakTokenRequest(BaseModel):
    refresh_token: str = Field(..., min_length=1)
    user_id: str = Field(
        ...,
        min_length=1,
        description="Expected Keycloak user id, usually the `sub` claim.",
    )
    username: str = Field(
        ...,
        min_length=1,
        description="Expected Keycloak username, usually the `preferred_username` claim.",
    )


class KeycloakTokenResponse(BaseModel):
    access_token: str
    token_type: str
    expires_in: int
    refresh_expires_in: int | None = None
    refresh_token: str | None = None
    scope: str | None = None
    user_id: str
    username: str


class AuthenticatedKeycloakUser(BaseModel):
    user_id: str
    username: str
    email: str | None = None
    claims: dict[str, Any] = Field(default_factory=dict)


def _mask_value(value: str | None, visible_prefix: int = 6, visible_suffix: int = 4) -> str:
    if not value:
        return "<empty>"

    if len(value) <= visible_prefix + visible_suffix:
        return value

    return f"{value[:visible_prefix]}...{value[-visible_suffix:]}"


def _token_fingerprint(token: str | None) -> str:
    if not token:
        return "<missing-token>"

    parts = token.split(".")
    if len(parts) != 3:
        return _mask_value(token, visible_prefix=10, visible_suffix=6)

    return f"{parts[0][:10]}...{parts[-1][-6:]}"


def get_keycloak_settings() -> KeycloakSettings:
    try:
        settings = KeycloakSettings()
        logger.info(
            "Keycloak config loaded. server_url=%s realm=%s client_id=%s verify_ssl=%s timeout=%s",
            settings.server_url,
            settings.realm,
            settings.client_id,
            settings.verify_ssl,
            settings.request_timeout_seconds,
        )
        return settings
    except Exception as exc:
        logger.exception("Keycloak configuration is invalid.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Keycloak configuration is missing or invalid in the environment.",
        ) from exc


def get_keycloak_endpoint(settings: KeycloakSettings, endpoint: str) -> str:
    return f"{settings.server_url.rstrip('/')}/realms/{settings.realm.strip('/')}/protocol/openid-connect/{endpoint}"


def extract_username(claims: dict[str, Any]) -> str | None:
    return claims.get("preferred_username") or claims.get("username")


async def _post_request(
    url: str,
    *,
    data: dict[str, Any],
    verify_ssl: bool,
    timeout_seconds: float,
    headers: dict[str, str] | None = None,
) -> requests.Response:
    return await asyncio.to_thread(
        requests.post,
        url,
        data=data,
        headers=headers,
        verify=verify_ssl,
        timeout=timeout_seconds,
    )


async def _get_request(
    url: str,
    *,
    verify_ssl: bool,
    timeout_seconds: float,
    headers: dict[str, str] | None = None,
) -> requests.Response:
    return await asyncio.to_thread(
        requests.get,
        url,
        headers=headers,
        verify=verify_ssl,
        timeout=timeout_seconds,
    )


def _decode_jwt_payload(token: str) -> dict[str, Any]:
    try:
        parts = token.split(".")
        if len(parts) != 3:
            return {}

        payload = parts[1]
        padding = "=" * (-len(payload) % 4)
        decoded = base64.urlsafe_b64decode(payload + padding)
        return json.loads(decoded.decode("utf-8"))
    except (ValueError, json.JSONDecodeError):
        return {}


def validate_identity(claims: dict[str, Any], expected_user_id: str, expected_username: str) -> None:
    user_id_candidates = {
        value
        for value in (
            claims.get("sub"),
            claims.get("email"),
            claims.get("preferred_username"),
            claims.get("username"),
        )
        if value
    }
    username_candidates = {
        value
        for value in (
            extract_username(claims),
            claims.get("email"),
            claims.get("sub"),
        )
        if value
    }

    logger.info(
        "Validating Keycloak identity. expected_user_id=%s expected_username=%s user_id_candidates=%s username_candidates=%s",
        expected_user_id,
        expected_username,
        sorted(user_id_candidates),
        sorted(username_candidates),
    )

    if expected_user_id not in user_id_candidates:
        logger.warning(
            "Keycloak identity validation failed for user_id. expected=%s candidates=%s",
            expected_user_id,
            sorted(user_id_candidates),
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="The refresh token does not belong to the provided user_id.",
        )

    if expected_username not in username_candidates:
        logger.warning(
            "Keycloak identity validation failed for username. expected=%s candidates=%s",
            expected_username,
            sorted(username_candidates),
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="The refresh token does not belong to the provided username.",
        )

    logger.info(
        "Keycloak identity validation succeeded for user_id=%s username=%s",
        expected_user_id,
        expected_username,
    )


async def exchange_refresh_token(settings: KeycloakSettings, refresh_token: str) -> dict[str, Any]:
    request_body = {
        "grant_type": "refresh_token",
        "client_id": settings.client_id,
        "refresh_token": refresh_token,
    }

    if settings.client_secret:
        request_body["client_secret"] = settings.client_secret

    logger.info(
        "Keycloak step 1/3: exchanging refresh token. endpoint=%s token=%s",
        get_keycloak_endpoint(settings, "token"),
        _token_fingerprint(refresh_token),
    )

    try:
        response = await _post_request(
            get_keycloak_endpoint(settings, "token"),
            data=request_body,
            verify_ssl=settings.verify_ssl,
            timeout_seconds=settings.request_timeout_seconds,
        )
    except requests.RequestException as exc:
        logger.exception("Failed to reach Keycloak token endpoint.")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Unable to contact Keycloak token endpoint: {exc}",
        ) from exc

    logger.info(
        "Keycloak token endpoint responded. status_code=%s",
        response.status_code,
    )

    if not response.ok:
        try:
            error_payload = response.json()
        except ValueError:
            error_payload = {}

        error_description = error_payload.get("error_description") or error_payload.get("error") or "Keycloak rejected the refresh token request."
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=error_description,
        )

    payload = response.json()
    logger.info(
        "Keycloak step 1/3 complete. access_token_present=%s refresh_token_present=%s expires_in=%s",
        bool(payload.get("access_token")),
        bool(payload.get("refresh_token")),
        payload.get("expires_in"),
    )
    return payload


async def fetch_userinfo(settings: KeycloakSettings, access_token: str) -> dict[str, Any]:
    try:
        response = await _get_request(
            get_keycloak_endpoint(settings, "userinfo"),
            headers={"Authorization": f"Bearer {access_token}"},
            verify_ssl=settings.verify_ssl,
            timeout_seconds=settings.request_timeout_seconds,
        )
    except requests.RequestException as exc:
        logger.exception("Failed to reach Keycloak userinfo endpoint.")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Unable to validate the access token with Keycloak: {exc}",
        ) from exc

    if not response.ok:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="The supplied access token is invalid or expired.",
        )

    try:
        return response.json()
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Keycloak returned an invalid userinfo response.",
        ) from exc


async def introspect_token(
    settings: KeycloakSettings,
    token: str,
    token_type_hint: str = "access_token",  # noqa: S107
) -> dict[str, Any]:
    request_body = {
        "token": token,
        "token_type_hint": token_type_hint,
        "client_id": settings.client_id,
    }

    if settings.client_secret:
        request_body["client_secret"] = settings.client_secret

    logger.info(
        "Keycloak step 2/3: introspecting token. endpoint=%s token_type_hint=%s token=%s",
        get_keycloak_endpoint(settings, "token/introspect"),
        token_type_hint,
        _token_fingerprint(token),
    )

    try:
        response = await _post_request(
            get_keycloak_endpoint(settings, "token/introspect"),
            data=request_body,
            verify_ssl=settings.verify_ssl,
            timeout_seconds=settings.request_timeout_seconds,
        )
    except requests.ConnectTimeout as exc:
        logger.exception("Failed to reach Keycloak introspection endpoint.")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=("Unable to connect to Keycloak introspection endpoint before timeout. Check VPN, firewall, proxy, or increase KEYCLOAK__REQUEST_TIMEOUT_SECONDS."),
        ) from exc
    except requests.RequestException as exc:
        logger.exception("Failed to reach Keycloak introspection endpoint.")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Unable to validate the token with Keycloak: {exc}",
        ) from exc

    logger.info(
        "Keycloak introspection endpoint responded. status_code=%s",
        response.status_code,
    )

    if not response.ok:
        try:
            error_payload = response.json()
        except ValueError:
            error_payload = {}

        error_description = error_payload.get("error_description") or error_payload.get("error") or "Keycloak rejected the token introspection request."
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=error_description,
        )

    try:
        payload = response.json()
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Keycloak returned an invalid introspection response.",
        ) from exc

    if not payload.get("active"):
        logger.warning(
            "Keycloak introspection returned inactive token. token=%s",
            _token_fingerprint(token),
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="The supplied access token is invalid or expired.",
        )

    logger.info(
        "Keycloak step 2/3 complete. token active for sub=%s username=%s scope=%s",
        payload.get("sub"),
        payload.get("username") or payload.get("preferred_username"),
        payload.get("scope"),
    )
    return payload


def merge_claims(introspection_claims: dict[str, Any], token: str) -> dict[str, Any]:
    token_claims = _decode_jwt_payload(token)
    merged_claims = {**token_claims, **introspection_claims}

    username = extract_username(merged_claims)
    if username:
        merged_claims["preferred_username"] = username

    logger.info(
        "Keycloak claims merged. sub=%s username=%s email=%s",
        merged_claims.get("sub"),
        merged_claims.get("preferred_username"),
        merged_claims.get("email"),
    )

    return merged_claims


async def create_access_token(payload: KeycloakTokenRequest) -> KeycloakTokenResponse:
    logger.info(
        "Keycloak token API called. requested_user_id=%s requested_username=%s refresh_token=%s",
        payload.user_id,
        payload.username,
        _token_fingerprint(payload.refresh_token),
    )
    settings = get_keycloak_settings()
    token_payload = await exchange_refresh_token(settings, payload.refresh_token)
    access_token = token_payload.get("access_token")

    if not access_token:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Keycloak did not return an access token.",
        )

    introspection_claims = await introspect_token(settings, access_token)
    claims = merge_claims(introspection_claims, access_token)
    logger.info("Keycloak step 3/3: validating access token claims against request payload.")
    validate_identity(claims, payload.user_id, payload.username)

    logger.info(
        "Keycloak token API succeeded. user_id=%s username=%s access_token=%s",
        payload.user_id,
        payload.username,
        _token_fingerprint(access_token),
    )

    return KeycloakTokenResponse(
        access_token=access_token,
        token_type=token_payload.get("token_type", "Bearer"),
        expires_in=int(token_payload.get("expires_in", 0)),
        refresh_expires_in=token_payload.get("refresh_expires_in"),
        refresh_token=token_payload.get("refresh_token"),
        scope=token_payload.get("scope"),
        user_id=payload.user_id,
        username=payload.username,
    )


_DEV_USER = AuthenticatedKeycloakUser(
    user_id="test-user-id-00000000",
    username="test_user",
    email="test@example.com",
    claims={
        "sub": "test-user-id-00000000",
        "preferred_username": "test_user",
        "email": "test@example.com",
        "email_verified": True,
        "active": True,
    },
)


async def require_keycloak_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),  # noqa: B008
) -> AuthenticatedKeycloakUser:
    from api.config import settings as app_settings  # local import to avoid circular dependency at module load

    if app_settings.app_env == "dev":
        logger.info("DEV mode: skipping Keycloak auth, returning test user.")
        return _DEV_USER

    if credentials is None:
        logger.warning("Keycloak bearer validation failed: missing Authorization header.")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authorization bearer token is required.",
        )

    logger.info(
        "Keycloak bearer validation started. token=%s",
        _token_fingerprint(credentials.credentials),
    )
    settings = get_keycloak_settings()
    introspection_claims = await introspect_token(settings, credentials.credentials)
    claims = merge_claims(introspection_claims, credentials.credentials)
    user_id = claims.get("sub")
    username = extract_username(claims)

    if not user_id or not username:
        logger.warning(
            "Keycloak bearer validation failed: required claims missing. sub=%s username=%s",
            user_id,
            username,
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="The access token is missing required Keycloak claims.",
        )

    logger.info(
        "Keycloak bearer validation succeeded. user_id=%s username=%s email=%s",
        user_id,
        username,
        claims.get("email"),
    )

    return AuthenticatedKeycloakUser(
        user_id=user_id,
        username=username,
        email=claims.get("email"),
        claims=claims,
    )
