"""Tenant lifecycle commands wrapping docker-compose for the FastAPI app.

Each command takes a tenant ID as a positional CLI argument:

    poetry run tenant:start <tenant>   # compose up -d, inject .env.<tenant> into process env, hand off to `poetry run start` (reload)
    poetry run tenant:serve <tenant>   # compose up -d, inject .env.<tenant> into process env, hand off to `poetry run serve` (no reload)
    poetry run tenant:down <tenant>    # compose down for the tenant
    poetry run tenant:use <tenant>     # symlink .env -> .env.<tenant> (for plain `poetry run start` outside this wrapper)

`tenant:start`/`tenant:serve` inject the tenant's vars directly into the
process environment before exec, so multiple tenants can run concurrently in
separate shells without contending over a shared `.env` symlink. Process env
takes precedence over `.env` in pydantic-settings, so the active symlink
(if any) does not affect the launched app.

Compose file and per-tenant .env must already exist:

    poetry run gen-tenant tenants/<tenant>.env

Compose `up -d` is idempotent, so `tenant:start`/`tenant:serve` double as the "switch tenant" flow.
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path

from api.scripts.gen_tenant_stack import parse_env_file

logger = logging.getLogger(__name__)


def _parse_tenant_arg(action: str) -> str:
    parser = argparse.ArgumentParser(
        prog=f"tenant:{action}",
        description=f"Tenant '{action}' command — see api/scripts/tenant_ops.py.",
    )
    parser.add_argument(
        "tenant",
        help="Tenant ID — slug used in docker-compose.<tenant>.yml / .env.<tenant>",
    )
    return parser.parse_args().tenant


def _resolve_artifacts(tenant: str) -> tuple[Path, Path]:
    repo_root = Path.cwd()
    compose_file = repo_root / f"docker-compose.{tenant}.yml"
    env_file = repo_root / f".env.{tenant}"
    missing = [p for p in (compose_file, env_file) if not p.exists()]
    if missing:
        joined = ", ".join(str(p.relative_to(repo_root)) for p in missing)
        sys.exit(f"Missing artifacts for tenant '{tenant}': {joined}\nRun: poetry run gen-tenant tenants/{tenant}.env")
    return compose_file, env_file


def _symlink_env(env_file: Path) -> None:
    target = Path.cwd() / ".env"
    if target.is_symlink() or target.exists():
        target.unlink()
    os.symlink(env_file.name, target)
    logger.info(f"Symlinked .env -> {env_file.name}")


def _inject_env(env_file: Path) -> None:
    """Load .env.<tenant> KEY=VALUE pairs into os.environ for the child process.

    pydantic-settings prefers process env over the `.env` file, so the child
    FastAPI process picks up these values regardless of which tenant the
    `.env` symlink currently points at. This is what lets multiple tenants
    run concurrently from different shells.
    """
    parsed = parse_env_file(env_file)
    for key, value in parsed.items():
        os.environ[key] = value
    logger.info(f"Loaded {len(parsed)} vars from {env_file.name} into process env")


def _compose(compose_file: Path, tenant: str, *args: str) -> None:
    docker = shutil.which("docker")
    if docker is None:
        sys.exit("`docker` not found on PATH.")
    try:
        subprocess.run(  # noqa: S603
            [docker, "compose", "-f", compose_file.name, "-p", tenant, *args],
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        sys.exit(f"docker compose {' '.join(args)} failed for tenant '{tenant}' (exit {exc.returncode}). See docker output above.")


def _bring_up_and_handoff(action: str, runner: str) -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    tenant = _parse_tenant_arg(action)
    compose_file, env_file = _resolve_artifacts(tenant)
    logger.info(f"Bringing up tenant '{tenant}' infrastructure...")
    _compose(compose_file, tenant, "up", "-d")
    _inject_env(env_file)
    logger.info(f"Handing off to `poetry run {runner}` for tenant '{tenant}'...")
    poetry = shutil.which("poetry")
    if poetry is None:
        sys.exit(f"`poetry` not found on PATH; cannot hand off to `poetry run {runner}`.")
    try:
        os.execvp(poetry, [poetry, "run", runner])  # noqa: S606 — replaces this process
    except OSError as exc:
        sys.exit(f"Failed to exec `poetry run {runner}`: {exc}. Infrastructure is still running. Tear down with: poetry run tenant:down {tenant}")


def start() -> None:
    """tenant:start <id> — compose up -d, inject .env.<tenant> into process env, hand off to `poetry run start` (reload)."""
    _bring_up_and_handoff("start", "start")


def serve() -> None:
    """tenant:serve <id> — compose up -d, inject .env.<tenant> into process env, hand off to `poetry run serve` (no reload)."""
    _bring_up_and_handoff("serve", "serve")


def down() -> None:
    """tenant:down <id> — docker compose down for the tenant."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    tenant = _parse_tenant_arg("down")
    compose_file, _ = _resolve_artifacts(tenant)
    logger.info(f"Tearing down tenant '{tenant}'...")
    _compose(compose_file, tenant, "down")


def use() -> None:
    """tenant:use <id> — symlink .env -> .env.<tenant>, no compose action, no app start."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    tenant = _parse_tenant_arg("use")
    _, env_file = _resolve_artifacts(tenant)
    _symlink_env(env_file)
