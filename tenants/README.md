# Multi-tenant infra on one host

Generate a per-tenant `docker-compose.<tenant>.yml` + `.env.<tenant>` from a tenant input file in this directory, so multiple tenants can run side-by-side without container/port/volume collisions. The FastAPI app still runs via `poetry run start` on the host — only the infra (Milvus, MinIO, etcd, Redis) is per-tenant.

## One-time setup

```bash
poetry install   # picks up the gen-tenant script entry
```

## Add a new tenant

```bash
cp tenants/example.env tenants/<tenant>.env
# edit PORT_OFFSET and any per-tenant secrets — the tenant ID comes from the filename, do not set TENANT_ID
poetry run gen-tenant tenants/<tenant>.env
```

This writes (at repo root):
- `docker-compose.<tenant>.yml` — infra stack with prefixed names, offset host ports, and an isolated network `<tenant>-milvus`
- `.env.<tenant>` — host-side connection strings for the FastAPI app
- `volumes/<tenant>/{etcd,minio,milvus,redis}` — bind-mount roots

All three are gitignored. `--force` overwrites existing outputs.

## Bring a tenant up

```bash
poetry run tenant:start <tenant>   # dev mode — uvicorn reload=True
poetry run tenant:serve <tenant>   # production-ish — no reload, mirrors `poetry run serve`
```

Each command brings up the tenant's infra (`docker compose ... up -d`), loads `.env.<tenant>` into the process environment, then hands off to `poetry run start` (or `poetry run serve`). pydantic-settings prefers process env over `.env`, so multiple tenants can run concurrently from different shells without contending over a shared `.env` symlink.

## Switch which tenant the app is talking to

Stop the running app (Ctrl+C), then re-run `tenant:start` (or `tenant:serve`) with the new ID:

```bash
poetry run tenant:start <other-tenant>
```

`docker compose up -d` is idempotent, so the same command handles both first-time bring-up and switching. If you only want to swap the symlink without restarting (e.g., uvicorn is managed elsewhere):

```bash
poetry run tenant:use <other-tenant>
```

## Tear down

```bash
poetry run tenant:down <tenant>
# add -v manually if you want to also delete named volumes:
#   docker compose -f docker-compose.<tenant>.yml -p <tenant> down -v
# (bind-mount data under volumes/<tenant>/ persists either way)
```

## Port allocation

Every host-side port = base + `PORT_OFFSET`. Use a unique multiple of 100 per tenant.

| Service | Container port | Base host port | Offset 0 | Offset 100 | Offset 200 |
|---|---|---|---|---|---|
| MinIO API | 9000 | 9002 | 9002 | 9102 | 9202 |
| MinIO console | 9001 | 9003 | 9003 | 9103 | 9203 |
| Milvus gRPC | 19530 | 19531 | 19531 | 19631 | 19731 |
| Milvus health | 9091 | 9092 | 9092 | 9192 | 9292 |
| Redis | 6379 | 6379 | 6379 | 6479 | 6579 |
| FastAPI app (uvicorn) | 8001 | 8001 | 8001 | 8101 | 8201 |

The app port is read by `start()`/`serve()` from `settings.app_port` ([api/__init__.py:135-141](../api/__init__.py#L135-L141)), which pydantic-settings populates from `APP_PORT` in `.env.<tenant>`. Multiple tenants' apps can therefore run simultaneously on the same host without colliding on `:8001`.

## Tenant input fields (`tenants/<tenant>.env`)

**Tenant ID:** *derived from the input filename's stem* (e.g. `tenants/iaf.env` → `iaf`). Must match `^[a-z0-9][a-z0-9_-]*$`. The slug `example` is rejected (template-only). If you set `TENANT_ID` in the file AND it differs from the filename, `gen-tenant` errors out — drop it and rename the file instead.

**Required:**
- `PORT_OFFSET` — integer, multiple of 100, range `0..9000`.

**Optional (defaults shown):**
- `MINIO_ACCESS_KEY=minioadmin`
- `MINIO_SECRET_KEY=minioadmin`
- `MINIO_BUCKET_NAME=default-bucket`
- `MILVUS_HOST=localhost` — host part of `MILVUS__HOST`. Set to a LAN IP/DNS name when Milvus runs on a different machine than the app.
- `MINIO_HOST=localhost` — host part of `MINIO__ENDPOINT`. Same idea for MinIO.
- `REDIS_HOST=localhost` — host part of `REDIS__REDIS_URL`. Same idea for Redis.

> Ports are always derived from `PORT_OFFSET`. Setting `MILVUS__PORT`, `MINIO__ENDPOINT`, or `REDIS__REDIS_URL` directly in the input has no effect — the generator overwrites them. Only the *host* portion is user-configurable, via `MILVUS_HOST` / `MINIO_HOST` / `REDIS_HOST`.

**Carried into `.env.<tenant>` if present:** `APP_ENV`, `MILVUS__CORPUS_COLLECTION_NAME`, `MILVUS__INPUT_COLLECTION_NAME`, `MILVUS__VECTOR_DIM`, `MILVUS__METRIC_TYPE`, `MINIO__SECURE`, all `LLM__*`, all `DOC_PARSER__*`, all `KEYCLOAK__*`.

## Notes

- The default single-tenant flow (`poetry run setup:milvus-minio`, plain `docker-compose.yml`, `.env.example`) is untouched — keep using it for solo dev.
- The compose `service` keys (`etcd`, `minio`, `standalone`, `redis`) are intentionally identical across tenants. They are network-scoped, and each tenant has its own bridge network, so internal hostnames don't collide. Only `container_name` and the network `name` are tenant-prefixed.
- Real tenant input files (`tenants/<tenant>.env`) are gitignored to keep secrets out of git. Only `example.env` and this README are tracked.

## TODO

- **`parse_env_file` silently overwrites duplicate keys** ([api/scripts/gen_tenant_stack.py:78](../api/scripts/gen_tenant_stack.py#L78)) — a duplicated `PORT_OFFSET=` line produces different ports than the user expects. At minimum warn; for a generator script, reject is better.
- **Add tests for pure validation/parsing functions in `gen_tenant_stack.py`** — silent bugs here cause port collisions or malformed stacks that only surface at deploy time. High-ROI cases:
  - `validate_port_offset` boundaries: 0, 9000, negative, non-multiple-of-100, non-int
  - `validate_tenant_id` regex: leading dash, uppercase, dots, empty
  - `parse_env_file`: malformed lines, duplicate keys, quoted values, line numbers in errors
