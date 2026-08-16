# Developer setup helpers for starting/stopping local infrastructure.
# Milvus and MinIO are managed via docker-compose; Ollama is run as a subprocess.
# Run `python -m api.scripts.setup` or call the functions directly from a task runner.
import logging
import os
import shutil
import subprocess
import time

import requests

from api.config import settings

OLLAMA_CHAT_MODEL = settings.llm.ollama_chat_model
OLLAMA_EMBED_MODEL = settings.llm.ollama_embed_model

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# -----------------------------
# Subprocess Command Utility runner
# -----------------------------
def run(cmd: list[str]) -> None:
    logger.info(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)  # noqa: S603, S607


# -----------------------------
# Ollama
# -----------------------------
def start_ollama() -> None:
    logger.info("Starting Ollama server...")

    subprocess.Popen(
        ["ollama", "serve"],  # noqa: S603, S607
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    # Wait for Ollama Service to Start before pulling the models
    while True:
        try:
            # Try reaching Ollama Server
            # Ref: https://docs.ollama.com/api/introduction#base-url
            requests.get("http://localhost:11434", timeout=50)
            break
        except Exception:
            time.sleep(0.3)

    logger.info("Ollama started")

    logger.info("Pulling embedding model...")
    run(["ollama", "pull", OLLAMA_EMBED_MODEL])

    logger.info("Pulling LLM...")
    run(["ollama", "pull", OLLAMA_CHAT_MODEL])


# -----------------------------
# (Milvus + MinIO) Docker services
# -----------------------------
def start_milvus_minio() -> None:
    logger.info("Starting Milvus + MinIO...")

    run(
        [
            "docker",
            "compose",
            "up",
            "-d",
        ]
    )


# -----------------------------
# Full setup
# -----------------------------
def start_all() -> None:
    start_milvus_minio()
    start_ollama()

    logger.info("Started all the services")


# -----------------------------
# Stop Milvus-MinIO Docker container
# -----------------------------
def stop_milvus_minio() -> None:
    logger.info("Stopping Milvus + MinIO...")

    run(
        [
            "docker",
            "compose",
            "down",
        ]
    )

    logger.info("Milvus + MinIO stopped")


# -----------------------------
# Stop Ollama Server
# -----------------------------
def stop_ollama() -> None:
    logger.info("Stopping Ollama server...")

    subprocess.run(
        ["pkill", "-f", "ollama serve"],  # noqa: S603, S607
        check=False,
    )

    logger.info("Ollama stopped")


# -----------------------------
# Stop all the services
# -----------------------------
def stop_all() -> None:
    stop_ollama()
    stop_milvus_minio()

    logger.info("All services stopped")


# -----------------------------
# Clear all persisted data (volumes)
# -----------------------------
def clear_data() -> None:
    logger.info("Stopping services before clearing data...")
    stop_all()

    # volume_base = os.environ.get("DOCKER_VOLUME_DIRECTORY", os.getcwd())
    # volumes_dir = os.path.join(volume_base, "volumes")
    volumes_dir = os.path.join(os.getcwd(), "volumes")

    if os.path.exists(volumes_dir):
        logger.info(f"Removing volumes directory: {volumes_dir}")
        shutil.rmtree(volumes_dir)
        logger.info("All data cleared")
    else:
        logger.info(f"No volumes directory found at '{volumes_dir}', nothing to clear")
