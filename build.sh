#!/usr/bin/env bash
# Exit immediately if a command exits with a non-zero status
set -e

echo "=== [1/4] Upgrading pip ==="
python -m pip install --upgrade pip

echo "=== [2/4] Installing lightweight CPU-only PyTorch for cloud containers ==="
pip install torch --extra-index-url https://download.pytorch.org/whl/cpu

echo "=== [3/4] Installing application requirements ==="
pip install -r requirements.txt

echo "=== [4/4] Executing database migrations and seeding initial dataset ==="
alembic upgrade head
python -m quantpulse.db.seed

echo "=== Build completed successfully! ==="
