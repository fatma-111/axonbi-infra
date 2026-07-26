#!/usr/bin/env bash
set -euo pipefail
cd /root/axonbi-infra
git fetch origin
git reset --hard origin/main
cd langgraph
docker compose up -d --build
docker image prune -f
