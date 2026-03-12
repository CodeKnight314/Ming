#!/usr/bin/env bash

set -euo pipefail

recreate_redis_container() {
  local name="$1"
  local port="$2"
  local description="$3"

  if docker ps -a --format '{{.Names}}' | grep -q "^${name}$"; then
    echo "Removing existing ${description} container (${name})..."
    docker rm -f "${name}" >/dev/null
  fi

  echo "Starting fresh ${description} container (${name}) on port ${port}..."
  docker run -d -p "${port}:6379" --name "${name}" redis:latest >/dev/null
}

recreate_redis_container "redis" "6379" "Context Redis"
recreate_redis_container "redis-queries" "6380" "Queries Redis"
recreate_redis_container "redis-kg" "6381" "KG Redis"