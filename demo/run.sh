#!/usr/bin/env bash
set -euo pipefail

demo_directory=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repository=$(cd -- "$demo_directory/.." && pwd)

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required: https://docs.astral.sh/uv/getting-started/installation/" >&2
  exit 1
fi

cd -- "$repository"
exec uv run --locked --no-dev -m demo.local \
  --config "$demo_directory/config.yaml" "$@"
