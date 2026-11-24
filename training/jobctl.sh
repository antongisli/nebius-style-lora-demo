#!/usr/bin/env bash
# Thin wrapper around `nebius ai job` for the sweep.
#   training/jobctl.sh list
#   training/jobctl.sh logs <job-id>      # last 24 h, follows
#   training/jobctl.sh get <job-id>
#   training/jobctl.sh cancel <job-id>
set -Eeuo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck disable=SC1091
source "$ROOT/deploy.env"
PROFILE="${NEBIUS_PROFILE:-default}"
cmd="${1:-list}"; shift || true
case "$cmd" in
  list) nebius --profile "$PROFILE" ai job list --parent-id "$PROJECT_ID" ;;
  logs) nebius --profile "$PROFILE" ai job logs "$1" --since 24h --follow ;;
  get) nebius --profile "$PROFILE" ai job get "$1" ;;
  cancel) nebius --profile "$PROFILE" ai job cancel "$1" ;;
  *) echo "usage: $0 list|logs|get|cancel [job-id]" >&2; exit 2 ;;
esac
