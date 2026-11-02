#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."
export DEPLOY_SHA=${DEPLOY_SHA:-local}
: "${DISCORD_TOKEN:?Set GitHub secret DISCORD_TOKEN}"
: "${OPENAI_API_KEY:?Set GitHub secret OPENAI_API_KEY}"
: "${DSBOT_DATA_DIR:?Set GitHub variable DSBOT_DATA_DIR to the data directory on mercury}"
[[ "$DSBOT_DATA_DIR" = /* && "$DSBOT_DATA_DIR" != / ]]
compose=(docker compose --env-file /dev/null -f compose.yaml)

if [[ $(docker info --format '{{.Name}}') != mercury ]]; then
    echo '::error::Expected the Docker daemon on mercury.' >&2
    exit 1
fi
docker compose version
"${compose[@]}" config --quiet
"${compose[@]}" build bot

"${compose[@]}" up -d --no-build --force-recreate bot
printf 'Compose started `%s`.\n' "$DEPLOY_SHA" >> "$GITHUB_STEP_SUMMARY"
