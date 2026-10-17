#!/bin/sh
# Self-play training into a model dir of its own: ./bot-train.sh --out bot-models-x --suite starters:train
# Shares bot-vs-bot.sh's build; see BotTrain.java for options.
set -e
cd "$(dirname "$0")"
BOT_MAIN=com.gempukku.lotro.bots.simulation.BotTrain JAVA_OPTS="${JAVA_OPTS:--Xmx16g}" exec ./bot-vs-bot.sh "$@"
