#!/bin/bash
# Rank decks by playing every pair of them against each other.
#
#   ./rank-decks.sh <format> <deck list> [games per pair] [--quick]
#   ./rank-decks.sh king_block decks/king8.txt 40
#
# The deck list has one deck spec per line (starter:<name>, or a file holding
# ringbearer|ring|sites|cards); '#' lines are comments. Prints each deck's wins over all
# its games, with a 95% interval.
#
# By default both sides play with the search bot (BotTeach --play --both): 2 rollouts per
# option, cut off one turn after the branch and scored by a value model, from copies of the
# game -- about 2 CPU-seconds a game, and a ranking that differs from the plain bot's (see
# BOT-NOTES.md, "Does the cheap bot rank decks like the strong one?"). --quick uses the plain
# v2 models instead: ~0.1 CPU-seconds a game, good for screening many decks before a search
# run on the finalists.
#
# Needs bot-models-exp/v2 (bot-train.sh, see BOT-NOTES.md) and, for search,
# bot-models-exp/value-s2-logistic, fit on decks *other* than the ones being ranked:
#   BOT_MAIN=com.gempukku.lotro.bots.simulation.BotProbe ./bot-vs-bot.sh --bot general@bot-models-exp/v2 \
#     --suite starters:test --train-suite starters:train --games 20 \
#     --save-value bot-models-exp/value-s2-logistic --value-features 2 --value-model logistic
set -e
cd "$(dirname "$0")"
[ $# -ge 2 ] || { sed -n '2,10p' "$0"; exit 1; }
format=$1; list=$2; per=${3:-20}
quick=; for a in "$@"; do [ "$a" = --quick ] && quick=1; done
[ "$per" = --quick ] && per=20

decks=$(grep -cv '^\s*\(#\|$\)' "$list")
pairs=$(( decks * (decks - 1) / 2 ))
echo "$decks decks, $pairs pairs, $per games a pair ($(( pairs * per )) games)" >&2

if [ -n "$quick" ]; then
    exec ./bot-vs-bot.sh --suite "rr:$format:$list" --a general@bot-models-exp/v2 --b general@bot-models-exp/v2 \
        --games "$per"
fi
BOT_MAIN=com.gempukku.lotro.bots.simulation.BotTeach exec ./bot-vs-bot.sh --play --both --clone \
    --value bot-models-exp/value-s2-logistic --rollouts 2 --depth 1 --max-candidates 10 \
    --suite "rr:$format:$list" --games $(( pairs * per )) --out "/tmp/rank-decks-$$"
