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
# option -- card plays, skirmish assignments, yes/no questions and the burden bid -- cut off one
# turn after the branch and scored by a value model, from copies of the
# game -- about 4 CPU-seconds a game with both sides searching, and a ranking that differs from the plain bot's (see
# BOT-NOTES.md, "Does the cheap bot rank decks like the strong one?"). --quick uses the plain
# v2 models instead: ~0.1 CPU-seconds a game, good for screening many decks before a search
# run on the finalists.
#
# Needs bot-models-exp/v2-real (bot-train.sh on decks/real, see BOT-NOTES.md, "Training on real
# decks") and, for search, bot-models-exp/value-real, fit on decks *other* than the ones being
# ranked; BOT_MODELS and BOT_VALUE override them. The value model was fit like this:
#   BOT_MAIN=com.gempukku.lotro.bots.simulation.BotProbe ./bot-vs-bot.sh --bot safe@bot-models-exp/v2 \
#     --suite decks/real/suite-test.txt --train-suite decks/real/suite-train.txt --games 2 \
#     --save-value bot-models-exp/value-real --value-features 2 --value-model logistic
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
    exec ./bot-vs-bot.sh --suite "rr:$format:$list" --a safe@${BOT_MODELS:-bot-models-exp/v2-real} --b safe@${BOT_MODELS:-bot-models-exp/v2-real} \
        --games "$per"
fi
BOT_MAIN=com.gempukku.lotro.bots.simulation.BotTeach exec ./bot-vs-bot.sh --play --both --clone \
    --base ${BOT_MODELS:-bot-models-exp/v2-real} --value ${BOT_VALUE:-bot-models-exp/value-real} --rollouts 2 --depth 1 --max-candidates 10 --assign --choices \
    --suite "rr:$format:$list" --games $(( pairs * per )) --out "/tmp/rank-decks-$$"
