#!/bin/bash
# Self-play rounds: expert iteration anchored to people. Each generation:
#   1. play  -- the current bot (seat A, searched) against the league; at a share of card-action decisions every
#               option is tried with short play-outs scored by the win prediction (BotTeach -Dsearch.judge), and the
#               board with every option's value is kept (soft targets);
#   2. learn -- the network fine-tuned from its last checkpoint on people's decisions (all ten months) plus those
#               boards: target = its own prior x exp(value / lambda) over the searched options (board_net2.py --teacher);
#   3. check -- exported next to the trees as bot-models-exp/sp-gNN; July agreement, head-to-head against its parent
#               and sel10-net6 (games logged), the pool; adopted if no weaker than its parent and no less human-like;
#   4. report -- labels/nn/rounds/gen-NN.json for the dashboard.
#
# usage: scripts/selfplay.sh <first gen> <last gen>
#   env: START (the bot generation 0 is, default sel10-n2p), START_PT (its checkpoint), JUDGE_PT (the judge),
#        GAMES (teacher games a round, 600), ROLLOUTS (8), TEACH_RATE (0.3), LAM (0.1), MIX ("default 0": pure network),
#        NET6_EVERY (1: the match against sel10-net6 every generation), PHASE_CUT (true: play-outs end with the phase they branched in -- 3.5x the searched decisions an hour of
#        ending them with the turn, value gaps a third smaller), PEOPLE_FRAC (0.25 of people's decisions a round)
set -u
cd ~/dev/gemp-lotr/gemp-lotr
L=~/dev/gemp-replays/labels
N=$L/nn
LOG=$N/selfplay.log
STATE=$N/selfplay-state           # the adopted bot and its checkpoint, carried across generations
START=${START:-sel10-n2p}
START_PT=${START_PT:-$N/board_net2-d256-e8.pt}
JUDGE_PT=${JUDGE_PT:-$N/judge2-d256-best.pt}  # AUC 0.761 on July: 2 positions a game an epoch
GAMES=${GAMES:-600}; ROLLOUTS=${ROLLOUTS:-8}; TEACH_RATE=${TEACH_RATE:-0.3}; LAM=${LAM:-0.1}; MIX=${MIX:-default 0}
POOL="general@bot-models-exp/pool-racer,general@bot-models-exp/pool-builder,general@bot-models-exp/pool-wide-fellowship,general@bot-models-exp/pool-big-minions,general@bot-models-exp/pool-swarm-grinder"
[ -f $STATE ] || printf "%s\t%s\n" "$START" "$START_PT" > $STATE

h2h() { # <a> <b> <game log>: "<a> vs <b> A x% +- y" and the games' rate
  printf "%s vs %s " $1 $2
  JAVA_OPTS="-Dbvb.gameLog=$3" ./bot-vs-bot.sh --suite decks/real/suite-test.txt --a general@bot-models-exp/$1 \
    --b general@bot-models-exp/$2 --games ${H2H_GAMES:-16} --max-decisions 3000 2>/dev/null \
    | grep -E "A \(general|games in" | sed -E 's/.*(A [0-9.]+% \+- [0-9.]+).*/\1/' | tr '\n' ' '
  echo
}

for gen in $(seq $1 $2); do
  G=$(printf "%02d" $gen)
  read PREV PREV_PT < $STATE
  NAME=sp-g$G
  echo "== generation $gen: $NAME from $PREV ($(date))" | tee -a $LOG
  # the league: the parent itself, the adopted bot before it if any, sel10-net6 and the five style bots
  OLDER=$(ls -d bot-models-exp/sp-g* 2>/dev/null | grep -v "/$PREV$" | tail -1)
  LEAGUE="general@bot-models-exp/$PREV,general@bot-models-exp/sel10-net6,$POOL${OLDER:+,general@$OLDER}"

  # 1. play
  rm -rf $N/sp-teach-$G && mkdir -p $N/sp-teach-$G
  S=$(/tmp/snap-classes.sh)
  t0=$(date +%s)
  JAVA_OPTS="-Dsearch.phaseCut=${PHASE_CUT:-true} -Dsearch.judge=bot-models-exp/$PREV -Dteach.opponents=$LEAGUE -Dteach.noTrain -Xmx20g" SNAP=$S \
    /tmp/run-classes.sh BotTeach --clone --base bot-models-exp/$PREV --value bot-models-exp/value-real \
    --rollouts $ROLLOUTS --depth 1 --move-rollouts $ROLLOUTS --move-depth 1 --max-candidates 8 --teach-rate $TEACH_RATE --max-decisions 3000 \
    --suite decks/real/suite-train.txt --games $GAMES --out $N/sp-teach-$G --boards $N/sp-teach-$G/boards.jsonl \
    2>&1 | grep -v SLF4J | tail -3 >> $LOG
  rm -rf $S
  echo "play: $(wc -l < $N/sp-teach-$G/boards.jsonl) searched boards from $GAMES games, $(( ($(date +%s) - t0) / 60 )) min" | tee -a $LOG
  echo "$GAMES games in $(( $(date +%s) - t0 ))s" >> $LOG

  # 2. learn: this round's boards and the last two rounds' (the search's labels from older play are still about
  # the game; the people's months anchor everything)
  TEACH=$(ls $N/sp-teach-*/boards.jsonl | tail -3 | paste -sd, -)
  (cd nn && uv run python board_net2.py --data $N --init $PREV_PT --teacher $TEACH --teacher-weight ${TEACHER_WEIGHT:-1} --lam $LAM --epochs 1 --lr 1e-4 --people-frac ${PEOPLE_FRAC:-0.25} \
    --save $N/$NAME.pt 2>&1 | grep -E "searched|overrules|^epoch|Error|Traceback") | tee -a $LOG

  # 3. check
  rm -rf bot-models-exp/$NAME && cp -R bot-models-exp/$PREV bot-models-exp/$NAME
  (cd nn && uv run python export_onnx2.py --model $N/$NAME.pt --judge $JUDGE_PT --out ../bot-models-exp/$NAME 2>&1 | grep exported) | tee -a $LOG
  printf '%b\n' "$MIX" > bot-models-exp/$NAME/board-net2-mix
  (cd ~ && HL_OUT=$L/hl HL_EXTRA="--decklists $N/decklists-2022.jsonl" /tmp/human-like.sh $NAME 2>&1 | grep counterfactual) | tee -a $LOG
  GL=$N/sp-games-$G.jsonl; rm -f $GL
  h2h $NAME $PREV $GL | tee -a $LOG
  # the full match against sel10-net6 every NET6_EVERY generations (the parent's decides adoption; the pure check shows
  # the learning) -- a mixed generation's checks were half its time
  [ $((gen % ${NET6_EVERY:-1})) -eq 0 ] && h2h $NAME sel10-net6 $GL | tee -a $LOG
  if [ -n "${PURE_CHECK:-}" ]; then
    # the network on its own, whatever the lineage's mix: is it the one learning?
    rm -rf bot-models-exp/$NAME-pure && mkdir bot-models-exp/$NAME-pure
    for f in bot-models-exp/$NAME/*; do ln -s ../$NAME/$(basename $f) bot-models-exp/$NAME-pure/; done
    rm bot-models-exp/$NAME-pure/board-net2-mix && echo "default 0" > bot-models-exp/$NAME-pure/board-net2-mix
    h2h $NAME-pure sel10-net6 $N/sp-games-$G-pure.jsonl | tee -a $LOG
  fi
  [ -n "${POOL_EVERY:-}" ] && [ $((gen % POOL_EVERY)) -eq 0 ] && $L/scripts/pooleval.sh $NAME | grep "pool mean" | tee -a $LOG

  # adopted: no weaker than the parent (its 95% interval reaches 50%) and no less human-like (within 0.3 points)
  read RATE CI <<< $(grep -E "^$NAME vs $PREV .*A [0-9.]+% \+-" $LOG | tail -1 | sed -E 's/.* A ([0-9.]+)% \+- ([0-9.]+).*/\1 \2/')
  NEW_HL=$(awk -F'\t' '{T++; if($9=="0") A++} END{print 100*A/T}' $L/hl/hl-$NAME.tsv)
  OLD_HL=$(awk -F'\t' '{T++; if($9=="0") A++} END{print 100*A/T}' $L/hl/hl-$PREV.tsv 2>/dev/null || echo 0)
  ADOPT=$(python3 -I -c "print('yes' if ${RATE:-0} + ${CI:-0} >= 50 and $NEW_HL >= $OLD_HL - 0.3 else 'no')")
  echo "adopted: $ADOPT ($NAME vs $PREV ${RATE:-?}% +- ${CI:-?}; July agreement $NEW_HL vs $OLD_HL)" | tee -a $LOG
  [ $ADOPT = yes ] && printf "%s\t%s\n" "$NAME" "$N/$NAME.pt" > $STATE

  # 4. report (this generation's slice of the log)
  awk -v g="== generation $gen:" 'index($0, g) == 1 {on=1} on' $LOG > $N/sp-round-$G.log
  python3 -I scripts/round_report.py --gen $gen --name $NAME --parent $PREV --log $N/sp-round-$G.log --games $GL \
    --hl $L/hl/hl-$NAME.tsv --adopted $ADOPT | tee -a $LOG
done
echo "selfplay done $(date)" | tee -a $LOG
