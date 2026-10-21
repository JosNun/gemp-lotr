# Bot notes

Working notes on generalising the GEMP bot beyond the FotR starters. Everything here
runs headless from this directory; no Docker, no database.

## Running it

```bash
./bot-vs-bot.sh aragorn gandalf --games 200                       # one matchup
./bot-vs-bot.sh --suite starters:test --a general@bot-models-exp/general --b fotr
./bot-vs-bot.sh --list-starters                                   # ~90 starters, by site block
./bot-train.sh --bot general --suite starters:train --out bot-models-exp/x
```

Both scripts rebuild `gemp-lotr-server` when a `.java` file under server, logic or
common is newer than the jar. **Do not edit Java sources while a training run is
going**: the next script call rebuilds the jar under the running JVM, which loads
classes lazily.

Bots (`--a`, `--b`): `fotr` (the shipped bot on `bot-models/`), `general` (the same
bot with its extensions on, see `FotrStarterBot.extended`), `random`, and
`fotr@<dir>` / `general@<dir>` for either on another model directory. A and B swap
decks every game, so a strong deck cannot pass for a strong bot.

`--suite rr:<format>:<deck list file>` plays every pair of the decks in the file.

Suites: `starters` pairs every starter with the next one of the same site block,
under that block's format (`fotr_block`, `ttt_block`, `king_block`, `expanded`).
`starters:train` / `starters:test` are alternate halves of each block. **Train on
`train`, score on `test`** -- the point is decks the bot has never seen. Three SotP
starters name cards the library lacks and are skipped.

## The metric

Head-to-head win rate of the new bot against the current one over `starters:test`,
20 games per matchup (~1,000 games, about 35s), reported with its 95% interval
(+-3 points at 1,000 games). Read it per format too. Against `random` every bot so
far wins 92-96%, which is too close to the ceiling to separate them.

"guessed N answers per game" counts decisions a bot answered at random because no
model or rule covered them, plus answers the engine rejected. Fewer is not the same
as stronger -- see below.

## What was measured (2026-10-17)

| Bot A | Bot B | Held-out decks |
|---|---|---|
| shipped `fotr` | `random` | 93.9% +- 1.5 |
| retrained on `starters:train`, 4 gens x 4000 | shipped | **65.5% +- 2.9** (62-68% every format) |
| same, on FotR starters only | shipped | 11.5% +- 2.0 |
| `general` (extensions + 3 more gens) | shipped | **70.1% +- 2.8** |
| `general` | same models, extensions off | 50.4% +- 3.1 |
| `general`, trees (`--model trees`, same protocol) | `general`, logistic | 46.2% +- 3.1 |
| `general`, trees | shipped | 68.1% +- 2.9 |
| *after the bug fixes below, same protocol (`--init diverse`, 3 gens x 4000):* | | |
| v1 retrained (`bot-models-exp/v1`) | shipped | 69.6% +- 2.9 |
| feature set 2 (`bot-models-exp/v2`) | shipped | 65.6% +- 2.9 |
| feature set 2 | v1 retrained | **49.7% +- 3.1** (48-52% every format) |
| feature set 2, trees | v1 retrained | 47.0% +- 3.1 |
| **feature set 2, final** (race block, vitality fixed; `bot-models-exp/v2`) | v1 retrained | **49.4% +- 3.1** |
| feature set 2, final | shipped | 65.3% +- 3.0 |

- **Diverse decks generalise; more of the same deck does not.** The FotR-only
  retrain ties the shipped bot on its own matchup (51.7%) and collapses elsewhere
  (62% against random). Retraining on varied starters lost nothing on FotR (51.0%)
  and won everywhere else.
- **The extensions halved random answers (4.2 -> 2.0 a game) and won nothing.**
  Neither did three more generations. That points at the model, not the coverage:

## Generalised without losing capability? (the answer for prod)

| Models | Held-out decks vs shipped | FotR starters vs shipped |
|---|---|---|
| diverse retrain, set 1, before the bug fixes (`bot-models-exp/diverse`) | 65.5% +- 2.9 | 51.0% +- 4.9 |
| set 1 retrained after the fixes (`bot-models-exp/v1`) | 69.6% +- 2.9 | **38.3% +- 4.8** |
| **set 2** (`bot-models-exp/v2`) | 65.3% +- 3.0 | **52.8% +- 4.9** |

Feature set 2 is the one that plays any deck without giving up the decks the shipped bot
was trained on: two-thirds of games against it on unseen starters in every format, level
on its own matchup. Set 1, trained further, gained a little elsewhere and lost FotR.
Recipe: `bot-train.sh --bot general --features 2 --init <set-1 models> --suite
starters:train --generations 3 --games 4000`, from a set-1 diverse retrain.

## The linear-model ceiling

Every card-action model scores each option as `sigmoid(w.state + v.card + b)`, and the
state vector is the same for every option of a decision. So `w.state` shifts every
option equally and never changes which one wins: "play X" beats "pass" exactly when
`v.card > 0`, whatever the twilight, wounds or board. The logistic bot is a fixed
card ranking. `TreeModel` (gradient-boosted trees behind the same `SoftClassifier`
interface, `BotTrain --model trees`) can split on state and card together.

**Trees did not help (46.2% +- 3.1 against logistic, same protocol).** The likely reason
is the label, not the model: every decision in a won game is labelled good, so a model
free to use the state mostly learns which *states* win, and the option's own effect is
lost in that noise. The logistic model is forced to put everything in the card term.
Either way, better decisions need a signal about the decision, not about the game.

## Lookahead

### Replays are exact (`BotReplay`)

A game seeded with `GameState.setRandomSeed` and fed the same answers plays out the same
way: **1,030 of 1,030 games replayed identically, all 224,933 decisions** -- type, text
and every parameter, card ids included -- and ended the same way; a different seed
changed 1,021 (the rest end in pregame, corrupted by the bid). Three engine changes:

- All randomness goes through `GameState.getRandom()`: deck shuffle, `GameUtils.getRandomCards`
  (now takes the `Random`), random discard, the seating tie-break. Unseeded, it is as
  random as the `ThreadLocalRandom` it replaced.
- `PhysicalCardImpl` hashes by the id it was *created* with. Card ids change mid-game
  (`assignNewCardId` hides a card's history) and cards are map keys, so the live id
  cannot be the hash; identity hashes made every `HashSet` of cards iterate differently
  each run (first seen: "Choose site to play" listing the same sites in a new order).
- `DefaultActionsEnvironment`'s pending effect results and gathered triggers are
  insertion-ordered: trigger order was the last thing that differed.

Use well-mixed seeds (`SplittableRandom`): `java.util.Random`'s first draws from adjacent
seeds are correlated, and seed + 1 left 5 of 412 shuffles unchanged. The full server
suite passes (6,698 of 6,699; the one failure, `PatchNotesLibraryTest` on a 2026-09-30
note's "Sealed" tag, fails the same on upstream master).

### The teacher (`BotTeach`)

Branches a share of card-action decisions in self-play: each option is taken in
`--rollouts` replays of the game, reseeded and with both decks reshuffled at the branch
(so the deck order is not known; the opponent's hand still is), played out by the bot on
both sides, every option on the same set of reshuffles. Best option(s) are labelled
good, ones that won `--margin` less bad, the rest skipped; the main game continues with
the bot's own choice. Labels go to `<out>/teacher-steps.jsonl`; the card-action models of
`--base` are then refit on them alone. About 5 rollouts a second per core: a rollout is a
replay plus a playout, a few hundred engine decisions, and samples put the time in the
engine (modifier lookups, filters), not the bot. Cutting rollouts off after a few turns
and scoring the position with a value model is the obvious speed-up.

**Does lookahead find better moves?** `--play` pits a bot that branches *every*
card-action decision (8 rollouts per option) against plain v2 on `starters:test`:
**search won 74 of 102, 72.5% +- 8.7** -- the first change in this whole effort to beat
its baseline by more than noise, from card actions alone. That is the ceiling the student
is reaching for. (Its 5,504 labels, in `bot-models-exp/search-play/`, came from the
*test* decks; do not train on them and then score on `starters:test`.)

**The student did not learn it.** Refitting v2's card-action models on 1,242 taught
decisions (3,283 labels, 300 games) lost to v2, 39.1% +- 3.0; L2 (lambda 1, 10) changed
nothing, and refitting only the three models with 500+ labels came back to 47.9%. The
labels are not one-sided (pass judged better 45% of the times it was tried, playing 61%);
a few hundred to 3,000 labels per model replacing fits on ~300k self-play rows is the
likely problem. Worth retrying with far more labels, or mixed with the self-play data.

### Cost against strength

A comparison needs ~1,000 games for +-3 points, so cost per game matters as much as
strength. `BotTeach --play` and `bot-vs-bot.sh` both report CPU-seconds a game.

- Plain v2 models: **0.18 CPU-seconds a game**.
- Value model for cut-off rollouts: `BotProbe --save-value <dir>` fits P(win | state) on
  set 2 with trees: held-out AUC **0.785** against 0.733 for logistic on the same
  features -- a value model has a direct label, which is where trees pay. `--depth N`
  stops each rollout N turns after the branch and scores it with `<base>/value.model`.
  Every rollout still replays the game to the branch, so depth saves only the playout:
  depth 1 ran ~2.4x the rollouts a second of full playouts.

Search player against plain v2 on `starters:test`, 102 games each (+-9 points):

| Search | Won | CPU-seconds a game |
|---|---|---|
| none (plain v2 models) | 50% | 0.18 |
| 2 rollouts, depth 1 | 67.3% | **18.2** |
| 4 rollouts, depth 1 | 67.3% | 54.2 |
| 8 rollouts, depth 1 | 67.6% | 47.4 |
| 8 rollouts, played out | 72.5% | ~145 (estimated) |

At this sample size more rollouts buy nothing, and depth 1 with 2 rollouts is the cheap
end of the search bots -- still ~100x the plain bot.

### Judging cut-off positions: the value model (2026-10-18)

The value model behind the depth-1 numbers above was fit on games of the **test decks
themselves** (`BotProbe --save-value` used to fit on its own suite). `--train-suite` now
fits on one suite and scores on another. Search player vs plain v2, `starters:test`,
2 rollouts, depth 1, 204 games each (+-6.5):

| Value model (fit on) | Opponent's hand | Won |
|---|---|---|
| set 2, trees (test decks -- leaky) | seen | 68.6% |
| set 2, trees (train decks) | seen | 60.8% |
| set 2, trees (train decks) | redealt | 60.3% |
| **set 2, logistic (train decks)** | redealt | **66.7%** |
| set 3, logistic (train decks) | redealt | 55.4% |
| set 3, trees (train decks) | redealt | 57.4% |

- Trees looked better split by game (AUC 0.785 vs 0.733) because they learn each deck's
  win rate. On decks never seen, logistic generalises better, and in search it is worth
  ~6 points. **Fit the value model on decks other than the ones being evaluated**, or it
  carries their old results into the new evaluation.
- Rollouts used to keep the opponent's real hand. It is now shuffled back into their
  unseen cards and redealt at the branch (`--peek` for the old way). Cost: nothing
  measurable. It stays on: a search that knows the tricks coming plays a game nobody can.
- **Set 3** (`KnowledgeFeatures`, appended to set 2): what is left in my deck, what the
  opponent has shown by culture and type, a summary of their unseen cards, and hand clog
  (wrong side for the turn, unaffordable, unique characters already in play or dead).
  Split by game it lifts the AUC 0.85 -> 0.91, almost all of it deck identity; deck-
  disjoint it averages no better than set 2 over four runs (0.756 vs 0.762), and in the
  search it loses ~10 points. Kept in the code, not used. The information is real but a
  value model fit on ~45 starters learns the decks from it rather than the game; it may
  pay with a much larger deck pool.

### Strength against cost, after copies and the engine fix (2026-10-18)

Search player vs plain v2 on `starters:test`, all with `--clone`, the logistic set-2 value
model fit on `starters:train`, the opponent's hand redealt. CPU-seconds a game include the
plain bot's side.

| Search | Games | Won | CPU-s a game |
|---|---|---|---|
| none (plain v2) | -- | 50% | 0.07 |
| 2 rollouts, depth 1 | 204 | 62.7% +- 6.6 | 1.0 |
| **2 rollouts, depth 1, up to 10 options** | 816 | **63.6% +- 3.3** | **1.0** |
| 4 rollouts, depth 1 | 204 | 63.2% +- 6.6 | 3.5 |
| 4 rollouts, depth 2 | 204 | 59.8% +- 6.7 | 4.0 |
| 4 rollouts, played out | 204 | 58.8% +- 6.8 | 9.9 |
| 8 rollouts, depth 1, up to 10 options | 816 | 66.7% +- 3.2 | 5.6 |

The same search cost 18.2 CPU-seconds a game this morning (replays, stats refresh). Depth
beyond one turn and full playouts do not help: the playouts are by the v2 policy, and a
longer one adds its mistakes faster than it adds information. More rollouts buy ~3 points
per 5x. `--max-candidates 10` (all options, in practice) costs nothing extra.

### How Ring-bearers die, and searching assignments (2026-10-18)

`BotDeaths` plays games with the log recorded and reports how Ring-bearers die. Plain v2
against the 29 Shadows-site starters, with four Expanded decks: 434 of 464 games ended
with a dead Ring-bearer, all but one in a skirmish -- 192 with no companion left in play,
133 with one. Fellowships were not starved (3-4 companions in play at most Shadow phases);
they were worn down, and 80 of 173 "move again" choices were made with one or two left.

The search player only branched card actions. `--assign` adds skirmish assignments (the
assignment model's best few legal candidates, plus assigning nothing for the Free
Peoples); `--choices` adds multiple-choice decisions (move again, optional costs) and small
number choices (the burden bid, a uniform 0-4 in the models). Against plain v2 on
`starters:test`, 816 games each:

| Search | Won | CPU-s a game |
|---|---|---|
| card actions | 63.6% +- 3.3 | 1.0 |
| + choices | 66.2% +- 3.2 | 1.3 |
| + assignments | 71.9% +- 3.1 | 1.5 |
| + both | **73.4% +- 3.0** | 1.7 |

Games against plain v2 still end 92% by a killed Ring-bearer; between two such searchers
on four Expanded decks, 14% now end at site 9 (against under 5% before).

### Refitting the value model on search games (no)

`BotTeach --record-values <file>` writes the deciding player's set-2 state at every
card-action decision of the main games, labelled with whether that player won;
`BotProbe --fit <file> --save-value <dir>` fits a value model on it. One round: 1,060
search-vs-search games on `starters:train` (assignments and choices searched, 6.0 CPU-s a
game), 205,123 positions, logistic fit. Search with it against plain v2 on
`starters:test`: **52.3% +- 3.4**, against 73.4% with the value model fit on plain v2
games. Not a bug: on positions from search-vs-search games the old model's AUC is only
0.574 (0.75 on v2 games) and the new one 0.643, with half the spread (sd 0.12 vs 0.22).
Games between two searchers are closer and noisier per position, so the fit comes out
timid, and the positions it judges -- one turn of v2 playout past the branch -- look like v2
games, which the old model knows. A value model for longer plans needs a different
target or model, not just stronger games.

### Distilling the search into a cheap bot (not yet)

The search runs left 130,008 labels. Trained on the 126,725 from `starters:test` and
King-8 decks, scored on `starters:train` (decks it never saw labels for): logistic
**53.0% +- 3.0** against v2 (52.6% leaving out the King block, which King-8 overlaps);
trees 49.3%. A sliver of the search's +17-22 points, at the plain bot's cost (0.25
CPU-seconds a game). Most labels came from 2-rollout searches, which are noisy, and a
best/worst label per option throws away how much better.
 (4 rollouts costing more than 8 is
per-game variance; the decks and game lengths differ run to run.)

### Rollouts from a copy of the game (`BotTeach --clone`)

A replay rollout spends most of its time getting back to the branch. `GameCopier` copies
the live game at the decision instead -- game, feedback with its pending decision,
state, modifiers, actions, stack, the engine's Random -- and `BotGame.resume` plays on
from the copy; the `Branching` bots then see the branch as their first question and
reseed and reshuffle exactly as at the end of a replay. Shared, not copied: everything
reachable from the card library, the formats, the decks and the engine's static fields
(~140,500 objects, walked once at start-up, by identity, because a game's actions point
straight into blueprints). HashMaps and HashSets are rebuilt with the original's table
size so they iterate the same way; lambdas and records go through their constructors;
any other JDK class is refused rather than guessed at. No engine change.

`BotCloneCheck` (1,612 games on `starters:test` and `starters:train`, a random decision
each): a copy continued with the recorded answers asks every recorded question and ends
the same, 1,612 of 1,612; the original, replayed on after a copy of it was played out by
random bots on a reseeded engine, 1,612 of 1,612; a rollout from a copy against the
replay rollout on the same world, step for step and result, 3,224 of 3,224 (800 of them
cut off at depth 1, value-model scores equal). Per rollout, thread CPU: copy **~170 us**,
replay to the branch **~10.8 ms** (decision ~105 on average).

Search player, 2 rollouts, depth 1, 48 games on `starters:test`, two runs each (different
games, so per rollout is the fair figure): replay 43.5 and 22.3 ms of CPU a rollout
(14.5 and 4.1 CPU-seconds a game), copies 5.2 and 6.3 ms (1.1 and 2.7 CPU-seconds a
game); won 72.9% / 64.6% replaying, 66.7% / 68.8% copying, all +-13. What is left is the
playout.

Fragile: it reads JDK fields through `sun.misc.Unsafe` (fine on 21, warns from 23). A new
JDK collection type in game state fails loudly with its path (`CLONE_TRACE=1`); add it to
`GameCopier.RAW_JDK` if it holds no hashes. Run `BotCloneCheck` after engine changes.

## Real games: GEMP's replay summaries (2026-10-18)

https://files.lotrtcgpc.net/replays/ publishes every GEMP game since 2020: monthly
"Replay_Summaries" zips (15-20 MB, one JSON per game: both decklists, format, bids, who went
first, win and loss reasons, sites reached) and full replays (hundreds of MB a month). Six
months of summaries (Sep 2024 - Feb 2025, ~44,000 games) are in `~/dev/gemp-replays/`
(outside the repo). `~/dev/gemp-replays/tools/extract.py <out> <summary dirs>` writes:

- `decks/real/<format>/<id>.txt` -- every distinct deck (4,638 Movie Block, 1,784 Expanded, ...),
  gitignored: they are other players' decks;
- `decks/real/<format>.csv` -- each deck's human record (games, wins, how its losses ended);
- `decks/real/games.csv` -- every game between two library decks.

`decks/real/suite-train.txt` / `suite-test.txt` pair decks with 3+ human games, split by deck
id, 800 matchups each (Movie Block and Expanded). 19 decks name cards the library lacks and
are dropped (DeckCheck).

**How real games end** (Feb 2025; ~45% are conceded, these are the rest):

| Format | Ring-bearer killed | Survived to site 9 | Corrupted |
|---|---|---|---|
| Movie Block | 44% | 37% | 18% |
| Expanded | 50% | 27% | 23% |
| bots, Expanded field, search (before the guard) | 85% | 11% | 3% |

("Other player won" in GEMP's records is the loser's side of a site-9 win.) Humans bid low:
Movie Block 0 67%, 1 24%, 2 8%.

**What real decks exposed:** between the 16 most-played Movie Block decks, 72% of plain-bot
games ended with a Ring-bearer corrupted, most on turn 1 -- by its own player. The uniform 0-4
burden bid corrupted Galadriel, Bearer of Wisdom (resistance 3) before the game started, and
the models paid burden costs (an Elf deck's retrieval, The One Ring's +1 strength while a
Huorn fought) until the Ring-bearer corrupted. Fixed by a human-like bid and `Guarded` (see the
commit): corruption 72% -> 41% -> 13% of games (humans 9.5%); guarded beats unguarded v2 66.7%
+- 2.3 on held-out real decks and ties on the starters. Bot kind `safe@<dir>`; BotTeach's
rollout policy is guarded unless `BOT_UNGUARDED` is set.

### Training on real decks

The ~45 starters were the whole training pool; the summaries have thousands of decks.
Round 1, Movie Block and Expanded (`suite-train.txt`, 800 matchups):

- **Policy** (`bot-train.sh --features 2 --init v2 --suite decks/real/suite-train.txt
  --generations 3 --games 4000`, 5 minutes) -> `v2-real`. Guarded, against guarded v2:
  **68.3% +- 1.6** on held-out real decks, and **54.7% +- 3.1 on `starters:test`** -- better
  than v2 on the decks v2 was trained for. Variety was the missing ingredient.
- **Value model** fit on real-deck games of the train half (guarded v2 playing): held-out
  AUC 0.814 (logistic, set 2) on the test half. Search with it against plain v2 on held-out
  real decks: **77.8% +- 2.9**, against 71.2% +- 3.2 with the starter-fit value model.

Round 2, every format with enough decks (`suite-all-train.txt`: Fellowship, Towers, King,
Movie, Expanded; PC Movie dropped -- its errata mapping fails on these decks): `v3-real` from
`v2-real`, 4 generations x 8,000 games. It **ties** `v2-real` (48.5% +- 2.5 held-out real,
48.6% +- 3.1 starters): the variety was the gain, more of it and more generations add nothing
to a linear per-option model. A value model fit on the all-format games ties the round-1 one
in search (69.7% vs 67.4% +- 3.3). Card ids in the summaries carry
cosmetic suffixes (`*` foil, `T` tengwar); they are stripped from the deck files.

**Realism, guarded `v2-real` on the 16 most-played Movie Block decks** (4,800 games):
median 7 turns (4 before), corruption 10% of games, kills 89%, site 9 under 1%. Human decided
Movie Block games: kills 44%, site 9 37%, corruption 18%; 34% end at site 9 and ~70% at site 6
or later, where bot games end around site 5. The gap left is the Free Peoples side:
fellowships are worn down and never reach site 9.

**A lever for that, not built:** the full replays (`GEMP-LOTR-Replays_<month>.zip`, ~12,000
games and 3 GB a month, to 2022-11) are each player's event stream: every decision put to
them (`type="D"`, with its parameters), every card placement, message and stats update. The
answers can be read off what follows. Imitating human decisions needs a state tracker that
rebuilds the board from those events for the feature extractors -- days of work, but it is
the only source of good Free Peoples play there is. (Tried the other way round, by replaying the
games in the engine instead of tracking them: see "Re-creating human games" below.)

Also tried on held-out real decks (`v2-real` searching, against plain `v2-real`, 1,584
games each, +- 2.3): value model with trees fit on real decks 65.1% vs logistic 69.1% --
logistic still wins; searching single-card picks from a list (`--card-picks`) 69.3% vs 69.1%
at +30% CPU -- no gain, opt-in.

**The value models have the race backwards.** Probed on recorded positions, both say the
further your own fellowship has got, the less likely you win: `value-real` 59% at site 2,
42% at site 8 (`value-s2-logistic` 65% -> 19%), and the further the opponent, the more likely.
In bot games advancing correlates with dying -- later sites are deadlier, moving feeds the
Shadow twilight -- and outcome-fit models learn exactly that. A progress term for cut-off
values (`--progress-weight w`: + w x ((my site/9)^2 - (their site/9)^2)) changed nothing:
69.9% (w 0.2) and 67.4% (0.5) against 69.1%, site-9 wins ~2% of search's wins throughout.
Steering one-turn lookahead does not keep a fellowship alive; the attrition happens in play
the value model never sees. Off by default.

## Re-creating human games in the engine (`HumanReplay`, 2026-10-19)

Instead of a state tracker: play each recorded human game again inside our engine, both seats
answered with the humans' answers and every shuffle forced to deal what they drew. Where the
engine asks the question the human was asked, the engine's own state at that point is the
board they saw, and the feature extractors run on it as on any bot game.

```bash
python3 -I scripts/replay_manifest.py <extract> <manifests> --format "Fellowship block"
./replay.sh HumanReplay --manifests <manifests> --jobs 2 --report r.tsv [--steps human.jsonl]
python3 -I scripts/replay_buckets.py r.tsv <manifests>     # why games diverged, cards named
```

`replay.sh` builds with `mvn package` and runs this checkout's jars ahead of the shared
classpath (no `mvn install`). `--trace` prints every comparison; `--only <name>` one game.

**How it works.**
- *Card ids line up by construction.* The 2020 server, like today's engine, numbers cards as it
  creates them: player by player in HashMap order of the names, sites, then the draw deck in
  deck-list order, then Ring-bearer and Ring; a card gets a new id each time it enters play.
  So a card's original id is its slot in its owner's deck list, and the 2020 summaries give no
  decklists at all (only ids; GEMP wrote `deckReadout` from 2022): the decks are rebuilt from
  the ids each recording shows (a drawn card keeps its id in hand). Cards never seen under
  their original id are inferred (played straight from the deck, or listed by a deck search
  more often than the seen copies explain) or filled with an inert card (The One Ring);
  unplayed sites with any site of the missing number. Built from the real player names, the
  game hands out the 2020 ids: every reproduced decision below matched them exactly.
- *Shuffles* (`GameState.DeckShuffler`, the one engine change: null by default, consulted in
  `shuffleDeck`): the cards each player's recording shows leaving the deck after this point go
  on top, in that order. Ties in the bid are settled by trying seeds until the seating shuffle
  picks the player who chose.
- *Answers* are not recorded. Each plausible answer is played on a copy of the game
  (`GameCopier`), and the events it produces -- cards placed, removed, moved, tokens,
  assignments, skirmishes, phases, twilight, and the next questions with their options --
  are held against both recordings (as a multiset per decision: the 2020 server batched some
  differently); the answer that reproduces them is kept. A decision counts as reproduced when
  the engine asked the same question (type, text as far as it survived, options) and an answer
  reproduced everything up to the next question. Candidates come from what the recordings show
  next (card ids, blueprints, AA events, numbers in messages), then the rest of the options.
  A random discard or reveal is matched by reseeding the engine's Random (357 answers).
- *Accepted differences*, each found in the data: messages and stats are not compared, nor CAC
  (the 2020 engine also flashed a card paying a cost on itself); a missing or extra CA
  (Ford of Bruinen flashed its per-turn reset then) does not break a match; repeated phase and
  twilight updates are dropped; adventure-deck and end-of-game deck reveals are today's only.
  Questions only today's engine asks are answered for the player and not counted: deck
  inspections before a search, card-action prompts with nothing to play (the human's
  auto-pass skipped them), searches with one card to take, ordering two required triggers
  (55,050 of them). Texts were reworded freely since 2020 ("Choose characters to wound" /
  "Choose cards to wound", "Frór" mangled to "Fr??r" by the 2020 files' encoding): card
  selections, multiple and arbitrary choices are compared by their options only, action texts
  by card id, and searches may now stop short (min 0).

**Fellowship block, March 2020** (1,575 games with both recordings; 80 more had no stats to read
the deck sizes from), 12 minutes on 2 threads:

| | |
|---|---|
| human decisions reproduced | **198,067 of 454,632 (43.6%)**; with something to choose 173,008 of 398,352 (43.4%) |
| games reproduced to the end | **327 of 1,575 (20.8%)**; median game 42% of its decisions; 694 games at least half |
| asked with other options than the human had (counted) | 1,874 |
| another answer fitted the recording as well (not written as data) | 8,839 |

Decisions reproduced before the first divergence, games: 1-9: 53, 10-24: 105, 25-49: 233,
50-99: 406, 100-199: 451, 200-399: 302, 400+: 25.

**Why games diverge** (the first divergence of the 1,248 that did; decisions they left unreproduced):

| Cause | Games | Decisions lost |
|---|---|---|
| engine asks a question the human was not asked | 269 | 60,418 |
| human was asked a question the engine does not ask | 236 | 47,144 |
| same point, a different question | 161 | 33,891 |
| twilight differs (TP) | 105 | 21,277 |
| a token (wound/burden) instead of the recorded question, or the reverse | 137 | 28,796 |
| other event differences (removals, phases, placements, tokens), engine ending early | 340 | 65,039 |

The cards named in the first divergence: Blade Tip 197 games, Gimli, Son of Glóin 101, Frodo
(1_289) 62, Sting 53, Bill the Pony 48, Bilbo (2_96) 45, The Bridge of Khazad-dûm 37, ... a
long tail. The ones looked at are all **the card behaving differently in today's engine than in
2020's**, i.e. errata in the broad sense -- bug fixes and rules corrections, not printed errata
(Fellowship block plays the original cards):
- Blade Tip and Black Breath: the 2020 engine kept offering "transfer this condition from your
  support area" after it had been transferred, and re-fired the trigger; today it does not.
- Bill the Pony: moving to an underground site, 2020 still applied his shadow -1 to the move's
  twilight (TP 7 recorded, 8 now); several other off-by-one twilight cases.
- Gimli: after he was used in a skirmish and the Shadow player passed, 2020 ended the skirmish;
  today the Free Peoples player is asked again.
- Aragorn (1_365, "add 2 burdens to prevent exhausting"), Galdor, Úlairë Nertëa: a question asked
  now that was not then, or the reverse.
That none of this can be read off the repository: every set 1-3 card was rewritten from Java to
hjson in 2022-24, so git history marks them all as changed. Divergences that are the harness's
own (a wrong inference) were fixed one at a time until the remaining ones were card behaviour;
the 13 copier failures (a deck view inside a pending decision) were fixed too.

**Training data.** `--steps` writes each reproduced decision in BotTeach's teacher-label form
(`LearningStep.toJson`, feature set 2 state, the board before the answer): card-action and
multiple choices one row per option, reward 10 for the human's and 0 for the others; other
types the human's answer only (reward 10); not decisions with nothing to choose, ambiguous
answers or ACTION_CHOICE. March 2020 Fellowship block: **458,737 rows** (440 MB,
`~/dev/gemp-replays/training/fotr-2020-03.jsonl`, outside the repo: other players' games):
101,730 card-action choices (67,380 as Free Peoples), 39,805 card selections, 11,937
multiple choices, 8,073 arbitrary-card choices, 7,428 assignments, 3,643 skirmish
selections, 471 numbers. `BotTeach --train-only --steps` reads it.

**A first look at using it (not tuned):** refitting `v2-real`'s models on all of it breaks the
bot (5.3% +- 1.4 against v2-real): the types written with positives only (assignments,
selections) fit a model that says yes to everything, and assigning everything to the
Ring-bearer loses on turn 1. Refitting only the card-action models (399,317 rows) loses too:
**44.0% +- 1.8** against guarded v2-real on held-out real Fellowship block decks
(`suite-all-test`, fotr_block, 3,000 games), 34.1% +- 2.9 on `starters:test`. The same
result as the teacher labels: a per-option logistic model refit on labels about the choice,
replacing one fit on ~all self-play outcomes, does not get better (see "The linear-model
ceiling"). The data is the valuable part; using it needs a model that scores state and option
together, or the labels mixed with self-play, or imitation as an auxiliary loss.

**Trees on the human labels work.** Refit only the card-action models of `v2-real` with
trees (`BotTeach --train-only --steps card-actions.jsonl --model trees --min-steps 500`,
375,443 rows; the logistic refit above lost 44.0%): **55.2% +- 1.8** against guarded
`v2-real` on held-out real Fellowship block decks (3,000 games), 46.2% +- 3.1 on
`starters:test` (the human data is all Fellowship block). It plays more like people: self-play
on the held-out FotR decks reaches site 9 in 9.8% of games against 3.3% for `v2-real`, and
games run a median 10 turns against 8 (human decided FotR games: site 9 48%, killed 38%,
corrupted 5%). Adding the human multiple-choice labels (12,588 "another move?", 6,108
mulligans) to the trees made it worse: 44.9% +- 1.8 against the card-action-only refit --
human move decisions do not suit how the bot plays the rest of the game. Models in
`bot-models-exp/human-trees` (card actions).

**More months.** April 2020 reproduced better than March: 51.4% of 1,328,111 human decisions
(4,627 games, 1,252 to the end), 1.59 M training rows; May is processed too
(`~/dev/gemp-replays/training/fotr-2020-0{3,4,5}.jsonl`). Trees on March + April card actions
(1,683,187 rows, `-Dtrees.maxRows=600000`, ~40 minutes) -> `human-trees-2m`: 52.1% +- 1.8
against the March-only `human-trees` on held-out real FotR decks, 49.7% +- 3.1 on the
starters, site 9 in 7.6% of self-play games -- a little stronger, no more human-like.
Making TreeModel's row cap configurable changed its computed serialVersionUID and broke
loading every saved tree model; it is pinned now to the old value.

**Can it scale?** The pipeline does: ~0.9 CPU-seconds a game, and the full replays run
2020-03 .. 2022-11 at 4,000-12,000 games a month -- millions of reproduced human decisions if
other months and formats reproduce as well as this one (not measured). What limits it is the
2020-22 engine's card behaviour: each fix since ends the games that touch that card, and the
gain comes card by card (the five cards named most appear in 461 of the 1,248 first
divergences). Two cheap levers: replay later months (likely closer to today's engine; not
measured), and resync instead of stopping -- every recorded event names card ids and
blueprints, so a decision's state is still right if the divergence touched nothing the
features read. Not tried: other formats (the manifests support any; set 4+ cards add their own
differences), and reseeding the engine for random draws other than from hands.

## A cheap bot instead of search (2026-10-19)

The goal is now the bot itself, cheap to run.

**Searching only the big decisions** (`BotTeach --no-card-search`: assignments and choices searched, card actions
left to the models) costs 0.3-1 CPU-seconds a game and adds nothing: `human22-ca-as` with it beats plain `v2-real`
57.8% (2 rollouts), 56.9% (8), 56.7% (8, rollouts cut off 3 turns on) -- plain `human22-ca-as` is 58.0%. The full
search's strength is in the card actions.

**Distilling the search** (BotTeach teacher mode, base `human22-ca-as`, 8 rollouts, depth 1, `value-real`, 10% of
card-action decisions, 3,000 games of `suite-train`: 24,273 decisions, 40,463 labels, 21 minutes). Trees refit on
the human labels plus these: x1 60.7%, x10 (teacher labels repeated) **61.6% +- 1.7** against plain `v2-real`
(`human22-ca-as` 58.7% in the same run), and `distill22-x10` beats `human22-ca-as` **55.1% +- 1.7** head to head.

**How the bot uses each card** (`-DcardUse=<blueprint>|all` on BotVsBot, `CardUse`; `scripts/card_use.py` holds it
against the human rate per card from the replays). The One Ring, Answer To All Riddles (4_1): people put it on in
2.1% of 13,381 offers, mostly losing the skirmish by 3+ (where wounds would kill the Ring-bearer); the bots 0%
(`v2-real`), 1.4%, 3.2% -- the same places; none uses it when -1 down, where people do (2.8%). Across 521 cards the
gaps are elsewhere: Hand Axe (discard it for archery +1) 58% against people's 15%; Double Shot (spot an Elf archer
for archery +1) 1% against 53%; The One Ring, The Ring of Rings (9_1) 0% of 38,844 against 5.2%; site abilities
1-4% against 22-39%.

**Ability features** (card feature set 3, `AbilityFeatures`): each card's abilities -- activated, event, response,
optional trigger -- read from the card definitions, their cost types (burdens, discard self, exert, twilight...) and
effect types (strength, archery total, wounds, plays a card...) by amount, appended to set 2's card vector. Self-play
with them (`bot-train.sh --bot general --features 3 --init v2-real`, 3 x 4,000 games) ties `v2-real` (50.5% +-
1.7); the human + teacher refit on them ties `distill22-x10` (50.9%) and beats `v2-real` 63.3%. Hand Axe fell to 27%
(people 15%); Double Shot stayed at 2%.

**Why Double Shot stays at 2%** (`ScoreProbe` on the 511 human decisions where it was offered; people took it 54%):
`v2-real` ranks it first 82% of the time, `human22-ca-as` 25%, `distill22-x1` 13%, `distill22-x10` 0.4% -- though 13
of its 14 teacher labels say play it. The card-action model is a per-option yes/no fit over rows pooled from every
decision, so ten copies of the teacher labels (mostly "pass was best") move the base rate of passing, not what the
model knows about the card. Top-1 agreement with people on November card actions (in-sample for the human models):
`v2-real` 43.2%, `human22-ca-as` 64.4%, `distill22-x10` 62.2%.

**Agreement with people, held out** (`ScoreProbe <models> all`, trained on October card actions, scored on 58,899
November decisions): trees as before 64.1% (54.7% where people played a card, 82.2% where they passed), with ability
features 64.7%, with 300 trees of 64 leaves (`-Dtrees.n=300 -Dtrees.nodes=64`) **66.6%** (58.0% / 83.2%). Held-out
matches in-sample (~64%): the trees were short of capacity, not data. ScoreProbe scores the one card-action model
(AnyPhaseCardActionTrainer); the bot answers most card actions with the per-phase models first. Making the one model
answer everything (`any-phase-first` file in the model directory) plays worse: 33.9% against 40.1% for the same
October models, both against `distill22-x10`.

**Bigger trees in play** (October + November, card actions and assignments): `human22-big` (set 2) beats
`human22-ca-as` 53.2% and `distill22-x10` 48.1%; `human3-big` (set 3) 51.5% / 49.0% -- the ability features add
nothing in play.

**More months.** August and September 2022 Movie Block replayed: 2,841 games, 285,390 decisions reproduced (31-32%).
Four months' card actions and assignments (1,404,402 rows) plus the teacher labels x3, big trees, 900,000 rows a
model -> `h4m-big-x3`: **52.1% +- 1.7** against `distill22-x10`, 54.2% against `human22-big`, 64.6% against `v2-real`.
The best plain bot so far, by a little.

**Where a plain game's time goes** (JFR, 800 games of `distill22-x10`, 0.1 CPU-seconds each): the engine's modifier
and filter queries ~45%, tree predictions ~15%, `Guarded`'s game copies ~10%. Refitting was the slow part of the
loop -- 17-60 minutes a model set, ~2.7 of 8 cores busy -- so `BotTeach --train-only` now fits the models side by
side, biggest first.

**Distilling again, from the better bot.** Teacher mode with `h4m-big-x3` as base (8 rollouts, 15% of card
actions, 5,000 games of `suite-train`: 63,743 decisions, 106,891 labels, 80 minutes). Refit on the four months plus
both rounds' labels x3 (1,846,464 rows; 38 minutes with the models fit side by side) -> `h4m-d2`: **53.2% +- 1.7**
against `h4m-big-x3`, 54.3% against `distill22-x10`, **67.8%** against `v2-real`. Each round's search starts from a
better policy (rollouts and the bot that branches), so its labels are better.

**A third round plateaus.** Labels from `h4m-d2` (5,000 games, 107,717 labels) -> `h4m-d3`: 52.3% +- 1.7 against
`h4m-d2` at 4 games a matchup, **51.1% +- 1.2** at 8 -- noise; 67.7% against `v2-real`, as `h4m-d2`.

**Assignment search had been off.** BotTeach searched skirmish assignments only if its bot was a FotrStarterBot;
since main games and rollouts are guarded by default it is a Guarded, so `--assign` did nothing in every run after
that -- the human value model, the move and partial searches, the distillation teachers. No log
since has an "assignment decisions searched" line. Fixed (BotTeach unwraps the guard). The +7 points it was measured
at are from before.

**Teaching assignments.** Teacher mode now labels assignments too (`--assign --assign-teach-rate 1 --teach-rate 0`:
the model's best five legal assignments and, for the Free Peoples, none, 8 rollouts each). From `h4m-d3`, 4,000
games: 33,585 assignments taught, 97,348 labels, ~40 minutes. Refit on everything before plus these x3 (2,461,659 rows,
60 minutes) -> `h4m-d4a`: **51.6% +- 1.2** against `h4m-d3` (8 games a matchup), **70.2% +- 1.6** against `v2-real`.
The best plain bot.

**All of 2022 adds nothing.** January to November 2022 Movie Block replayed: 19,439 games, 2,023,087 human
decisions reproduced (30.7-33.5% a month). All assignment rows and 60% of the card-action rows (2,742,983; all of it
did not fit in 28 GB with the fits side by side -- `-Dtrain.threads=3` now limits them, and a failed fit is reported
instead of the muted exit that hid it) plus the four rounds of teacher labels x3, 900,000 rows a model, 60 minutes
-> `h11m-d5`: **49.7% +- 1.2** against `h4m-d4a`, 68.3% against `v2-real`. Past four months, more people's games do
not make a stronger bot; `h4m-d4a` stays the best.

**A value model from the better bot ties.** Fit on 597,689 positions of `h4m-d4a` playing `suite-train`
(`value-d4a`; held-out AUC 0.752 logistic set 2, trees 0.773): search over `h4m-d4a` with it 67.5% +- 2.3 against
plain `h4m-d4a`, with `value-real` 67.8% (assignment search working again: ~7,900 assignments searched in each). Search
is still worth ~18 points over the best plain bot, at 5.5-6 CPU-seconds a game; distillation has taken part of it.

**Cheaper search, by fewer candidates.** Searching, the bot's own choice is now always a candidate; with
`--max-candidates 2` the search weighs only it against passing. Over `h4m-d4a`, against plain `h4m-d4a`: 2
candidates 57.4% +- 2.4 at 2.2 CPU-seconds a game, 3 (one random other) 64.3% at 4.2, up to 10 (in practice all)
67.8% at 6.0. Most of the search's worth is in options the model would not have picked.

**Weighting the teacher more ties.** All four rounds of teacher labels x5 instead of x3 (3,166,497 rows) ->
`h4m-x5`: 49.4% +- 1.2 against `h4m-d4a`. With round 3 (51.1%) and all of 2022 (49.7%), the per-option trees have
stopped improving with more or heavier labels; `h4m-d4a` (70.2% against `v2-real`) is where this recipe tops out.

**The 300,000-row cap.** `AbstractTrainer` sampled every model down to 300,000 rows before TreeModel's own cap, so
every human-label refit (1.4-3.8 M rows) had been fit on 300,000 of them whatever `-Dtrees.maxRows` said. Now
`-Dtrain.maxRows`. At 900,000 rows `h4m-d4a`'s recipe ties it: 50.9% +- 1.2 (87 minutes). The rows were not the ceiling
either.

**Comparing options within a decision** (`ListwiseTreeModel`, `BotTeach --model listwise`): boosted regression trees
fit to the softmax over each decision's options (each round fits `chosen - P(option | decision)`), decisions sampled
whole; the steps are grouped into decisions as they are read (one decision's rows are written together). Only the
any-phase card-action model is fit this way (its decisions hold every option and pass). On October, held out on
November (300 trees, 64 leaves): agreement 66.9% (yes/no trees 66.6%), 59.8% where people played a card (58.0%), 80.6%
where they passed (83.2%); it passes 36.7% of the time against people's 34.1% (39.2%). Double Shot, on its 511 human
decisions: ranked first **80.8%** of the time, against 25% for the yes/no trees -- the base rate of passing is gone.

**...but it does not play better.** The full recipe of `h4m-d4a` with the any-phase model listwise (`h4m-lw`, 31
minutes), against `h4m-d4a`, 8 games a matchup: listwise answering every card action (`any-phase-first`) **45.0% +-
1.2**; the per-phase yes/no models first and listwise where they do not answer 49.7%; `h4m-d4a` with only its
any-phase model swapped for the listwise one 50.4%. The per-phase models, with their own features (the skirmishing
companion and minions, the bearer of an item), play better than one model over all card actions however it is fit,
and agreeing with people on a held-out month (66.9% against 66.6%) does not track strength. Double Shot ranked right
and passing in proportion did not make a stronger bot.

**Visibility or specialisation?** The per-phase models see a little the any-phase model did not: an item's class and
its holder, whether a card can touch anyone in the current skirmish (10 numbers, `-DanyPhase.rich`, an
`any-phase-rich` file in the model directory). And each is its own model. Listwise any-phase model answering every
card action, everything else `h4m-d4a`'s (`-Dtrain.only=AnyPhaseCardActionTrainer --base h4m-d4a`), against
`h4m-d4a`: as before 45.0%; with the 10 numbers (`d4a-lwr`) 46.3% +- 1.2; with them and one listwise model per
phase context (`-DanyPhase.perContext`: fellowship, shadow, maneuver, archery, assignment, skirmish, regroup, response,
picked from the prompt; `d4a-lwrc`, 70 minutes) **52.1% +- 1.2**. Specialisation was the gap; per phase, comparing a
decision's options against each other beats the yes/no per-phase models. `d4a-lwrc` is the best plain bot, and one
scorer per phase for every option -- play, use, pass -- is what a phase planner needs.

**Which phase the search's points come from** (`-Dsearch.phases`, card actions only, over `h4m-d4a` against plain
`h4m-d4a`, 1,600 games each; all phases with assignments and choices 67.8%): Fellowship **44.5%** -- worse than not
searching; Shadow **57.8%** at 1.3 CPU-seconds a game; skirmish 49.8% (few prompts matched); maneuver, archery,
assignment, regroup 54.3%; responses ("Choose action to play", "Optional responses") 56.1%. One turn on, the value
model misjudges Fellowship plays.

**A Shadow-phase planner judged at the end of the phase fails.** Rollouts cut where the phase they branched in ends
(`-Dsearch.phaseCut`), scored by `value-real`: 30.8% +- 2.3 against plain `d4a-lwrc` (2 or 4 rollouts alike), where
cutting one turn on gets 57.8%. Fit on outcomes, the value model rates minions in hand above minions in play mid-turn;
a Shadow phase can only be judged once its skirmishes are fought.

**Longer horizons do not fix Fellowship search.** Fellowship-only search over `d4a-lwrc` against plain `d4a-lwrc`,
rollouts cut 2 turns on: 46.3% +- 2.4 (2.5 CPU-seconds a game); 3 turns: 44.5% (3.2) -- as at 1 turn. The Free Peoples'
board lasts, so a one-turn look was the suspect; it is not the problem. More likely the noise: Fellowship decisions offer
many options of near-equal worth, and two rollouts apiece pick among them nearly at random, where the model picks as
people do. Yet the Fellowship labels help a refit: dropping them (30,348 of round 3's 107,717, and the like from the
others) from `d4a-lwrc`'s recipe -> 48.2% +- 1.2 against `d4a-lwrc`. Acted on one at a time the search's noise costs
games; averaged over a hundred thousand labels it washes out and its signal stays.

**Card use by type** (`scripts/card_use.py`; plain `d4a-lwrc` in 3,200 Movie Block games against people in 2022's
replays, share of offers taken): Shadow minions 32.0% / 31.9%, Shadow conditions 41.5% / 43.0%, Shadow possessions 56.5%
/ 49.6%, Shadow events 36.4% / 41.3%, Free Peoples companions 28.3% / 26.6%, possessions 16.4% / 17.8%, conditions
28.8% / 29.3%, events 24.8% / 30.5%, site abilities 37.2% / 26.9%. Conditions -- lasting Shadow cards a one-turn view
could undervalue -- are played as people play them; events a little less, sites more.

**Distilling from the per-phase listwise bot.** Teacher mode from `d4a-lwrc` (all phases, 8 rollouts, 15% of card
actions, 5,000 games: 108,230 labels, ~95 minutes), then the per-phase listwise any-phase model refit on the four months
plus every round's labels x3 (`-Dtrain.only=AnyPhaseCardActionTrainer`, base `d4a-lwrc`, 71 minutes) -> `d4a-lwrc-d5`:
**51.9% +- 1.2** against `d4a-lwrc`, **71.4% +- 1.6** against `v2-real`. The best plain bot.

## The bot against people, on people's games (2026-10-20)

`HumanReplay --counterfactual <bot>` (`Counterfactual`): at a sampled share of the reproduced decisions of re-created
human games, the bot is asked what it would do; where it differs from the human, both answers are played on copies of
the game (the replay's forced draws off, both decks reshuffled, the bot finishing for both players), 8 times each on
the same futures, and the deciding player's wins compared. `scripts/counterfactual_report.py` sums it by kind of
decision and by the human's skill (their 2022 Movie Block win rate, 20+ games). The continuation is the bot's, which
credits answers that suit its own play; it is still the nearest thing to the bot against people without people.

November 2022 Movie Block, `d4a-lwrc-d5`, 20% of decisions: 32,120 asked, the bot agreed with the human on 59.0%;
13,175 disagreements played out: **bot - human -0.9 +- 0.3 points**. Against strong players (60%+ wins) -1.0 +- 0.6,
middling -1.1 +- 0.4, weak -0.4 +- 0.5. By kind of decision (bot - human where they differ; net = summed over them):

| Decision | Agreed | Bot - human | Net |
|---|---|---|---|
| "Do you want to make another move?" | 69.2% | **-6.0 +- 3.3** | -19.5 |
| Free Peoples assignment | 46.7% | -2.6 +- 1.3 | -16.9 |
| Card action, "Choose action to play or Pass" (skirmishes) | 76.3% | -1.5 +- 1.2 | -12.1 |
| Reconcile: card to discard | 12.4% | -0.5 +- 0.7 | -10.2 |
| Starting fellowship | 36.9% | -1.1 +- 1.1 | -8.5 |
| Choose cards to wound | 31.6% | -3.2 +- 2.8 | -5.8 |
| Choose next skirmish to resolve | 48.3% | -1.5 +- 1.2 | -5.4 |
| Card action, Shadow phase | 59.0% | +0.4 +- 0.8 | +6.1 |
| Card action, Fellowship phase | 61.9% | +0.3 +- 0.5 | +6.5 |

By type: card selections -0.8 (net -41.6), choices -3.2 (-28.1), assignments -2.4, card actions -0.3. The card
actions -- where the human labels, search labels and listwise fits went -- are at people's level; what loses is what
was never fit to people. "Move again?": in 298 of the 323 disagreements the human moved and the bot would have
stopped, and moving was worth 5.6 points; the bot said yes in 5 of 727 agreements. People race; the bot does not.

**Fixing the worst one.** `AnotherMoveTrainer` refit as trees on the 37,454 human "another move?" rows of August to
November (`-Dtrain.only=AnotherMoveTrainer`, base `d4a-lwrc-d5`) -> `d5-move`: **52.7% +- 1.2** against
`d4a-lwrc-d5`. The Free Peoples assignment model fit listwise (decisions grouped, human + teacher labels): 49.7%, a
tie; both together 52.3%. `d5-move` is the best plain bot.

**Card selections from people.** The card-selection and arbitrary-card models (which companion to exert or wound,
skirmish order, reconcile discard, starting fellowship, sanctuary heals, items to attach...) had only ever been fit to
self-play. The human labels record the cards not chosen as well (`CardSelectionAction`); six models used them as
negatives, the rest skip them, rightly for self-play outcomes, where an unchosen card's worth is unknown.
`-Dtrain.notChosen` uses them for all. Refit as trees on August-November's 188,995 human selection rows (13 models,
2,600-40,000 steps each; `-Dtrain.only=...`, base `d5-move`) -> `d5-move-sel`: **56.1% +- 1.2** against `d5-move`.
The biggest step since the human assignments.

## Human Movie Block games, and assignments learned from them (2026-10-19)

**2022 replays.** October and November 2022 (`GEMP-LOTR-Replays_2022-1{0,1}.zip`, ~340 MB
each, unpacked 6 GB, deleted once the manifests were built). Movie Block: 3,696 games,
1,196,110 recorded human decisions, 385,902 reproduced (32%), 554 games to the end. At first
only 27.8% reproduced (first 100 games): the 2022 server asked questions today's engine
settles by itself -- "Choose a card to play" with one card, "Choose number of Elf companions
you wish to spot", "Play Gollum from your deck / discard pile" -- and the replay stopped
there. `HumanReplay` now passes over such a recorded question when everything after it
matches (`recordedOnlyEvent`): 27.8% -> 38.5% on those 100 games. **Expanded reproduces
3.4%**: from set 11 the player chooses which site to play, and the engine builds a different
site 1 than the recording; not fixed.

**Assignments, with the ones turned down.** The first try at human labels wrote each human
assignment alone, positive, and the model learned to say yes to everything (assigning every
minion to the Ring-bearer, losing on turn 1). `TrainingSteps` now writes the human's
assignment and up to 12 others the rules allowed (`AbstractAssignmentTrainer.allAnswers`),
labelled 0; an answer the enumeration does not make, or a forced one, is skipped. Two months:
95,745 Free Peoples rows (~12,000 decisions), 1,100 Shadow (mostly forced, so skipped).

**Trees on it** (`BotTeach --train-only --model trees --min-steps 500`, base `v2-real`,
710,076 card-action rows + 96,845 assignment rows, `-Dtrees.maxRows=600000`, 17 minutes).
Plain, against guarded `v2-real`, 4 games a matchup:

| Models refit | Held-out Movie/Expanded (3,168) | Held-out FotR (1,200) |
|---|---|---|
| card actions only (`human22-ca`) | 50.7% +- 1.7 | |
| assignments only (`human22-as`) | 56.0% +- 1.7 | |
| both (`human22-ca-as`) | **58.0% +- 1.7** | 55.8% +- 2.8 |

The assignments are the gain; card play from 2022 Movie Block adds a little with them and
nothing alone (2020 FotR card play gave 55.2% on FotR). Searching (`--assign --choices`,
value-real) against plain `v2-real`: 69.6% with `human22-as`, 66.7% with `human22-ca-as`,
68.9% with `v2-real` -- search already tries the top few assignments, so it gains little.

**Endings.** The 16 most-played Movie Block decks, round robin, 20 games a pair,
both sides searching with `human22-ca-as`: corruption 16.5% (people 18%), site 9 6.1%
(people 37%, `v2-real` search ~2%). 8.2 CPU-seconds a game.

## Should the bot race? (2026-10-19)

The question: surviving to site 9 is a real strategy with a healthy fellowship, and moving
twice makes the opponent move twice to keep up, which feeds your Shadow side. The bots
barely race: search wins at site 9 in 2-3% of games, people in 37-48%.

**What people do.** 2020 Fellowship Block (March-May), 51,476 reproduced "move again?" answers: yes
23%. More with a bigger fellowship (17% with one or two companions, 28% with five, 39%
with six), more when behind (44% when the move still leaves them a site back, 17% level),
most at the end (44% from site 8). The search bot says yes 0.5% of the time (25 of 4,625).

**What people's positions are worth.** `HumanReplay --values <csv>` writes the set-2
state at every reproduced card-action decision, labelled with the recorded winner, for
`BotProbe --fit`. March 2020: 119,266 Fellowship Block positions (40,513 Movie Block from
845 games, thinner past site 6). Among people, two sites ahead wins 62%, two behind 41%,
site 8 62%, site 9 76% -- the race the right way round, where `value-real` has it backwards.
A model fit on them (`value-human03`) rates site 2 at 43% and site 8 at 84%, but it also
learned that the winner's companions carry more wounds (+0.17 per s.d. of `myWounds`) and
that dead companions hardly matter: in human games the winner is the one out in front,
fighting. Search scoring with it throws companions away: **34.1% +- 2.7** against plain
`v2-real` on held-out real FotR decks, against 65.7% for `value-real` (site 9 in 5.9% of
games against 3.1%). Human outcomes are correlation, not what an action does.

**Searching the move itself.** `--move-rollouts N [--move-depth D]` gives "move again?" its
own N rollouts, played out to the end by default: no value model judges the race, the bot's
own future play does. Against plain `v2-real`:

| Move search | FotR test (1,200) | Movie/Expanded test (1,600) | moved on |
|---|---|---|---|
| as before (2 rollouts, 1 turn) | 64.0% +- 2.7 | 68.9% +- 2.3 | 0.5% / 1.2% |
| 8 rollouts, 3 turns | 64.0% | 67.8% | 3.6% / 5.8% |
| 8 playouts | 62.3% | 64.5% | 7.1% / 9.0% |

Site-9 wins stay at 1.6-3.1% of games in every row. The human-imitation policy
(`human-trees-2m`), which survives better, the same: 59.9% -> **54.0% +- 2.8** with 8
playouts, moving on 8.5% of the time against 0.8%, site 9 7.8% either way.

So moving on more loses for these bots, measured by their own play: the extra twilight
buys the opponent minions, and the fellowship cannot take them. The race is not where the
bot is wrong; the skirmishes after a move are. The human move pattern above -- move on with
five or six companions, or to catch up -- is the target once the Free Peoples play
survives a double move. Caveat: 8 playouts decide a yes/no with a lot of noise, so some of
the extra moves are noise; 32 would tell better, at 4x the cost of those decisions.
`--move-rollouts` stays opt-in.

## Engine speed (2026-10-18)

A JFR profile of v2 self-play put most of a game's CPU in two places that do no game
work, both fixed (commit "Speed the engine up 2.5x"):

- `AddKeywordModifier.getKeywords()` built a new `HashSet` on every call, and
  `ModifiersLogic` calls it for every keyword modifier on every keyword check: 28% of
  samples. Now built once (and in `CancelKeywordBonusTargetModifier`).
- `TurnProcedure` refreshed `GameStats` after every effect, for the UI: 16%, and the
  frame the hour-long strength evaluations were found in. A game with no
  `GameStateListener` (bot against bot) now refreshes once when it stops for a decision,
  which is when bots read it; live games are unchanged.

`BotTrace` plays a fixed set of seeded games and hashes every question and answer, for
diffing two builds. Before and after: all 200 games identical, **34.1 -> 13.4
CPU-seconds**. The search player (2 rollouts, depth 1) went **18.2 -> 4.1 CPU-seconds a
game** in a quiet machine. Server tests: 6,699 run, only `PatchNotesLibraryTest` fails, as
on upstream master. Worth upstreaming: both are behaviour-preserving and the second also
narrows the hour-long-evaluation hang to decision points.

For that diff to mean anything the bots had to be repeatable too: they drew the burden
bid, exploration and fallbacks from unseeded `Random`s. They now draw from `BotRandom`, a
per-thread Random that `BotGame` seeds from the game seed and restores afterwards (a
rollout is a game inside a game); rollouts reseed it at the branch with the world.

What is left (after the fix, v2 self-play): `CharacterDeathRule.checkCharactersZeroVitality`
~20% (vitality of every character on every loop step), keyword-modifier scans ~30%
(`getKeywordModifiersAffectingCard` walks every modifier of a type for each card), the
bots' own features ~26%. In a search game 93% of CPU is engine work inside rollouts and
most of that is the prefix replay -- hence cloning (above).

Retried 2026-10-18 with the cheap search: 4 rollouts, depth 1, every option, every
card-action decision of 1,060 `starters:train` games (~104,000 labels, 25 minutes with
copies), card-action models refit (min 500 labels): **47.3% +- 3.1** against v2 on
`starters:test`. More and better labels did not help, which points at the student, not the
labels: a logistic per-option model cannot let the state change its choice (see "The
linear-model ceiling"), so it cannot hold what the search knows. Distilling needs a model
that scores state and option together -- trees on `CardFeatures.join`'s crosses lost
before; a small neural net or per-option Q-regression on rollout values is the next try.

## Feature set 2 (`--features 2`)

A model directory's `features.txt` says which set its models were trained on (none =
1); the bot reads it and feeds them only that set, so bots on either set share a
process. `CardFeatures.version()` is thread-local, set by the bot around each decision
and by BotTrain around training.

- **State** (`GeneralStateFeatures`, ~64 numbers, one layout for every decision): the
  board by *side* -- Free Peoples companions, Shadow minions -- with `iAmFp` saying which
  side the bot is on, so the Shadow bot finally sees its own minions; modified strength,
  vitality and resistance (the engine's `getResistance` already subtracts burdens);
  site, threats, initiative, move limit; hand composition; the current skirmish's
  margin from the bot's side; phase one-hot. No deck identity.
- **Card**: set 1's 22, plus resistance, unique, 22 keywords as counts (damage, fierce,
  archer, ambush, toil...), culture and race one-hot.
- **State x option** (`CardFeatures.join`): 11 state features times a 12-number summary
  of the option's card, appended wherever a trainer scores one option per card. This is
  what lets a logistic model's choice depend on the board at all.

Training a new set from models of the old one: the old models play generation 0
(`FotrStarterBot.recordWith`) and the steps record the new set's state; no old model is
copied into the new directory, and a model that cannot be fit becomes a
`ConstantModel` (every option ties).

### Do the features know who is winning? (`BotProbe`)

`BOT_MAIN=com.gempukku.lotro.bots.simulation.BotProbe ./bot-vs-bot.sh --suite starters:test --games 60`
records both sets' state at every card-action decision of the same games, fits
P(decider wins | state) on 80% of the games, and reports held-out AUC -- a one-minute
check of a feature change, where a training run takes half an hour.

- First version of set 2: **0.687 against set 1's 0.750**. It described only the active
  board, and the Shadow player's own fellowship is inactive -- invisible, though it is
  half of whether that player wins. Fixed by a per-player race block (burdens,
  resistance, companions, vitality, wounds, dead).
- After: 0.764 against 0.769 on 3,000 games, and 0.757 against 0.765 with the vitality
  fix; standardising the inputs changes neither.
  The board's information about the result is about saturated near 0.77 with summary
  features, and set 1 already had most of it. What set 2 adds is in the option terms --
  which only pay if the label rewards a good choice. That is the case for working on
  the signal (lookahead) next, with set 2 as the features the student learns from.

The set-2 directories trained before the race block (`v2`, `v2-trees`) were deleted:
the state grew, and `CardFeatures.join` refuses a state of the wrong length. So was
`v2-race` (47.5% +- 3.1 against v1, 72.1% against shipped), for a subtler reason: it
subtracted wounds from `ModifiersQuerying.getVitality`, which already has them taken off
-- a fellowship with 3 vitality left on its weakest member read as -2. Same length, so it
would have loaded and been fed differently scaled inputs. **Dump a few vectors by name
(`PROBE_DUMP=1` with BotProbe) after any feature change**; that is how this was seen.

## Other things found

- **Real decks run into the decision limit.** About 5% of games between real decks ended with a bot conceding at
  the live server's 1,000-decision limit (long games, many card actions each); the evaluations here pass
  `--max-decisions 3000`.

- **The engine can loop forever without asking for a decision.**
  `TurnProcedure.carryOutPendingActionsUntilDecisionNeeded` only stops at a pending
  decision or a winner; about 2 games in 1,000 between starters spin there (seen in
  Regroup and Archery). On the live server that table would hang. `BotGame` ends such
  a game after 60s and reports it as `engine stuck`.
- **One engine call can run for hours.** Seen in a search rollout (Big Baddies / Shadows-era
  boards): `GameStats.updateGameStats` -> `getStrength` -> keyword modifiers whose
  condition is `CanSpotSameCulture` -> `Filters.countSpottable` over the active cards ->
  `isCardType` -> modifiers again. Not a loop, a nesting that blows up with enough
  conditional modifiers in play, and it never returns to where a winner would stop it.
  A live table would hang the same way. `BotVsBot` gives up on a game after 15 minutes,
  `BotTeach` once no game has finished for 15 minutes, and both exit the JVM at the end,
  since nothing else stops such a thread.
- `FotrStartersRLGameStateFeatures` counts minions owned by the *opponent*, so when the
  bot is the Shadow player its own minions are invisible to it. Strengths are printed,
  not modified. "Deck index" is 1 for Aragorn's starter, 2 for Gandalf's, 0 for
  everything else.
- `RandomDecisionBot.chooseIntegerDecision` used `Math.min(range, 1)`, so it always
  answered the minimum -- and overflowed to an exception when max was
  `Integer.MAX_VALUE`. Now it returns the minimum on purpose.
- **Bots throwing is silent on the server**: `BotGameStateListener` falls back to a
  default answer. `BotGame.BOT_ERRORS` counts them and BotVsBot prints them;
  `BOT_TRACE=1` prints the first stack of each (add
  `JAVA_OPTS=-XX:-OmitStackTraceInFastThrow`, or a hot NPE arrives with no stack). It
  found: the set-1 features crashing on cards with no side (`getSide()` is null for
  some) -- the shipped bot answered those decisions at random off the FotR starters;
  the "general" site rule parsing temporary ids (`temp0`) and never running, which is
  part of why the extensions measured as nothing; set 2 reading hands before the deal.
- Set 1 trained the "pass" option of skirmish card actions on a different vector than
  it scored it with (`getCardFeatures(PASS, action)` vs `getPassCardFeatures()`); now
  the same. Live bots no longer keep every decision in memory when nothing collects
  them (the named server bots did, forever).
- `ActionChoiceAction` has no JSON form in `SemanticActionFactory`; those steps are
  dropped from training (no trainer reads them).

## Next steps, roughly in order

0. (2026-10-18) Next wins, in order: more and more varied training decks for the value
   model (it learns decks, not the game, from ~45 starters); distil the search with how much better each
   option scored (rollout values) rather than best/worst labels; the remaining engine
   hotspots (`checkCharactersZeroVitality`, keyword-modifier scans).

1. (Done: replays are exact; the teacher is built -- see Lookahead.)
2. (Done as feature set 2. It ties set 1 -- logistic or trees, in play and in BotProbe.)
3. A bigger, more varied training pool than ~45 starters: the Deck Library, or decks
   generated per format.
4. Better credit than "every decision in a won game was good": e.g. reward per turn
   from burdens/wounds/site reached, or comparing against the model's own estimate.
