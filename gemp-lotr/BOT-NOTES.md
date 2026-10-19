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

**Ranking decks: `./rank-decks.sh <format> <deck list> [games per pair] [--quick]`.** Every
pair plays with the search bot on both sides (~1-3 CPU-seconds a game), or the plain v2 bot
with `--quick` (~0.1). Screen many decks with `--quick`, confirm the finalists with search,
and read "Games needed" below before trusting a few points' difference.

Round robin, for ranking decks: `--suite rr:<format>:<deck list file>` plays every pair of
the decks in the file (one deck spec per line) and prints standings -- each deck's wins
over all its games. `decks/king8.txt` is an example. `BotTeach --play --both` does the
same with the search bot on both sides.

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

### Cost against strength (for using the bot to rank decks)

Ranking decks needs ~1,000 games per comparison for +-3 points, so cost per game matters
as much as strength. `BotTeach --play` and `bot-vs-bot.sh` both report CPU-seconds a game.

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
  ~6 points. **Fit the value model on decks other than the ones being ranked**, or it
  carries their old results into the new ranking.
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
against the 29 Shadows-site starters, with the user's four decks: 434 of 464 games ended
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

`rank-decks.sh` now uses both. Games against plain v2 still end 92% by a killed
Ring-bearer; between two such searchers on the user's decks, 14% now end at site 9
(against under 5% before). It re-ranked the user's decks: The Long Night went from 20% to
38% overall (10.7% -> 30.3% against Hunt the Ring-bearer) and passed Rangers (33% -> 27%) --
the earlier search had been misplaying The Long Night's assignments.

User's four decks against the 29 Shadows-site starters (Expanded, 40 games a matchup,
both sides searching), before and after searching assignments and choices:

| Deck | Before | After | Site-9 wins before -> after |
|---|---|---|---|
| Hunt the Ring-bearer | 76.7% | 74.3% +- 2.5 | 13 -> 89 |
| Khazad Rush | 64.1% | 67.4% +- 2.7 | 2 -> 23 |
| The Long Night | 37.6% | 46.7% +- 2.9 | 24 -> 75 |
| Rangers of the North Watch | 42.8% | 46.2% +- 2.9 | 27 -> 70 |

5.7 CPU-seconds a game, 4,640 games.

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

### Does the cheap bot rank decks like the strong one? (No.)

`decks/king8.txt`, round robin. Plain v2 at 100 games a pair (2,800 games, 66s) and twice
at 20; search (2 rollouts, depth 1, both sides) at 20 a pair (558 games, ~12 CPU-hours):

| Deck | v2, 100/pair | search, 20/pair |
|---|---|---|
| MD - Frodo | 73.9% | 65.2% |
| SoG - Merry | 63.3% | **46.4%** |
| Big Baddies - Gothmog's Conquering | 59.3% | 54.0% |
| RotK - Eomer | 53.7% | 52.5% |
| RotK - Aragorn | 41.9% | 48.6% |
| SoG - Pippin | 39.0% | 34.3% |
| MD - Sam | 37.7% | 50.7% |
| SHELBI - Gandalf | 31.3% | **48.6%** |

Rank correlation (Spearman) of the two: **0.48**, against 0.83-0.93 between v2's own runs
at the same sample sizes. The largest moves (Gandalf, Merry, ~17 points) are beyond both
intervals. Search compresses the spread and revalues particular decks: which deck looks
best depends on how well the bot plays it, so the cheap bot is fine for screening and
not for the final word. (Search against itself: 51.6%, as it should be.)

Rerun with the cheaper search (`rank-decks.sh`: copies, logistic value model, 2 rollouts,
depth 1, all options; 3.0 CPU-seconds a game with both sides searching, King decks), 40
games a pair, twice:

| Deck | v2, 100/pair | old search, 20/pair | new search, 2 x 40/pair pooled |
|---|---|---|---|
| MD - Frodo | 73.9% | 65.2% | 57.0% +- 4.1 |
| SoG - Merry | 63.3% | 46.4% | 56.1% +- 4.1 |
| MD - Sam | 37.7% | 50.7% | 53.9% +- 4.1 |
| Big Baddies - Gothmog's Conquering | 59.3% | 54.0% | 49.1% +- 4.1 |
| SHELBI - Gandalf | 31.3% | 48.6% | 48.6% +- 4.1 |
| RotK - Eomer | 53.7% | 52.5% | 47.9% +- 4.1 |
| RotK - Aragorn | 41.9% | 48.6% | 47.5% +- 4.1 |
| SoG - Pippin | 39.0% | 34.3% | 40.0% +- 4.1 |

The two new-search runs agree with each other at only Spearman 0.57: **under good play
these decks are close** (six of eight within 47-57%), and 280 games a deck (+-5.8) cannot
order the middle. So the old "0.48" was partly noise, not only bot quality. What survives
pooling: v2 spreads the decks twice as wide as search does, and misranks MD Sam (38% ->
54%) and SHELBI Gandalf (31% -> 49%) by more than both intervals -- decks the plain bot
plays badly. Frodo and Merry on top and Pippin last hold in every search run.

**Games needed.** A deck's win rate over n games has a 95% interval of about +-100/sqrt(n)
points. Separating decks 4 points apart wants +-2 each: ~2,500 games a deck. In a round
robin each game counts for two decks, so 8 decks at 2,500 = 10,000 games, ~8 CPU-hours
(about 1.5 hours on 6 cores) with the search bot. An 8-rollout reference run was stopped:
at ~16 CPU-seconds a game it would have taken an hour for +-6.8 a deck, too coarse here.

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

**Bot vs human deck ranking:** the 16 most-played Movie Block decks ranked by plain (guarded)
v2 against each other correlate with their human win rates at Spearman **-0.37**. Human win
rates mix deck and pilot (strong players favour some decks) and opponents differ, so this is
not a clean target -- but it is the only ground truth there is, and it says the plain bot's
ranking of real decks is not to be trusted.

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
in search (69.7% vs 67.4% +- 3.3). Defaults are now `v2-real` and `value-real`
(`rank-decks.sh`; `BOT_MODELS` / `BOT_VALUE` override). Card ids in the summaries carry
cosmetic suffixes (`*` foil, `T` tengwar); they are stripped from the deck files.

**Realism, guarded `v2-real` on the 16 most-played Movie Block decks** (4,800 games):
median 7 turns (4 before), corruption 10% of games, kills 89%, site 9 under 1%. Human decided
Movie Block games: kills 44%, site 9 37%, corruption 18%; 34% end at site 9 and ~70% at site 6
or later, where bot games end around site 5. Ranking against human win rates: Spearman
**0.04** (-0.37 before). The gap left is the Free Peoples side: fellowships are worn down
and never reach site 9.

**A lever for that, not built:** the full replays (`GEMP-LOTR-Replays_<month>.zip`, ~12,000
games and 3 GB a month, to 2022-11) are each player's event stream: every decision put to
them (`type="D"`, with its parameters), every card placement, message and stats update. The
answers can be read off what follows. Imitating human decisions needs a state tracker that
rebuilds the board from those events for the feature extractors -- days of work, but it is
the only source of good Free Peoples play there is.

Also tried on held-out real decks (`v2-real` searching, against plain `v2-real`, 1,584
games each, +- 2.3): value model with trees fit on real decks 65.1% vs logistic 69.1% --
logistic still wins; searching single-card picks from a list (`--card-picks`) 69.3% vs 69.1%
at +30% CPU -- no gain, opt-in.

### The user's decks, with the real-deck bot (2026-10-19)

Search on both sides with `v2-real`, `value-real`, the guard, assignments and choices.

Against the 30 most-played real Expanded decks (Sep 2024 - Feb 2025), 20 games a matchup:

| Deck | Won | Beats | Toughest |
|---|---|---|---|
| Hunt the Ring-bearer | 64.0% +- 3.8 | 20 of 30 | Exp Two Men/Canequeiros 20%, Gandalf wounding - WK Nazgul 30% |
| Khazad Rush | 56.3% +- 4.0 | 18 of 30 | Denetor uruk hunters 10%, Expanded Gondor Saruman 15% |
| Rangers of the North Watch | 46.8% +- 4.0 | 14 of 30 | Exp Two Men/Canequeiros 15%, Gondor/Gandalf Tokeny 15% |
| The Long Night | 34.3% +- 3.8 | 6 of 30 | EXP-archery 5%, Denetor uruk hunters 5% |

Against each other, 300 a pair: Hunt 72.4%, Khazad 68.8%, Rangers 39.2%, The Long Night
19.6% (it was 37.6% with `v2` and the same search: of the four, its result depends most on the
policy). About 5% of the games against real decks ended with a bot conceding at the live
server's 1,000-decision limit; `rank-decks.sh` now passes `--max-decisions 3000`.

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

0. (2026-10-18) Search is now cheap enough that games, not CPU, limit deck ranking. Next
   wins, in order: more and more varied training decks for the value model (it learns
   decks, not the game, from ~45 starters); distil the search with how much better each
   option scored (rollout values) rather than best/worst labels; the remaining engine
   hotspots (`checkCharactersZeroVitality`, keyword-modifier scans).

1. (Done: replays are exact; the teacher is built -- see Lookahead.)
2. (Done as feature set 2. It ties set 1 -- logistic or trees, in play and in BotProbe.)
3. A bigger, more varied training pool than ~45 starters: the Deck Library, or decks
   generated per format.
4. Better credit than "every decision in a won game was good": e.g. reward per turn
   from burdens/wounds/site reached, or comparing against the model's own estimate.
