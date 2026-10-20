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

The goal is now the bot itself, cheap to run; ranking decks with an inaccurate bot is not worth much.

**Searching only the big decisions** (`BotTeach --no-card-search`: assignments and choices searched, card actions
left to the models) costs 0.3-1 CPU-seconds a game and adds nothing: `human22-ca-as` with it beats plain `v2-real`
57.8% (2 rollouts), 56.9% (8), 56.7% (8, rollouts cut off 3 turns on) -- plain `human22-ca-as` is 58.0%. The full
search's strength is in the card actions.

**A better check of ranking like people** (`scripts/deck_fidelity.py`): every human Movie Block game in `games.csv`
with a deck of 20+ games (4,476 games, 135 decks) replayed by the bot, each deck's bot win rate over the matchups it
actually played against its human win rate. Human win rates split in half agree at r 0.62 (full record ~0.77). Plain
`v2-real` -0.01, `human22-ca-as` 0.10-0.11 (two runs), distilled below 0.13. The 16-deck round robin swung from 0.44
to -0.04 between bots that play equally well: too few decks to tell anything.

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

**Ranking like people.** The 16 most-played Movie Block decks, round robin, 20 games a pair,
both sides searching with `human22-ca-as`: Spearman against human win rates **0.44** (0.21
with `v2-real`; plain bots all ~0: v2-real 0.00, human22-as -0.10, human22-ca-as -0.01 over
4,800 games). Endings: corruption 16.5% (people 18%), site 9 6.1% (people 37%, `v2-real`
search ~2%). 8.2 CPU-seconds a game. Sixteen decks and human win rates that mix deck and
pilot make this a noisy check (+- ~0.25), but it is the direction wanted, and
`rank-decks.sh` now defaults to `human22-ca-as` (`BOT_MODELS=bot-models-exp/v2-real` for the
old one).

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
For deck ranking this means decks built to race to site 9 are probably undervalued.
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
