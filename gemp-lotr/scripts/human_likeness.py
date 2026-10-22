"""How human-like is each bot? Reads HumanReplay --counterfactual files, one per bot, made on the same games at the
same --cf-rate (the sampling is seeded by game, so they ask the same decisions); --cf-rollouts 0 is enough.

    python3 scripts/human_likeness.py name=cf-a.tsv name=cf-b.tsv [--min 50]

Per kind of decision, the share the bot answered as the human did; then over all decisions (dominated by the common
kinds, Fellowship and Shadow card actions) and the mean over kinds (each kind counted once, so a rare decision the bot
gets wrong -- reconcile, a choice of costs -- weighs as much as a common one).
"""
import collections, sys

argv = sys.argv[1:]
MIN = 50
if '--min' in argv:
    i = argv.index('--min'); MIN = int(argv[i + 1]); del argv[i:i + 2]
bots = []
for a in argv:
    name, f = a.split('=', 1)
    asked, agree = collections.Counter(), collections.Counter()
    for line in open(f, encoding='utf-8'):
        p = line.rstrip('\n').split('\t')
        if len(p) < 9:
            continue
        asked[p[2]] += 1
        agree[p[2]] += p[8] == '0'
    bots.append((name, asked, agree))

kinds = sorted({k for _, asked, _ in bots for k in asked}, key=lambda k: -bots[0][1][k])
kinds = [k for k in kinds if min(asked[k] for _, asked, _ in bots) >= MIN]
w = max(12, max(len(n) for n, _, _ in bots) + 2)
print(f"{'decision':52s} {'asked':>6s}" + ''.join(f"{n:>{w}s}" for n, _, _ in bots))
for k in kinds:
    print(f"{k[:52]:52s} {bots[0][1][k]:6d}" + ''.join(f"{100 * agree[k] / asked[k]:{w - 1}.1f}%" for _, asked, agree in bots))
print(f"{'all decisions':52s} {sum(bots[0][1].values()):6d}" + ''.join(
    f"{100 * sum(agree.values()) / max(1, sum(asked.values())):{w - 1}.1f}%" for _, asked, agree in bots))
print(f"{'mean over the kinds above':52s} {len(kinds):6d}" + ''.join(
    f"{100 * sum(agree[k] / asked[k] for k in kinds) / max(1, len(kinds)):{w - 1}.1f}%" for _, asked, agree in bots))
