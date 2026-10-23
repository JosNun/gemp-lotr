"""HumanReplay --cf-deaths rows: where the bot differs from the human, its answer minus the human's in Free Peoples
companions and Ring-bearers killed that turn, by side and kind.  python3 scripts/counterfactual_deaths.py cf.tsv"""
import collections, math, sys
g=collections.defaultdict(lambda: [[],[]]); asked=collections.Counter(); agreed=collections.Counter()
for l in open(sys.argv[1]):
    p=l.rstrip('\n').split('\t')
    if len(p)<9: continue
    n=int(p[8]); side=p[9] if len(p)>9 else '?'; kind=p[2][:40]
    for k in [(side,'all'), (side,kind)]:
        asked[k]+=1
        if n<=0: agreed[k]+=1; continue
        g[k][0].append((int(p[7])-int(p[6]))/n); g[k][1].append((int(p[11])-int(p[10]))/n)
print("where the bot differs from the human: its answer minus the human's, Free Peoples companions / Ring-bearers killed this turn")
for k in sorted(asked, key=lambda k:(k[0], k[1]!='all', -asked[k])):
    a,b=g[k]
    if asked[k]<40 or len(a)<10: continue
    ma=sum(a)/len(a); sa=math.sqrt(sum((x-ma)**2 for x in a)/max(1,len(a)-1)/len(a))
    mb=sum(b)/len(b); sb=math.sqrt(sum((x-mb)**2 for x in b)/max(1,len(b)-1)/len(b))
    print(f"  {k[0]:6s} {k[1]:40s} asked {asked[k]:5d} agree {100*agreed[k]/asked[k]:4.0f}% differ {len(a):4d}  companions {ma:+.3f} +- {1.96*sa:.3f}  Ring-bearer {mb:+.3f} +- {1.96*sb:.3f}")
