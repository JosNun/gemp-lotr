"""The dashboard's feeds, small (2026-11-09): every 15 seconds the page fetched every stream whole -- 67 of them, 43 MB.
This writes dashboard/data/feeds/index.json (each stream's name, last record, record count and meta, newest first) and
dashboard/data/feeds/<stream>.json (its records thinned: the last TAIL whole, TOTAL - TAIL spread evenly over the rest),
rewriting a stream's file only when its source changed. The page reads the index, draws the recent streams, and fetches
an older one only when it is opened.

usage: python3 -I scripts/feed_index.py [--every 15]
"""
import json, os, sys, time

FEED = os.path.expanduser('~/dev/gemp-replays/labels/feed')
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'dashboard', 'data', 'feeds')
TOTAL, TAIL = 400, 150


def thin(records):
    if len(records) <= TOTAL:
        return records
    head, tail = records[:-TAIL], records[-TAIL:]
    step = len(head) / (TOTAL - TAIL)
    return [head[int(i * step)] for i in range(TOTAL - TAIL)] + tail


def write(path, obj):
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(obj, f, separators=(',', ':'))
    os.replace(tmp, path)


def once(seen):
    os.makedirs(OUT, exist_ok=True)
    index = []
    for name in sorted(os.listdir(FEED)):
        if not name.endswith('.jsonl'):
            continue
        stream = name[:-6]
        src = os.path.join(FEED, name)
        mp = os.path.join(FEED, stream + '.meta.json')
        stamp = (os.path.getmtime(src), os.path.getmtime(mp) if os.path.exists(mp) else 0)
        if seen.get(stream, (None,))[0] != stamp:
            records = []
            for line in open(src):
                try:
                    records.append(json.loads(line))
                except ValueError:
                    pass  # a line being written
            if not records:
                continue
            meta = json.load(open(mp)) if os.path.exists(mp) else {}
            write(os.path.join(OUT, stream + '.json'), {'name': stream, 'meta': meta, 'total': len(records),
                                                         'records': thin(records)})
            seen[stream] = (stamp, {'name': stream, 'meta': meta, 'total': len(records), 'last': records[-1],
                                    't': records[-1].get('t', 0)})
        index.append(seen[stream][1])
    index.sort(key=lambda e: -e['t'])
    write(os.path.join(OUT, 'index.json'), {'updated': time.time(), 'streams': index})


def main():
    every = int(sys.argv[sys.argv.index('--every') + 1]) if '--every' in sys.argv else 0
    seen = {}
    while True:
        once(seen)
        if not every:
            break
        time.sleep(every)


if __name__ == '__main__':
    main()
