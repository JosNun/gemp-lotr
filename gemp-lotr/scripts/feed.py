"""Dashboard feeds: any job appends records to a stream, and the dashboard draws them, working out how from their shape.

    from feed import emit; emit('months', month='2021-05', games=2836, done=1, total=22)     # Python
    python3 -I scripts/feed.py months month=2021-05 games=2836 done=1 total=22                # shell
    Feed.emit("bvb", Map.of("done", 12, "total", 6336))                                        # Java (simulation.Feed)
    python3 -I scripts/feed.py months --meta title="Older months" x=month                    # how to draw it

A stream is ~/dev/gemp-replays/labels/feed/<stream>.jsonl, one record a line, each stamped with "t" (Unix seconds).
The page (dashboard/app.js, "Live" panels) reads them straight from the Mac every 15 seconds. What it draws:
  done + total               a progress bar, its rate and the time left
  wins + games (+ a, b)      a matchup: the win rate with its 95% interval, and how it moved as games came in
  month (YYYY-MM)            a calendar of the months, coloured by the first number
  numbers that change        a small line chart each, against x (meta), epoch, step or done, else time
  words with few values      counts of each (how games ended, ...), coloured by route where they are routes
  note / text                the last few lines, as a log
<stream>.meta.json (--meta) can name a title, x, the fields to chart ("y": [...]) or hide ("hide": [...]).

The page is served on the LAN: emit numbers and names of bots, decks or months -- never people's names.
"""
import json, os, sys, time

DIR = os.path.expanduser('~/dev/gemp-replays/labels/feed')


def _path(stream, suffix):
    if not stream or not all(c.isalnum() or c in '-_.' for c in stream):
        raise ValueError(f'stream names are letters, digits, - _ .: {stream!r}')
    os.makedirs(DIR, exist_ok=True)
    return os.path.join(DIR, stream + suffix)


def emit(stream, **fields):
    """Appends one record to {stream}; returns it."""
    rec = {'t': round(time.time(), 3), **fields}
    with open(_path(stream, '.jsonl'), 'a') as f:  # one write of one line: appends from several jobs do not interleave
        f.write(json.dumps(rec, separators=(',', ':')) + '\n')
    return rec


def meta(stream, **fields):
    """Sets how {stream} is drawn (merged into what was set before)."""
    p = _path(stream, '.meta.json')
    m = json.load(open(p)) if os.path.exists(p) else {}
    m.update(fields)
    with open(p + '.tmp', 'w') as f:
        json.dump(m, f)
    os.replace(p + '.tmp', p)


def _value(v):
    if v[:1] in '[{"':
        try:
            return json.loads(v)
        except ValueError:
            return v
    for cast in (int, float):
        try:
            return cast(v)
        except ValueError:
            pass
    return {'true': True, 'false': False}.get(v, v)


def main(args):
    if not args:
        sys.exit(__doc__)
    stream, rest = args[0], args[1:]
    is_meta = '--meta' in rest
    pairs = dict(a.split('=', 1) for a in rest if '=' in a)
    fields = {k: _value(v) for k, v in pairs.items()}
    if is_meta:
        meta(stream, **fields)
    else:
        emit(stream, **fields)


if __name__ == '__main__':
    main(sys.argv[1:])
