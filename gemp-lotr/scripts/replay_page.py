#!/usr/bin/env python3
"""Build one self-contained HTML page: a results chart and a replay viewer.

    python3 scripts/replay_page.py --results results.json \
        --replays replays/ --out report.html

--results  the chart data (bots, formats, cells, cost, notes). Nothing on the
           page is hardcoded; "sample": true puts a banner on it.
--replays  one or more directories. Sub-directories named
           <format>-<A>-vs-<B> (A, B in upstream/cheap/strong) hold
           replay-*.json, one recorded game each. JSON files lying directly in
           a given directory are taken too, with the bots guessed from
           meta.botA / meta.botB.

The page needs no server and no other file. Each game is gzipped and base64'd
into the page and inflated in the browser (DecompressionStream) when opened.
The only network use is card images from GEMP's public image host; every card
is also drawn as text, so the page reads the same without them.

Standard library only.
"""

import argparse
import base64
import gzip
import json
import os
import re
import sys

BOT_KEYS = ("upstream", "cheap", "strong")
DIR_RE = re.compile(r"^(?P<fmt>[A-Za-z0-9_]+)-(?P<a>%s)-vs-(?P<b>%s)$"
                    % ("|".join(BOT_KEYS), "|".join(BOT_KEYS)))
BP_RE = re.compile(r"^(\d+)_(\d+)")

# The web client's own tables of cards whose image does not follow the
# LOTR<set><number> rule (cards/Card.js getFixedImage): looked up in this
# order, first hit wins. set40.js is left out: it points at plain http, which a
# page served over https may not load.
IMAGE_TABLES = ("cards/CardImages.js", "hobbit.js", "PC_Cards.js")
TABLE_LINE = re.compile(r"""^\s*['"](\d+_\d+)['"]\s*:\s*['"](https://[^'"]+)['"]""")
ERRATA_RE = re.compile(r"static\s+remadeErratas\s*=\s*\{(.*?)\}", re.S)
ERRATA_SET = re.compile(r"""['"](\d+)['"]\s*:\s*\[([^\]]*)\]""")


def natural(s):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", s)]


def load_image_rules(web_js, wanted):
    """Overrides and errata for the blueprints in `wanted`, read from the client."""
    fixed, errata = {}, []
    if not web_js or not os.path.isdir(web_js):
        print("note: web client js not found at %s; using the main-set image "
              "rule only" % web_js, file=sys.stderr)
        return fixed, errata
    for rel in IMAGE_TABLES:
        path = os.path.join(web_js, rel)
        if not os.path.exists(path):
            continue
        table = {}
        with open(path, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                m = TABLE_LINE.match(line)   # commented-out lines do not match
                if m:
                    table[m.group(1)] = m.group(2)
        for bp, url in table.items():
            if bp in wanted and bp not in fixed:
                fixed[bp] = url
    card_js = os.path.join(web_js, "cards/Card.js")
    if os.path.exists(card_js):
        with open(card_js, encoding="utf-8", errors="replace") as fh:
            m = ERRATA_RE.search(fh.read())
        if m:
            for s, nums in ERRATA_SET.findall(m.group(1)):
                for n in re.findall(r"\d+", nums):
                    bp = "%s_%s" % (s, n)
                    if bp in wanted and bp not in fixed:
                        errata.append(bp)
    return fixed, errata


def guess_bot(raw):
    raw = (raw or "").lower()
    if raw.startswith("search"):
        return "strong"
    if raw.startswith("general"):
        return "cheap"
    return "upstream"


def trivial(step):
    """A prompt with nothing to choose: 'Play X action or Pass' with no action."""
    return not step.get("opts") and step.get("a") in ("Pass", "(none)", "", None)


def collect(replay_dirs):
    """-> list of (group fmt key or None, a, b, path), in a stable order."""
    found = []
    for root in replay_dirs:
        root = os.path.expanduser(root)
        if not os.path.isdir(root):
            print("warning: %s is not a directory" % root, file=sys.stderr)
            continue
        for name in sorted(os.listdir(root), key=natural):
            path = os.path.join(root, name)
            m = DIR_RE.match(name)
            if os.path.isdir(path) and m:
                for f in sorted(os.listdir(path), key=natural):
                    if f.endswith(".json"):
                        found.append((m.group("fmt"), m.group("a"), m.group("b"),
                                      os.path.join(path, f), name))
            elif os.path.isfile(path) and name.endswith(".json"):
                found.append((None, None, None, path, os.path.basename(root.rstrip("/"))))
    return found


def build(args):
    with open(os.path.expanduser(args.results), encoding="utf-8") as fh:
        results = json.load(fh)

    index, blobs, wanted = [], [], set()
    for fmt, a, b, path, group in collect(args.replays):
        try:
            with open(path, encoding="utf-8") as fh:
                game = json.load(fh)
            meta, steps = game["meta"], game["steps"]
            game["cards"]
        except (ValueError, KeyError, OSError) as e:
            # A job may be writing this file right now; a half-written game is
            # skipped, not fatal.
            print("warning: skipping %s (%s)" % (path, e), file=sys.stderr)
            continue
        if not steps:
            print("warning: skipping %s (no steps)" % path, file=sys.stderr)
            continue
        for bp in game["cards"]:
            m = BP_RE.match(bp)
            if m:
                wanted.add(m.group(0))
        stem = os.path.splitext(os.path.basename(path))[0]
        index.append({
            "id": "%s/%s" % (group, stem.replace("replay-", "")),
            "fmt": fmt,
            "fmtName": meta.get("format"),
            "a": a or guess_bot(meta.get("botA")),
            "b": b or guess_bot(meta.get("botB")),
            "pA": meta.get("playerA"), "pB": meta.get("playerB"),
            "deckA": meta.get("deckA"), "deckB": meta.get("deckB"),
            "winner": meta.get("winner"), "first": meta.get("first"),
            "reason": meta.get("reason"), "turns": meta.get("turns"),
            "steps": len(steps),
            "real": sum(1 for s in steps if not trivial(s)),
            "changed": sum(1 for s in steps if "own" in s),
            "searched": sum(1 for s in steps if s.get("searched")),
        })
        raw = json.dumps(game, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        blobs.append(base64.b64encode(gzip.compress(raw, 9, mtime=0)).decode("ascii"))

    web_js = args.web_js or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..",
        "gemp-lotr-async", "src", "main", "web", "js", "gemp-022")
    fixed, errata = load_image_rules(web_js, wanted)

    def js(obj):
        # "</" would end the <script> block it is embedded in.
        return json.dumps(obj, separators=(",", ":"), ensure_ascii=False).replace("</", "<\\/")

    games_html = "\n".join(
        '<script type="application/octet-stream" id="g%d">%s</script>' % (i, b)
        for i, b in enumerate(blobs))
    html = (PAGE
            .replace("/*__RESULTS__*/null", js(results))
            .replace("/*__INDEX__*/null", js(index))
            .replace("/*__IMG__*/null", js({"fixed": fixed, "errata": sorted(errata, key=natural)}))
            .replace("<!--__GAMES__-->", games_html))
    out = os.path.expanduser(args.out)
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(html)
    size = os.path.getsize(out)
    print("%s: %d games, %d bytes (%.2f MB)%s" % (
        out, len(index), size, size / 1e6,
        "  ** over 8 MB **" if size > 8e6 else ""))
    if results.get("sample"):
        print("note: results are marked \"sample\": the page carries a sample-data banner")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--results", required=True, help="results json (chart data)")
    ap.add_argument("--replays", required=True, nargs="+",
                    help="directory of <format>-<A>-vs-<B>/replay-*.json (one or more)")
    ap.add_argument("--out", required=True, help="the .html file to write")
    ap.add_argument("--web-js", default=None,
                    help="gemp-022 web client js directory, for the card image tables "
                         "(default: found relative to this script)")
    build(ap.parse_args())


PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>GEMP bot: results and replays</title>
<style>
:root{
  --bg:#f7f7f5; --surface:#ffffff; --sunken:#f0f0ed; --ink:#1b1b19; --ink2:#52524c; --muted:#7b7b73;
  --line:#deded8; --line2:#c9c9c2; --sel:#eceff3;
  --s-cheap:#2a78d6; --s-strong:#eb6834; --s-upstream:#8d8d86;
  --wound:#c23b2a; --warn-bg:#fff4d6; --warn-line:#e0b84a; --warn-ink:#5b4300;
  --shadow-bg:#2c2c2a; --shadow-ink:#f3f3f0;
}
@media (prefers-color-scheme: dark){
  :root{
    --bg:#161615; --surface:#1f1f1e; --sunken:#262624; --ink:#ececE8; --ink2:#b4b4ac; --muted:#8a8a82;
    --line:#343431; --line2:#48483f; --sel:#2a2f37;
    --s-cheap:#3987e5; --s-strong:#d95926; --s-upstream:#7d7d76;
    --wound:#ef6a58; --warn-bg:#3a3010; --warn-line:#7d6520; --warn-ink:#f2dfa0;
    --shadow-bg:#e6e6e1; --shadow-ink:#1b1b19;
  }
}
*{box-sizing:border-box}
html{-webkit-text-size-adjust:100%}
body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 system-ui,-apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif}
main{max-width:1180px;margin:0 auto;padding:24px 16px 64px}
h1{font-size:26px;line-height:1.2;margin:0 0 10px}
h2{font-size:19px;margin:40px 0 10px;padding-top:14px;border-top:1px solid var(--line)}
h3{font-size:15px;margin:0 0 8px}
p{margin:0 0 10px;max-width:74ch}
.muted{color:var(--muted)} .small{font-size:13px}
a{color:inherit}
button,select,input{font:inherit;color:inherit}
.banner{background:var(--warn-bg);border:1px solid var(--warn-line);color:var(--warn-ink);padding:10px 14px;border-radius:6px;margin:0 0 18px;font-weight:600}
.sw{display:inline-block;width:10px;height:10px;border-radius:2px;vertical-align:baseline;margin-right:6px;flex:none}
.sw.upstream{background:var(--s-upstream)} .sw.cheap{background:var(--s-cheap)} .sw.strong{background:var(--s-strong)}
dl.bots{margin:12px 0 0;display:grid;grid-template-columns:max-content 1fr;gap:6px 16px;max-width:86ch}
dl.bots dt{font-weight:600;white-space:nowrap} dl.bots dd{margin:0;color:var(--ink2)}
@media (max-width:560px){dl.bots{grid-template-columns:1fr;gap:0} dl.bots dd{margin-bottom:8px}}
.panel{background:var(--surface);border:1px solid var(--line);border-radius:8px;padding:14px 16px}
.grid2{display:grid;grid-template-columns:minmax(0,1.15fr) minmax(0,1fr);gap:16px;margin-top:16px}
@media (max-width:820px){.grid2{grid-template-columns:minmax(0,1fr)}}
.legend{display:flex;flex-wrap:wrap;gap:4px 18px;font-size:13px;color:var(--ink2);margin:2px 0 6px}
.legend span{display:inline-flex;align-items:center}
.legend .ref{display:inline-block;width:22px;border-top:2px dashed var(--ink2);margin-right:6px}
.chart{position:relative}
.chart svg{display:block;width:100%;height:auto;overflow:visible}
.chart text{fill:var(--ink2);font-size:12px}
.chart text.val{fill:var(--ink);font-weight:600}
.chart text.grp{fill:var(--ink)}
.chart .gridl{stroke:var(--line);stroke-width:1}
.chart .base{stroke:var(--line2);stroke-width:1}
.chart .refl{stroke:var(--ink2);stroke-width:1.5;stroke-dasharray:5 4}
.chart .whisk{stroke:var(--ink);stroke-width:1.5}
.chart .bar.cheap{fill:var(--s-cheap)} .chart .bar.strong{fill:var(--s-strong)} .chart .bar.upstream{fill:var(--s-upstream)}
.chart .hit{fill:transparent}
.chart .hit:hover{fill:var(--ink);fill-opacity:.05}
.tip{position:absolute;pointer-events:none;background:var(--ink);color:var(--bg);font-size:12.5px;line-height:1.35;padding:6px 9px;border-radius:5px;max-width:240px;z-index:5;display:none}
table{border-collapse:collapse;width:100%;font-size:13.5px}
th,td{text-align:left;padding:5px 10px 5px 0;border-bottom:1px solid var(--line);vertical-align:top}
th{font-weight:600;color:var(--ink2)} td.num,th.num{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}
details{margin-top:10px} summary{cursor:pointer;color:var(--ink2);font-size:13.5px}
.tblwrap{overflow-x:auto}
ul.notes{margin:14px 0 0;padding-left:18px;color:var(--ink2);font-size:13.5px;max-width:86ch}

/* index */
.fmt{margin-top:12px}
.tabs{display:flex;flex-wrap:wrap;gap:6px;margin-top:12px}
.tabs button{background:var(--surface);border:1px solid var(--line2);border-radius:16px;padding:4px 12px;cursor:pointer;font-size:13.5px}
.tabs button.on{background:var(--ink);color:var(--bg);border-color:var(--ink)} .tabs button.on .muted{color:inherit;opacity:.7}
[hidden]{display:none!important}
.mu{margin:0 0 12px}
.mu .muh{font-size:13.5px;color:var(--ink2);margin:0 0 4px;display:flex;flex-wrap:wrap;align-items:center;gap:2px 6px}
.games{display:grid;grid-template-columns:repeat(auto-fill,minmax(250px,1fr));gap:6px}
.game{display:block;width:100%;text-align:left;background:var(--surface);border:1px solid var(--line);border-radius:6px;padding:7px 10px;cursor:pointer;font-size:13px;line-height:1.4}
.game:hover{border-color:var(--line2)} .game[aria-current=true]{border-color:var(--ink);box-shadow:inset 0 0 0 1px var(--ink)}
.game b{font-weight:600} .game span.l1,.game span.l2{display:block} .game .l2{color:var(--ink2)} .game i{font-style:normal}

/* viewer */
#viewer{margin-top:18px;scroll-margin-top:8px}
#board{scroll-margin-top:90px}
.vhead{display:flex;flex-wrap:wrap;gap:4px 14px;align-items:baseline;margin-bottom:8px}
.vhead h3{margin:0}
.ctl{display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin-bottom:8px}
.ctl button{background:var(--surface);border:1px solid var(--line2);border-radius:5px;padding:5px 10px;cursor:pointer;min-height:34px}
.ctl button:hover:not(:disabled){border-color:var(--ink)} .ctl button:disabled{opacity:.4;cursor:default}
.ctl button.on{background:var(--ink);color:var(--bg);border-color:var(--ink)}
.ctl select{background:var(--surface);border:1px solid var(--line2);border-radius:5px;padding:5px 6px;min-height:34px;max-width:100%}
.ctl label{display:inline-flex;align-items:center;gap:5px;font-size:13.5px;color:var(--ink2);white-space:nowrap}
.ctl .grow{flex:1 1 220px;display:flex;align-items:center;gap:8px;min-width:0}
.ctl input[type=range]{flex:1;min-width:0;accent-color:var(--ink2)}
.ctl .pos{font-size:13px;color:var(--ink2);font-variant-numeric:tabular-nums;white-space:nowrap}
.vgrid{display:grid;grid-template-columns:minmax(0,1fr) 350px;gap:14px;align-items:start}
.vside{display:flex;flex-direction:column;gap:14px;min-width:0;position:sticky;top:90px;max-height:calc(100vh - 98px)}
.vside .logwrap{min-height:0;display:flex;flex-direction:column;flex:1 1 auto}
.vside .decision{flex:none;max-height:60vh;overflow-y:auto}
.ctlbar{position:sticky;top:0;z-index:4;background:var(--bg);padding:6px 0 2px;margin-bottom:6px}
@media (min-width:901px){.vside .log{max-height:none}}
@media (max-width:900px){.vgrid{display:flex;flex-direction:column;align-items:stretch} .vside{display:contents} .ctlbar{position:static}
  #decision{order:1;max-height:none;min-height:150px} #board{order:2} .vside .logwrap{order:3}}
.status{display:flex;flex-wrap:wrap;gap:4px 16px;align-items:center;font-size:14px;margin-bottom:8px}
.status b{font-weight:600}
.tw{display:inline-flex;align-items:center;justify-content:center;min-width:24px;height:24px;padding:0 6px;border-radius:12px;background:var(--shadow-bg);color:var(--shadow-ink);font-weight:700;font-size:13px}
.path{display:grid;grid-template-columns:repeat(9,minmax(0,1fr));gap:3px;margin:8px 0}
.site{border:1px solid var(--line);border-radius:4px;padding:3px 4px;font-size:11px;line-height:1.25;min-height:44px;background:var(--surface);overflow:hidden;cursor:default}
.site.none{background:transparent;border-style:dashed;color:var(--muted)}
.site .n{font-weight:700;font-size:12px;display:flex;align-items:center;gap:3px}
.site .n .sw{margin:0;width:9px;height:9px;border-radius:50%}
.site .t{display:block;overflow:hidden;text-overflow:ellipsis;display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;color:var(--ink2);overflow-wrap:anywhere}
@media (max-width:560px){.site .t{display:none} .site{min-height:0;text-align:center} .site .n{justify-content:center}}
.side{background:var(--surface);border:1px solid var(--line);border-radius:8px;padding:10px 12px}
.side.deciding{border-color:var(--ink2)}
.sideh{display:flex;flex-wrap:wrap;align-items:center;gap:4px 10px;margin-bottom:2px}
.sideh .who{font-weight:700}
.role{font-size:11.5px;font-weight:700;letter-spacing:.03em;text-transform:uppercase;padding:1px 7px;border-radius:10px;border:1px solid var(--line2);color:var(--ink2);white-space:nowrap}
.role.shadow{background:var(--shadow-bg);color:var(--shadow-ink);border-color:var(--shadow-bg)}
.facts{font-size:13px;color:var(--ink2);display:flex;flex-wrap:wrap;gap:2px 14px;margin-bottom:6px}
.facts b{color:var(--ink);font-weight:600}
.row{display:grid;grid-template-columns:78px minmax(0,1fr);gap:8px;padding:6px 0;border-top:1px solid var(--line)}
.row>.lab{font-size:11.5px;text-transform:uppercase;letter-spacing:.04em;color:var(--muted);padding-top:2px;line-height:1.3}
.row>.cards{display:flex;flex-wrap:wrap;gap:6px;min-width:0}
.row .empty{color:var(--muted);font-size:13px}
@media (max-width:560px){.row{grid-template-columns:minmax(0,1fr);gap:3px}}
.card{width:88px;border:1px solid var(--line2);border-radius:5px;background:var(--sunken);font-size:11.5px;line-height:1.25;overflow:hidden;cursor:zoom-in;position:relative;flex:none}
.card.fight{outline:2px solid var(--ink);outline-offset:1px}
.card.idle{opacity:.62}
.card .art{height:58px;background:var(--line);position:relative;overflow:hidden;display:flex;align-items:center;justify-content:center}
.card .art .ph{font-size:9.5px;letter-spacing:.05em;color:var(--muted);text-align:center;padding:0 3px;text-transform:uppercase}
.card .art img{position:absolute;inset:0;width:100%;height:100%;object-fit:cover;object-position:50% 22%;opacity:0;transition:opacity .15s}
.card .art img.ok{opacity:1}
.card.shadowc .art{background:var(--line2)}
.card .nm{padding:3px 5px 2px;font-weight:600;overflow-wrap:break-word;hyphens:auto}
.card .st{padding:0 5px 3px;display:flex;flex-wrap:wrap;gap:1px 6px;color:var(--ink2);font-variant-numeric:tabular-nums}
.card .st b{color:var(--ink)}
.pips{color:var(--wound);letter-spacing:1px;white-space:nowrap}
.card .tag{display:block;margin:0 4px 3px;padding:1px 4px;border-radius:3px;font-size:10.5px;background:var(--surface);border:1px solid var(--line);color:var(--ink2);overflow-wrap:anywhere}
.card .tag.att{cursor:zoom-in}
.card .tag.vs{border-color:var(--ink2);color:var(--ink)}
.card .tag.rb{border-style:dashed}
.card.sm{width:76px} .card.sm .art{height:44px} .card.sm .nm{font-weight:500;font-size:11px}
.chip{display:inline-block;padding:1px 6px;border:1px solid var(--line);border-radius:10px;font-size:12px;margin:0 3px 2px 0;cursor:zoom-in;background:var(--sunken)}
.keyline{font-size:12px;color:var(--muted);margin:8px 0 0}
.decision{min-height:190px}
.dwho{display:flex;flex-wrap:wrap;align-items:center;gap:4px 8px;font-size:13.5px;color:var(--ink2);margin-bottom:4px}
.dwho b{color:var(--ink)}
.stag{font-size:11px;border:1px solid var(--line2);border-radius:10px;padding:0 6px;color:var(--muted)}
.q{font-weight:600;margin:0 0 8px;overflow-wrap:anywhere}
ul.opts{list-style:none;margin:0 0 8px;padding:0;display:flex;flex-direction:column;gap:3px}
ul.opts li{padding:4px 8px;border:1px solid var(--line);border-radius:5px;font-size:13.5px;color:var(--ink2);overflow-wrap:anywhere}
ul.opts li.chosen{border-color:var(--ink);background:var(--sel);color:var(--ink);font-weight:600}
ul.opts li.chosen::before{content:"\2713\00a0"}
ul.opts li.own{border-style:dashed;border-color:var(--ink2)}
ul.opts li.own::after{content:" \2014  the bot alone";font-weight:400;color:var(--muted);font-size:12px}
.ans{font-size:13.5px;margin:0 0 6px;overflow-wrap:anywhere}
.ans b{background:var(--sel);border:1px solid var(--ink);border-radius:5px;padding:1px 7px;font-weight:600}
.changed{border:1px solid var(--s-strong);border-left-width:4px;border-radius:5px;padding:6px 9px;font-size:13.5px;margin-top:8px;overflow-wrap:anywhere}
.changed b{font-weight:700}
.logwrap h3{display:flex;justify-content:space-between;align-items:baseline}
.log{max-height:360px;min-height:120px;flex:1 1 auto;overflow-y:auto;border-top:1px solid var(--line);margin:0 -16px -14px;position:relative}
.log .th{position:sticky;top:0;z-index:1;background:var(--sunken);font-size:11.5px;text-transform:uppercase;letter-spacing:.04em;color:var(--ink2);padding:3px 16px;border-bottom:1px solid var(--line)}
.log button{display:flex;gap:6px;align-items:baseline;width:100%;text-align:left;background:none;border:0;border-bottom:1px solid var(--line);padding:4px 16px;cursor:pointer;font-size:12.5px;line-height:1.35}
.log button:hover{background:var(--sunken)}
.log button.cur{background:var(--sel);box-shadow:inset 3px 0 0 var(--ink)}
.log button .sw{width:8px;height:8px;border-radius:50%;margin:0;transform:translateY(-1px)}
.log button .tx{min-width:0;flex:1;overflow-wrap:anywhere}
.log button .tx i{font-style:normal;color:var(--muted)}
.log button .mk{font-size:10.5px;color:var(--ink);border:1px solid var(--s-strong);border-radius:8px;padding:0 5px;white-space:nowrap}
#zoom{position:fixed;z-index:50;display:none;background:var(--surface);border:1px solid var(--line2);border-radius:8px;box-shadow:0 8px 30px rgba(0,0,0,.28);padding:8px;width:270px;max-width:calc(100vw - 24px);pointer-events:none}
#zoom.tap{pointer-events:auto;left:50%!important;top:50%!important;transform:translate(-50%,-50%);width:320px}
#zoom img{display:block;opacity:1;max-width:100%;max-height:62vh;margin:0 auto 6px;border-radius:6px}
#zoom .zn{font-weight:700;font-size:14px} #zoom .zm{font-size:12px;color:var(--ink2)}
.err{color:var(--wound);font-weight:600}
</style>
</head>
<body>
<main>
<div id="banner"></div>
<h1 id="title">GEMP bot: results and replays</h1>
<div id="intro"></div>

<h2>Results</h2>
<div class="panel">
  <h3 id="c1title"></h3>
  <div class="legend" id="c1legend"></div>
  <div class="chart" id="chart1"></div>
  <p class="small muted" id="c1foot" style="margin:6px 0 0"></p>
</div>
<div class="grid2">
  <div class="panel" id="c2panel">
    <h3 id="c2title"></h3>
    <div class="legend" id="c2legend"></div>
    <div class="chart" id="chart2"></div>
  </div>
  <div class="panel" id="costpanel">
    <h3>Cost: CPU time per decision</h3>
    <div class="tblwrap" id="cost"></div>
  </div>
</div>
<details><summary>All numbers as a table (games per cell)</summary><div class="tblwrap" id="cells"></div></details>
<ul class="notes" id="notes"></ul>

<h2>Replays</h2>
<p id="rintro"></p>
<div id="index"></div>

<div id="viewer" hidden>
  <div class="vhead"><h3 id="vtitle"></h3><span class="small muted" id="vsub"></span></div>
  <div class="ctlbar">
  <div class="ctl">
    <button id="bPrevTurn" title="Previous turn (Shift+Left)">&laquo; Turn</button>
    <button id="bPrev" title="Previous decision (Left)">&lsaquo; Prev</button>
    <button id="bPlay" title="Autoplay (Space)">Play</button>
    <button id="bNext" title="Next decision (Right)">Next &rsaquo;</button>
    <button id="bNextTurn" title="Next turn (Shift+Right)">Turn &raquo;</button>
    <label>Speed <select id="speed"><option value="0.5">0.5&times;</option><option value="1" selected>1&times;</option><option value="2">2&times;</option><option value="4">4&times;</option></select></label>
    <label>Jump to <select id="jump"></select></label>
  </div>
  <div class="ctl">
    <div class="grow"><input type="range" id="scrub" min="0" max="0" value="0" aria-label="Position in the game"><span class="pos" id="pos"></span></div>
    <label><input type="checkbox" id="fSkip" checked> Skip passes</label>
    <label id="fChangedL"><input type="checkbox" id="fChanged"> Only decisions search changed <span id="nChanged"></span></label>
  </div>
  </div>
  <div class="vgrid">
    <div class="vboard" id="board"></div>
    <div class="vside">
      <div class="panel decision" id="decision"></div>
      <div class="panel logwrap"><h3>Decisions <span class="small muted" id="logn" style="font-weight:400"></span></h3><div class="log" id="log"></div></div>
    </div>
  </div>
</div>
</main>
<div id="zoom"></div>

<script id="d-results" type="application/json">/*__RESULTS__*/null</script>
<script id="d-index" type="application/json">/*__INDEX__*/null</script>
<script id="d-img" type="application/json">/*__IMG__*/null</script>
<!--__GAMES__-->
<script>
(function(){
"use strict";
var R=JSON.parse(document.getElementById('d-results').textContent)||{};
var INDEX=JSON.parse(document.getElementById('d-index').textContent)||[];
var IMG=JSON.parse(document.getElementById('d-img').textContent)||{fixed:{},errata:[]};
var $=function(id){return document.getElementById(id)};
function esc(s){return String(s==null?'':s).replace(/[&<>"']/g,function(c){return{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]})}
var BOTS=R.bots||{};
function botLabel(k){return (BOTS[k]&&BOTS[k].label)||k}
function sw(k){return '<span class="sw '+esc(k)+'"></span>'}
var FORMATS=R.formats||[];
function fmtLabel(k,fallback){for(var i=0;i<FORMATS.length;i++)if(FORMATS[i].key===k)return FORMATS[i].label||k;return fallback||k}
function pct(x){return (Math.round(x*10)/10).toString().replace(/\.0$/,'')+'%'}

/* ---------- intro ---------- */
if(R.sample)$('banner').innerHTML='<div class="banner">Sample data &mdash; the numbers in the charts and tables on this page are placeholders for layout, not results.</div>';
(function(){
  var h='<p>Three bots for <a href="https://github.com/PlayersCouncil/gemp-lotr">GEMP</a>, the online <i>Lord of the Rings</i> trading card game server, played each other on starter decks. '+
    'This page shows how often the two new bots beat the bot that is in the repository today, what each costs in CPU time, and lets you step through recorded games one decision at a time.</p>';
  h+='<dl class="bots">';
  ['upstream','cheap','strong'].concat(Object.keys(BOTS).filter(function(k){return['upstream','cheap','strong'].indexOf(k)<0})).forEach(function(k){
    if(!BOTS[k])return; h+='<dt>'+sw(k)+esc(botLabel(k))+'</dt><dd>'+esc(BOTS[k].what||'')+'</dd>'});
  h+='</dl>';
  if(R.generated)h+='<p class="small muted" style="margin-top:10px">Generated '+esc(R.generated)+'.</p>';
  $('intro').innerHTML=h;
})();

/* ---------- charts ---------- */
var CELLS=R.cells||[];
function cell(f,a,b){for(var i=0;i<CELLS.length;i++){var c=CELLS[i];if(c.format===f&&c.a===a&&c.b===b)return c}return null}
function barChart(host,series,opp,refText){
  var groups=FORMATS.map(function(f){return{f:f,bars:series.map(function(s){return cell(f.key,s,opp)})}})
    .filter(function(g){return g.bars.some(Boolean)});
  if(!groups.length){host.innerHTML='<p class="muted small">No results for this comparison.</p>';return false}
  var tip=document.createElement('div');tip.className='tip';
  function draw(){
    var W=Math.max(260,Math.round(host.clientWidth)),wide=W>=640;
    var mL=38,mR=wide?124:6,mT=20,plotH=220;
    var plotW=W-mL-mR,band=plotW/groups.length;
    var bw=Math.max(8,Math.min(46,(band*0.72-2*(series.length-1))/series.length));
    var maxCh=Math.max(4,Math.floor((band-8)/6.8));
    var labs=groups.map(function(g){
      var words=(g.f.label||g.f.key).split(/\s+/),lines=[],cur='';
      words.forEach(function(w){if(cur&&(cur+' '+w).length>maxCh){lines.push(cur);cur=w}else cur=cur?cur+' '+w:w});
      if(cur)lines.push(cur); if(g.f.note)lines[lines.length-1]+=' *'; return lines});
    var nl=Math.max.apply(null,labs.map(function(l){return l.length}));
    var H=mT+plotH+10+nl*15+4;
    function y(v){return mT+plotH*(1-v/100)}
    var s='<svg viewBox="0 0 '+W+' '+H+'" width="'+W+'" height="'+H+'" role="img" aria-label="'+esc(host.getAttribute('data-label')||'')+'">';
    [0,25,50,75,100].forEach(function(t){
      if(t!==50)s+='<line class="'+(t===0?'base':'gridl')+'" x1="'+mL+'" x2="'+(mL+plotW)+'" y1="'+y(t)+'" y2="'+y(t)+'"/>';
      s+='<text x="'+(mL-6)+'" y="'+(y(t)+4)+'" text-anchor="end">'+t+'%</text>'});
    var hits='';
    groups.forEach(function(g,gi){
      var n=series.length,tot=n*bw+(n-1)*2,x0=mL+band*gi+(band-tot)/2;
      g.bars.forEach(function(c,si){
        if(!c)return; var x=x0+si*(bw+2),top=y(c.rate),r=Math.min(4,bw/2,Math.max(0,y(0)-top));
        s+='<path class="bar '+esc(c.a)+'" d="M'+x+','+y(0)+'V'+(top+r)+'q0,'+(-r)+' '+r+','+(-r)+'h'+(bw-2*r)+'q'+r+',0 '+r+','+r+'V'+y(0)+'z"/>';
        var lo=Math.max(0,c.rate-(c.ci||0)),hi=Math.min(100,c.rate+(c.ci||0)),cx=x+bw/2;
        if(c.ci)s+='<path class="whisk" fill="none" d="M'+cx+','+y(lo)+'V'+y(hi)+'M'+(cx-4)+','+y(lo)+'h8M'+(cx-4)+','+y(hi)+'h8"/>';
        s+='<text class="val" x="'+cx+'" y="'+(y(hi)-5)+'" text-anchor="middle"'+(bw<26?' style="font-size:11px"':'')+'>'+(bw<26?pct(c.rate).slice(0,-1):pct(c.rate))+'</text>';
        var t=(g.f.label||g.f.key)+': '+botLabel(c.a)+' against '+botLabel(c.b)+' — '+pct(c.rate)+(c.ci?' ± '+c.ci:'')+
          (c.games!=null?' ('+(c.wins!=null?c.wins+' of ':'')+c.games+' games)':'');
        hits+='<rect class="hit" x="'+(x-1)+'" y="'+mT+'" width="'+(bw+2)+'" height="'+plotH+'" data-t="'+esc(t)+'"><title>'+esc(t)+'</title></rect>'});
      labs[gi].forEach(function(l,li){s+='<text class="grp" x="'+(mL+band*(gi+.5))+'" y="'+(mT+plotH+18+li*15)+'" text-anchor="middle">'+esc(l)+'</text>'})});
    s+='<line class="refl" x1="'+mL+'" x2="'+(mL+plotW)+'" y1="'+y(50)+'" y2="'+y(50)+'"/>';
    if(wide){var rt=refText.split(' '),half=Math.ceil(rt.length/2);
      s+='<text x="'+(mL+plotW+8)+'" y="'+(y(50)-2)+'">'+esc(rt.slice(0,half).join(' '))+'</text><text x="'+(mL+plotW+8)+'" y="'+(y(50)+13)+'">'+esc(rt.slice(half).join(' '))+'</text>'}
    s+=hits+'</svg>';
    host.innerHTML=s;host.appendChild(tip);
  }
  host.addEventListener('pointermove',function(e){
    var t=e.target&&e.target.getAttribute&&e.target.getAttribute('data-t');
    if(!t){tip.style.display='none';return}
    var r=host.getBoundingClientRect();tip.textContent=t;tip.style.display='block';
    var x=e.clientX-r.left+12;if(x+tip.offsetWidth>r.width)x=Math.max(0,e.clientX-r.left-12-tip.offsetWidth);
    tip.style.left=x+'px';tip.style.top=Math.max(0,e.clientY-r.top-tip.offsetHeight-10)+'px'});
  host.addEventListener('pointerleave',function(){tip.style.display='none'});
  draw();
  var lastW=host.clientWidth;
  if(window.ResizeObserver)new ResizeObserver(function(){if(Math.abs(host.clientWidth-lastW)>1){lastW=host.clientWidth;draw()}}).observe(host);
  else window.addEventListener('resize',draw);
  return true;
}
(function(){
  var up=botLabel('upstream');
  $('c1title').textContent='Win rate against the '+up.toLowerCase().replace(/^upstream/,'upstream');
  $('c1legend').innerHTML='<span>'+sw('cheap')+esc(botLabel('cheap'))+'</span><span>'+sw('strong')+esc(botLabel('strong'))+
    '</span><span><i class="ref"></i>50%: level with upstream</span><span class="muted">whiskers: 95% interval</span>';
  $('chart1').setAttribute('data-label','Win rate of the new bots against the upstream bot, by format');
  barChart($('chart1'),['cheap','strong'],'upstream','level with upstream');
  var foot=FORMATS.filter(function(f){return f.note}).map(function(f){return '* '+esc(f.label||f.key)+': '+esc(f.note)+'.'});
  $('c1foot').innerHTML=foot.join(' ')+(foot.length?' ':'')+'Share of games won by the named bot; hover or tap a bar for the game count.';

  $('c2title').textContent='Search against no search: '+botLabel('strong')+' against '+botLabel('cheap');
  $('c2legend').innerHTML='<span>'+sw('strong')+esc(botLabel('strong'))+'</span><span><i class="ref"></i>50%: level with '+esc(botLabel('cheap').toLowerCase())+'</span>';
  $('chart2').setAttribute('data-label','Win rate of the new bot with search against the new bot without, by format');
  if(!barChart($('chart2'),['strong'],'cheap','level, search adds nothing'))$('c2legend').innerHTML='';

  var cost=R.cost||{},base=cost.upstream&&cost.upstream.ms_per_decision;
  function ms(v){return v>=100?Math.round(v)+'':v>=10?v.toFixed(1).replace(/\.0$/,''):v>=1?(+v.toFixed(2))+'':(+v.toPrecision(2))+''}
  var keys=['upstream','cheap','strong'].filter(function(k){return cost[k]&&cost[k].ms_per_decision!=null});
  if(!keys.length)$('costpanel').hidden=true;
  else{
    var h='<table><tr><th>Bot</th><th class="num">ms per decision</th><th class="num">'+(base?'&times; upstream':'')+'</th></tr>';
    keys.forEach(function(k){var v=cost[k].ms_per_decision;
      h+='<tr><td>'+sw(k)+esc(botLabel(k))+'</td><td class="num">'+ms(v)+'</td><td class="num">'+(base?(k==='upstream'?'1&times;':(v/base>=10?Math.round(v/base):(+(v/base).toFixed(1)))+'&times;'):'')+'</td></tr>'});
    h+='</table><p class="small muted" style="margin:8px 0 0">Mean CPU time for one decision, in milliseconds. Shown as a table because the range is too wide for a linear bar and it is a different quantity from win rate.</p>';
    $('cost').innerHTML=h;
  }
  var t='<table><tr><th>Format</th><th>Bot</th><th>Against</th><th class="num">Games</th><th class="num">Won</th><th class="num">Win rate</th><th class="num">95% interval</th></tr>';
  CELLS.forEach(function(c){t+='<tr><td>'+esc(fmtLabel(c.format))+'</td><td>'+sw(c.a)+esc(botLabel(c.a))+'</td><td>'+esc(botLabel(c.b))+'</td><td class="num">'+esc(c.games)+'</td><td class="num">'+esc(c.wins)+'</td><td class="num">'+pct(c.rate)+'</td><td class="num">'+(c.ci!=null?'&plusmn; '+esc(c.ci):'')+'</td></tr>'});
  $('cells').innerHTML=t+'</table>';
  $('notes').innerHTML=(R.notes||[]).map(function(n){return '<li>'+esc(n)+'</li>'}).join('');
})();

/* ---------- card images ----------
   Same rule as the web client (gemp-022/cards/Card.js getImageUrl): a fixed
   table first, then https://i.lotrtcgpc.net/decipher/LOTR<ss><nnn>.jpg, with
   the masterworks numbering for the tails of sets 12, 13, 15, 17 and 18. */
var badUrl={};
function pad(n,w){n=''+n;while(n.length<w)n='0'+n;return n}
function imgUrls(bp){
  var m=/^(\d+)_(\d+)/.exec(bp||'');if(!m)return[];
  var key=m[0],s=+m[1],n=+m[2];
  if(IMG.fixed[key])return[IMG.fixed[key]];
  var mw=(s===12||s===13)?n>194:s===15?(n>194&&n<204):s===17?n>148:s===18?n>140:false;
  var off=s===17?148:s===18?140:194;
  var main='https://i.lotrtcgpc.net/decipher/LOTR'+pad(s,2)+(mw?'O0'+(n-off):pad(n,3))+'.jpg';
  if(IMG.errata.indexOf(key)>=0)return['https://play.lotrtcgpc.net/gemp-lotr/images/erratas/LOTR'+pad(s,2)+pad(n,3)+'.jpg',main];
  return[main];
}
function imgTag(bp,eager){
  var u=imgUrls(bp).filter(function(x){return!badUrl[x]});
  if(!u.length)return'';
  return '<img alt=""'+(eager?'':' loading="lazy" width="88" height="58"')+' decoding="async" src="'+esc(u[0])+'"'+(u[1]?' data-alt="'+esc(u[1])+'"':'')+'>';
}
/* load/error do not bubble: listen in the capture phase, once, for every card image on the page */
document.addEventListener('load',function(e){var t=e.target;if(t&&t.tagName==='IMG')t.classList.add('ok')},true);
document.addEventListener('error',function(e){var t=e.target;if(!t||t.tagName!=='IMG')return;
  badUrl[t.getAttribute('src')]=1;var alt=t.getAttribute('data-alt');
  if(alt&&!badUrl[alt]){t.removeAttribute('data-alt');t.src=alt}else t.remove()},true);

/* ---------- index ---------- */
function gameBots(g){var a=botLabel(g.a),b=botLabel(g.b);if(g.a===g.b){a+=' (A)';b+=' (B)'}return{A:a,B:b}}
function shortDeck(d){return String(d||'?').replace(/\s+Starter$/,'')}
(function(){
  if(!INDEX.length){$('rintro').textContent='No recorded games were included in this build.';return}
  $('rintro').innerHTML='A few recorded games per match-up, to watch how each bot plays. They are examples, not the sample behind the charts above. Pick a game; the watcher sees both hands.';
  var order=FORMATS.map(function(f){return f.key}),groups={},keys=[];
  INDEX.forEach(function(g,i){var fk=g.fmt||('~'+(g.fmtName||'Other'));if(!groups[fk]){groups[fk]={label:g.fmt?fmtLabel(g.fmt,g.fmtName):(g.fmtName||'Other')+' (other recordings)',mus:{},muk:[]};keys.push(fk)}
    var mk=g.a+'|'+g.b;if(!groups[fk].mus[mk]){groups[fk].mus[mk]=[];groups[fk].muk.push(mk)}groups[fk].mus[mk].push(i)});
  keys.sort(function(x,y){var a=order.indexOf(x),b=order.indexOf(y);return(a<0?99:a)-(b<0?99:b)});
  var h='<div class="tabs" role="tablist">'+keys.map(function(fk,n){var G=groups[fk],c=0;G.muk.forEach(function(mk){c+=G.mus[mk].length});
    return '<button role="tab" data-tab="'+n+'">'+esc(G.label)+' <span class="muted">'+c+'</span></button>'}).join('')+'</div>';
  keys.forEach(function(fk,n){var G=groups[fk];G.muk.forEach(function(mk){G.mus[mk].forEach(function(i){INDEX[i].tab=n})});h+='<div class="fmt" data-tab="'+n+'">';
    G.muk.forEach(function(mk){var ids=G.mus[mk],g0=INDEX[ids[0]],nb=gameBots(g0);
      h+='<div class="mu"><p class="muh">'+sw(g0.a)+'<b>'+esc(nb.A)+'</b> against '+sw(g0.b)+'<b>'+esc(nb.B)+'</b></p><div class="games">';
      ids.forEach(function(i,k){var g=INDEX[i],w=g.winner==='A'?g.a:g.winner==='B'?g.b:null;
        h+='<button class="game" data-g="'+i+'"><span class="l1"><b>Game '+(k+1)+'</b> &middot; '+esc(g.turns)+' turns'+(g.changed?' &middot; search changed '+g.changed:'')+'</span>'+
          '<span class="l2">'+esc(shortDeck(g.deckA))+' <i class="muted">v</i> '+esc(shortDeck(g.deckB))+'</span>'+
          '<span class="l2">'+(w?sw(w)+esc(gameBots(g)[g.winner])+' won':'No winner')+' <span class="muted">&middot; '+esc(g.reason||'')+'</span></span></button>'});
      h+='</div></div>'});
    h+='</div>'});
  $('index').innerHTML=h;
  $('index').addEventListener('click',function(e){var b=e.target.closest('.game');if(b){openGame(+b.getAttribute('data-g'),null,true);return}
    var t=e.target.closest('[data-tab]');if(t&&t.tagName==='BUTTON')showTab(+t.getAttribute('data-tab'))});
  showTab(0);
})();
function showTab(n){
  [].forEach.call(document.querySelectorAll('#index [data-tab]'),function(el){var on=+el.getAttribute('data-tab')===n;
    if(el.tagName==='BUTTON'){el.classList.toggle('on',on);el.setAttribute('aria-selected',on)}else el.hidden=!on});
}

/* ---------- viewer ---------- */
var cache={},cur=null; /* cur: {i, g(index row), d(game), vis[], pos, timer} */
function inflate(i){
  if(cache[i])return Promise.resolve(cache[i]);
  if(typeof DecompressionStream==='undefined')return Promise.reject(new Error('This browser has no DecompressionStream; use a current Chrome, Firefox, Safari or Edge.'));
  var bin=atob($('g'+i).textContent.trim()),u=new Uint8Array(bin.length);
  for(var k=0;k<bin.length;k++)u[k]=bin.charCodeAt(k);
  var st=new Blob([u]).stream().pipeThrough(new DecompressionStream('gzip'));
  return new Response(st).text().then(function(t){return cache[i]=JSON.parse(t)});
}
function trivial(s){return(!s.opts||!s.opts.length)&&(s.a==='Pass'||s.a==='(none)'||!s.a)}
function phaseName(p){p=String(p||'');if(p==='PLAY_STARTING_FELLOWSHIP')return'Starting fellowship';p=p.replace(/_/g,' ').toLowerCase();return p.charAt(0).toUpperCase()+p.slice(1)}
function botOf(p){var g=cur.g;return p===g.pA?g.a:p===g.pB?g.b:null}
function nameOf(p){var g=cur.g,nb=gameBots(g);return p===g.pA?nb.A:p===g.pB?nb.B:p}
function words(t){ /* the game's own player names ("~1") mean nothing to a reader */
  var g=cur.g,nb=gameBots(g);t=String(t==null?'':t);
  if(g.pA)t=t.split(g.pA).join(nb.A);if(g.pB)t=t.split(g.pB).join(nb.B);return t}
function cardName(bp){var c=cur.d.cards[bp];return c?c.name:bp}
function shortName(bp){return cardName(bp).split(',')[0]}

function openGame(i,step,scroll){
  stop();
  var g=INDEX[i];if(!g)return;
  if(g.tab!=null)showTab(g.tab);
  $('viewer').hidden=false;
  $('vtitle').textContent='Loading…';$('vsub').textContent='';
  [].forEach.call(document.querySelectorAll('.game'),function(b){b.setAttribute('aria-current',b.getAttribute('data-g')==i?'true':'false')});
  if(scroll)$('viewer').scrollIntoView({block:'start'});
  inflate(i).then(function(d){
    cur={i:i,g:g,d:d,vis:[],pos:0,timer:null};
    var nb=gameBots(g),w=g.winner==='A'?nb.A:g.winner==='B'?nb.B:null;
    $('vtitle').innerHTML=sw(g.a)+esc(nb.A)+' <span class="muted" style="font-weight:400">('+esc(g.deckA)+')</span> against '+sw(g.b)+esc(nb.B)+' <span class="muted" style="font-weight:400">('+esc(g.deckB)+')</span>';
    $('vsub').textContent=fmtLabel(g.fmt,g.fmtName)+' · '+(g.first?(g.first==='A'?nb.A:nb.B)+' went first · ':'')+(w?w+' won':'no winner')+(g.reason?' ('+g.reason+')':'')+' · '+g.turns+' turns';
    $('fChanged').checked=false;$('fChanged').disabled=!g.changed;$('fChangedL').style.opacity=g.changed?1:.5;
    $('nChanged').textContent=g.changed?'('+g.changed+')':g.searched?'(none in this game)':'(no search bot in this game)';
    buildJump();refilter(step==null?0:step);
  }).catch(function(e){$('vtitle').innerHTML='<span class="err">Could not open this game: '+esc(e.message||e)+'</span>'});
}
function buildJump(){
  var S=cur.d.steps,h='',lt=null,lp=null,open=false;
  S.forEach(function(s,k){
    if(s.turn!==lt){if(open)h+='</optgroup>';h+='<optgroup label="Turn '+esc(s.turn)+' — '+esc(nameOf(s.fp))+' moves">';open=true;lt=s.turn;lp=null}
    if(s.phase!==lp){h+='<option value="'+k+'">T'+esc(s.turn)+' · '+esc(phaseName(s.phase))+'</option>';lp=s.phase}});
  if(open)h+='</optgroup>';$('jump').innerHTML=h;
}
function refilter(keepStep){
  var S=cur.d.steps,skip=$('fSkip').checked,ch=$('fChanged').checked;
  cur.vis=[];
  for(var k=0;k<S.length;k++){var s=S[k];if(ch){if(!('own' in s))continue}else if(skip&&trivial(s))continue;cur.vis.push(k)}
  $('scrub').max=Math.max(0,cur.vis.length-1);
  renderLog();
  goStep(keepStep==null?0:keepStep);
}
function goStep(k){ /* nearest shown decision at or after step k */
  var v=cur.vis,p=v.length-1;
  for(var i=0;i<v.length;i++)if(v[i]>=k){p=i;break}
  go(p);
}
function go(p){
  var v=cur.vis;
  if(!v.length){cur.pos=0;$('pos').textContent='0 / 0';$('decision').innerHTML='<p class="muted">No decisions match these filters.</p>';$('board').innerHTML='';return}
  cur.pos=Math.max(0,Math.min(v.length-1,p));
  render();
}
function turnStart(dir){
  var v=cur.vis,S=cur.d.steps,t=S[v[cur.pos]].turn,i=cur.pos;
  if(dir>0){while(i<v.length&&S[v[i]].turn===t)i++;return i<v.length?i:v.length-1}
  while(i>0&&S[v[i-1]].turn===t)i--;           /* start of this turn */
  if(i<cur.pos||i===0)return i;
  var t2=S[v[i-1]].turn;i--;while(i>0&&S[v[i-1]].turn===t2)i--;return i;
}
function stop(){if(cur&&cur.timer){clearTimeout(cur.timer);cur.timer=null}$('bPlay').textContent='Play';$('bPlay').classList.remove('on')}
function tick(){
  if(!cur||cur.pos>=cur.vis.length-1){stop();return}
  go(cur.pos+1);
  cur.timer=setTimeout(tick,1500/(+$('speed').value||1));
}
function togglePlay(){
  if(!cur)return;
  if(cur.timer){stop();return}
  if(cur.pos>=cur.vis.length-1)go(0);
  $('bPlay').textContent='Pause';$('bPlay').classList.add('on');
  cur.timer=setTimeout(tick,1500/(+$('speed').value||1));
}

function tile(c,ctx){
  var S=ctx.s,info=cur.d.cards[c.bp]||{},cls='card';
  if(ctx.small)cls+=' sm';
  if(info.side==='SHADOW')cls+=' shadowc';
  if(c.f)cls+=' fight';
  if(ctx.idle)cls+=' idle';
  var h='<div class="'+cls+'" data-bp="'+esc(c.bp)+'"><div class="art"><span class="ph">'+esc((info.type||'').replace(/_/g,' '))+'</span>'+imgTag(c.bp)+'</div>';
  h+='<div class="nm">'+esc(info.name||c.bp)+'</div>';
  if(c.str!=null||c.vit!=null||c.w){
    h+='<div class="st">';
    if(c.str!=null)h+='<span title="Strength">S <b>'+esc(c.str)+'</b></span>';
    if(c.vit!=null)h+='<span title="Vitality left">V <b>'+esc(c.vit)+'</b></span>';
    if(c.w)h+='<span class="pips" title="'+esc(c.w)+' wound'+(c.w>1?'s':'')+'">'+(c.w<=4?'●●●●'.slice(0,c.w):'●×'+c.w)+'</span>';
    h+='</div>'}
  if(ctx.rb&&ctx.rb===c.id)h+='<span class="tag rb">Ring-bearer</span>';
  if(c.f)h+='<span class="tag vs">in the skirmish</span>';
  if(c.vs!=null&&ctx.byId[c.vs])h+='<span class="tag vs">&rarr; on '+esc(shortName(ctx.byId[c.vs].bp))+'</span>';
  (ctx.foes[c.id]||[]).forEach(function(m){h+='<span class="tag vs">&larr; '+esc(shortName(m.bp))+'</span>'});
  (ctx.kids[c.id]||[]).forEach(function(k){h+='<span class="tag att" data-bp="'+esc(k.bp)+'">'+(k.z==='STACKED'?'under: ':'+ ')+esc(cardName(k.bp))+'</span>'});
  return h+'</div>';
}
function row(label,html,emptyText){return '<div class="row"><div class="lab">'+label+'</div><div class="cards">'+(html||'<span class="empty">'+(emptyText||'none')+'</span>')+'</div></div>'}
function render(){
  var S=cur.d.steps,k=cur.vis[cur.pos],s=S[k],g=cur.g;
  /* controls */
  $('scrub').value=cur.pos;
  $('pos').textContent=(cur.pos+1)+' / '+cur.vis.length;
  $('bPrev').disabled=$('bPrevTurn').disabled=cur.pos<=0;
  $('bNext').disabled=$('bNextTurn').disabled=cur.pos>=cur.vis.length-1;
  var opts=$('jump').options,sel=0;for(var i=0;i<opts.length;i++){if(+opts[i].value<=k)sel=i;else break}$('jump').selectedIndex=sel;
  try{history.replaceState(null,'','#game='+encodeURIComponent(g.id)+'&step='+k)}catch(e){}

  /* board */
  var byId={},kids={},foes={},cards=s.cards||[];
  cards.forEach(function(c){byId[c.id]=c});
  function rootOf(c){var n=0;while(c&&c.on!=null&&byId[c.on]&&n++<8)c=byId[c.on];return c}
  cards.forEach(function(c){
    if(c.on!=null&&byId[c.on]){var r=rootOf(c);(kids[r.id]=kids[r.id]||[]).push(c)}
    if(c.vs!=null&&byId[c.vs])(foes[c.vs]=foes[c.vs]||[]).push(c)});
  var path=cards.filter(function(c){return c.z==='ADVENTURE_PATH'}),siteAt={};
  path.forEach(function(c){if(c.n!=null)siteAt[c.n]=c});
  var names=Object.keys(s.players||{});
  names.sort(function(a,b){return(a===g.pA?0:a===g.pB?1:2)-(b===g.pA?0:b===g.pB?1:2)});
  var h='<div class="status"><span><b>Turn '+esc(s.turn)+'</b> · '+esc(phaseName(s.phase))+'</span><span>Twilight pool <span class="tw">'+esc(s.tw)+'</span></span><span class="muted small">The board as it stood when the question was asked.</span></div>';
  function side(p,top){
    var P=s.players[p]||{},isFp=p===s.fp,bk=botOf(p);
    var ctx={s:s,byId:byId,kids:kids,foes:foes,rb:P.rb,idle:false,small:false};
    var mine=cards.filter(function(c){return c.o===p&&!(c.on!=null&&byId[c.on])});
    var comp=mine.filter(function(c){return c.z==='FREE_CHARACTERS'});
    var min=mine.filter(function(c){return c.z==='SHADOW_CHARACTERS'});
    var sup=mine.filter(function(c){return c.z==='SUPPORT'});
    var other=mine.filter(function(c){return['FREE_CHARACTERS','SHADOW_CHARACTERS','SUPPORT','ADVENTURE_PATH'].indexOf(c.z)<0});
    var site=siteAt[P.site];
    var o='<div class="side'+(p===s.p?' deciding':'')+'"><div class="sideh">'+sw(bk||'')+'<span class="who">'+esc(nameOf(p))+'</span>'+
      '<span class="role'+(isFp?'':' shadow')+'">'+(isFp?'Free Peoples · their turn':'Shadow')+'</span>'+(p===s.p?'<span class="stag">deciding</span>':'')+
      '<span class="small muted">'+esc(p===g.pA?g.deckA:p===g.pB?g.deckB:'')+'</span></div>';
    o+='<div class="facts"><span>At site <b>'+esc(P.site)+'</b>'+(site?' <span class="chip" data-bp="'+esc(site.bp)+'">'+esc(cardName(site.bp))+'</span>':'')+'</span><span>Burdens <b>'+esc(P.burdens)+'</b></span><span>Deck <b>'+esc(P.deck)+'</b></span><span>Discard <b>'+esc(P.discard)+'</b></span></div>';
    var hand=(P.hand||[]).map(function(bp){return tile({bp:bp},{s:s,byId:byId,kids:{},foes:{},small:true})}).join('');
    var rows=[];
    rows.push(row('Hand <span>('+(P.hand||[]).length+')</span>',hand,'empty'));
    if(sup.length)rows.push(row('Support area',sup.map(function(c){return tile(c,ctx)}).join('')));
    ctx.idle=!isFp;
    rows.push(row(isFp?'Fellowship':'Fellowship<br>(idle this turn)',comp.map(function(c){return tile(c,ctx)}).join('')));
    ctx.idle=false;
    if(!isFp)rows.push(row('Minions',min.map(function(c){return tile(c,ctx)}).join(''),'none in play'));
    else if(min.length)rows.push(row('Minions',min.map(function(c){return tile(c,ctx)}).join('')));
    if(other.length)rows.push(row('Elsewhere',other.map(function(c){return tile(c,ctx)}).join('')));
    if((P.dead||[]).length)rows.push(row('Dead pile',P.dead.map(function(bp){return '<span class="chip" data-bp="'+esc(bp)+'">'+esc(cardName(bp))+'</span>'}).join('')));
    if(!top)rows.reverse();   /* the two fellowships and the minions between them face each other */
    return o+rows.join('')+'</div>';
  }
  if(names[0])h+=side(names[0],true);
  h+='<div class="path">';
  for(var n=1;n<=9;n++){var c=siteAt[n],here=names.filter(function(p){return s.players[p].site===n});
    h+='<div class="site'+(c?'':' none')+'"'+(c?' data-bp="'+esc(c.bp)+'"':'')+'><span class="n">'+n+here.map(function(p){return '<span class="sw '+esc(botOf(p)||'')+'" title="'+esc(nameOf(p))+' is here"></span>'}).join('')+'</span>'+(c?'<span class="t">'+esc(cardName(c.bp))+'</span>':'')+'</div>'}
  h+='</div>';
  if(names[1])h+=side(names[1],false);
  h+='<p class="keyline">S strength · V vitality left · <span class="pips">●</span> wounds · &rarr; a minion assigned to that character · dots on the site path show where each bot’s fellowship is. Hover or tap a card to see it.</p>';
  $('board').innerHTML=h;

  /* decision */
  var bk=botOf(s.p),a=words(s.a),own=('own' in s)?words(s.own):null;
  var d='<div class="dwho">'+sw(bk||'')+'<b>'+esc(nameOf(s.p))+'</b> decides, as '+(s.p===s.fp?'Free Peoples':'Shadow')+(s.searched?'<span class="stag" title="This answer came from the search">searched</span>':'')+'</div>';
  d+='<p class="q">'+esc(words(s.q))+'</p>';
  var matched=false;
  if(s.opts&&s.opts.length){
    var left=a,ownDone=false;d+='<ul class="opts">';
    s.opts.forEach(function(o){o=words(o);var cls='';
      var at=left===o?0:left.indexOf(o);
      if(o&&at>=0&&(left===o||a!==left||s.kind==='CARD_SELECTION'||s.kind==='ARBITRARY_CARDS')){cls='chosen';matched=true;left=left.slice(0,at)+left.slice(at+o.length)}
      else if(own!=null&&!ownDone&&o===own){cls='own';ownDone=true}
      d+='<li'+(cls?' class="'+cls+'"':'')+'>'+esc(o)+'</li>'});
    d+='</ul>';
  }
  if(!matched||s.kind==='ASSIGN_MINIONS')d+='<p class="ans">Answer: <b>'+esc(a||'(nothing)')+'</b></p>';
  if(own!=null)d+='<div class="changed"><b>Search changed this.</b> The bot alone would have answered “'+esc(own)+'”.</div>';
  $('decision').innerHTML=d;

  /* log */
  var L=$('log'),old=L.querySelector('.cur');if(old)old.classList.remove('cur');
  var b=L.querySelector('button[data-p="'+cur.pos+'"]');
  if(b){b.classList.add('cur');
    /* not scrollIntoView: it would scroll the page as well as the list */
    var top=b.offsetTop,bot=top+b.offsetHeight;
    if(top<L.scrollTop+26||bot>L.scrollTop+L.clientHeight)L.scrollTop=Math.max(0,top-L.clientHeight/2)}
}
function renderLog(){
  var S=cur.d.steps,h='',lt=null,lp=null;
  cur.vis.forEach(function(k,p){var s=S[k];
    if(s.turn!==lt||s.phase!==lp){h+='<div class="th">Turn '+esc(s.turn)+' · '+esc(phaseName(s.phase))+'</div>';lt=s.turn;lp=s.phase}
    h+='<button data-p="'+p+'">'+sw(botOf(s.p)||'')+'<span class="tx">'+esc(words(s.a))+' <i>— '+esc(words(s.q))+'</i></span>'+(('own' in s)?'<span class="mk">changed</span>':'')+'</button>'});
  $('log').innerHTML=h||'<p class="muted small" style="padding:8px 16px">Nothing to show.</p>';
  $('logn').textContent=cur.vis.length+' shown of '+S.length;
}
$('log').addEventListener('click',function(e){var b=e.target.closest('button');if(b){stop();go(+b.getAttribute('data-p'))}});
$('bPrev').onclick=function(){stop();go(cur.pos-1)};
$('bNext').onclick=function(){stop();go(cur.pos+1)};
$('bPrevTurn').onclick=function(){stop();go(turnStart(-1))};
$('bNextTurn').onclick=function(){stop();go(turnStart(1))};
$('bPlay').onclick=togglePlay;
$('scrub').oninput=function(){stop();go(+this.value)};
$('jump').onchange=function(){stop();goStep(+this.value)};
$('fSkip').onchange=$('fChanged').onchange=function(){stop();refilter(cur.vis.length?cur.vis[cur.pos]:0)};
document.addEventListener('keydown',function(e){
  if(!cur||e.altKey||e.ctrlKey||e.metaKey)return;
  var t=e.target.tagName;if(t==='INPUT'||t==='SELECT'||t==='TEXTAREA')return;
  if(e.key==='ArrowRight'){stop();go(e.shiftKey?turnStart(1):cur.pos+1);e.preventDefault()}
  else if(e.key==='ArrowLeft'){stop();go(e.shiftKey?turnStart(-1):cur.pos-1);e.preventDefault()}
  else if(e.key===' '&&t!=='BUTTON'){togglePlay();e.preventDefault()}
  else if(e.key==='Escape')hideZoom();
});

/* ---------- card zoom ---------- */
var Z=$('zoom'),zbp=null;
function showZoom(bp,x,y,tap){
  var info=(cur&&cur.d.cards[bp])||{};zbp=bp;
  Z.innerHTML=imgTag(bp,true)+'<div class="zn">'+esc(info.name||bp)+'</div><div class="zm">'+esc([info.type,info.culture,info.side==='SHADOW'?'Shadow':info.side==='FREE_PEOPLE'?'Free Peoples':info.side].filter(function(x){return x&&x!=='null'}).join(' · ').replace(/_/g,' '))+'</div>';
  Z.className=tap?'tap':'';Z.style.display='block';
  if(!tap)placeZoom(x,y);
}
function placeZoom(x,y){
  var w=Z.offsetWidth,hh=Math.max(Z.offsetHeight,400),L=x+18,T=y-hh/2;
  if(L+w>innerWidth-8)L=x-18-w;if(L<8)L=8;
  T=Math.max(8,Math.min(innerHeight-hh-8,T));
  Z.style.left=L+'px';Z.style.top=T+'px';Z.style.transform='none';
}
function hideZoom(){Z.style.display='none';zbp=null}
var hoverable=window.matchMedia&&matchMedia('(hover:hover) and (pointer:fine)').matches;
document.addEventListener('mouseover',function(e){
  if(!hoverable||Z.className==='tap')return;
  var c=e.target.closest&&e.target.closest('[data-bp]');
  if(c&&$('board').contains(c)){if(c.getAttribute('data-bp')!==zbp)showZoom(c.getAttribute('data-bp'),e.clientX,e.clientY,false)}else if(zbp)hideZoom()});
document.addEventListener('mousemove',function(e){if(hoverable&&zbp&&Z.className!=='tap')placeZoom(e.clientX,e.clientY)});
document.addEventListener('click',function(e){
  if(Z.style.display==='block'&&Z.className==='tap'){hideZoom();return}
  var c=e.target.closest&&e.target.closest('[data-bp]');
  if(c&&$('board').contains(c))showZoom(c.getAttribute('data-bp'),e.clientX,e.clientY,true)});
window.addEventListener('scroll',function(){if(zbp&&Z.className!=='tap')hideZoom()},{passive:true});

/* ---------- deep link ---------- */
function fromHash(){
  var m=/game=([^&]+)/.exec(location.hash),st=/step=(\d+)/.exec(location.hash);
  if(!m)return;var id=decodeURIComponent(m[1]),k=st?+st[1]:0;
  if(cur&&cur.g.id===id){if(cur.vis[cur.pos]!==k){stop();goStep(k)}return}
  for(var i=0;i<INDEX.length;i++)if(INDEX[i].id===id){openGame(i,k,false);return}
}
window.addEventListener('hashchange',fromHash);fromHash();
})();
</script>
</body>
</html>
"""

if __name__ == "__main__":
    main()
