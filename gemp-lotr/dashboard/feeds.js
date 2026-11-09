// Live panels: one for each stream in data/feed/ (scripts/feed.py, simulation.Feed), drawn from what its records hold.
// A stream needs no code here -- its fields say what to draw:
//   done + total -> progress, rate and time left;  wins + games -> a matchup with its 95% interval, and a game ticker;
//   a YYYY-MM field -> a calendar;  words with few values -> counts;  changing numbers -> small line charts;
//   note/msg/text -> a log.  <stream>.meta.json may set title, hint, x, y (fields to chart), hide, color.
// Uses app.js's helpers (el, svg, text, scale, niceRange, ago, PALETTE, ROUTE_COLORS, ROUTE_NAMES).
const FEEDS = 'data/feeds/';
const SKIP = new Set(['t', 'done', 'total']);
const LIVE_SECONDS = 600;   // a stream written to in the last 10 minutes is live
const RECENT_HOURS = 24, RECENT_MAX = 8;  // drawn on load; the rest are listed, and drawn when opened
const fmtNum = (v) => (Math.abs(v) >= 1e6 ? `${(v / 1e6).toFixed(2)}M` : Math.abs(v) >= 1e4 ? `${(v / 1e3).toFixed(1)}k`
  : Number.isInteger(v) ? `${v}` : Math.abs(v) >= 100 ? v.toFixed(0) : Math.abs(v) >= 1 ? v.toFixed(2) : v.toFixed(3));
const fmtDur = (s) => (s < 90 ? `${Math.round(s)}s` : s < 5400 ? `${Math.round(s / 60)} min`
  : s < 172800 ? `${(s / 3600).toFixed(1)} h` : `${(s / 86400).toFixed(1)} days`);
const MONTH = /^\d{4}-\d\d$/;
const QUIET = { unfinished: '#4a525e', other: '#6b7480', none: '#4a525e' };  // words that are not an outcome
const colorOf = (k, i) => ROUTE_COLORS[k] || QUIET[k] || PALETTE[i % PALETTE.length];

// scripts/feed_index.py writes data/feeds/index.json (each stream's last record, newest first) and a thinned copy of
// each stream: the page fetched 67 streams whole, 43 MB, every 15 seconds; now the index, the recent streams' copies
// when they change, and an older stream only when it is opened.
const cache = {};  // stream -> {t, data}
async function loadIndex() {
  return (await (await fetch(`${FEEDS}index.json?t=${Date.now()}`)).json()).streams;
}

async function loadStream(entry) {
  const c = cache[entry.name];
  if (c && c.t === entry.t) return c.data;
  const data = await (await fetch(`${FEEDS}${encodeURIComponent(entry.name)}.json?t=${entry.t}`)).json();
  cache[entry.name] = { t: entry.t, data };
  return data;
}

// what the fields are: numbers that change, numbers that do not, words with few values, words that never change
function profile(records, meta) {
  const fields = {};
  for (const r of records) for (const [k, v] of Object.entries(r)) {
    if (SKIP.has(k) || (meta.hide || []).includes(k)) continue;
    const f = fields[k] || (fields[k] = { nums: 0, strs: 0, values: new Set() });
    if (typeof v === 'number') f.nums++; else if (typeof v === 'string') { f.strs++; if (f.values.size < 40) f.values.add(v); }
  }
  const out = { numeric: [], constant: [], category: [], month: null, text: [] };
  for (const [k, f] of Object.entries(fields)) {
    if (f.nums >= f.strs && f.nums) {
      const vals = records.map((r) => r[k]).filter((v) => typeof v === 'number');
      (Math.min(...vals) === Math.max(...vals) ? out.constant : out.numeric).push(k);
    } else if (f.strs) {
      const vals = [...f.values];
      if (vals.every((v) => MONTH.test(v)) && !out.month) out.month = k;
      else if (['note', 'msg', 'text', 'message'].includes(k)) out.text.push(k);
      else if (f.values.size === 1) out.constant.push(k);
      else if (f.values.size <= 12) out.category.push(k);
      else out.text.push(k);
    }
  }
  return out;
}

function div(cls, parent, html) {
  const d = document.createElement('div');
  if (cls) d.className = cls;
  if (html !== undefined) d.innerHTML = html;
  if (parent) parent.appendChild(d);
  return d;
}

// ---- progress: done/total, the rate over the last stretch, time left
function progress(records, parent) {
  const last = records[records.length - 1];
  const p = Math.min(1, last.done / last.total);
  // the rate over the last hour, or since the start; none over less than two minutes (records written at once)
  const recent = records.filter((r) => typeof r.done === 'number' && r.t >= last.t - 3600);
  const first = recent.length > 1 && last.t - recent[0].t > 120 ? recent[0] : records.find((r) => typeof r.done === 'number');
  const rate = last.t - first.t > 120 ? (last.done - first.done) / (last.t - first.t) : 0;
  const left = rate > 0 ? (last.total - last.done) / rate : null;
  const box = div('fprog', parent);
  div('fbar', box, `<i style="width:${(100 * p).toFixed(1)}%"></i>`);
  div('fprog-text', box, `<b>${fmtNum(last.done)}</b> of ${fmtNum(last.total)} &middot; ${(100 * p).toFixed(1)}%`
    + (p >= 1 ? ' &middot; done'
      : Date.now() / 1000 - last.t > LIVE_SECONDS ? ` &middot; stopped ${ago(last.t)}`  // a run that ended short
      : left !== null ? ` &middot; ${fmtNum(rate * 3600)} an hour &middot; about ${fmtDur(left)} left` : ''));
}

// ---- a matchup: A's win rate with its 95% interval as a tug of war, its path as games came in, and a game ticker
function matchup(records, parent, prof) {
  const last = records[records.length - 1];
  const n = last.games, p = n ? last.wins / n : 0.5, ci = n ? 1.96 * Math.sqrt(p * (1 - p) / n) : 0.5;
  const a = last.a || 'A', b = last.b || 'B';
  const w = 560, h = 58, s = svg(w, h);
  const X = scale(0.3, 0.7, 10, w - 10);
  el('rect', { x: 10, y: 22, width: w - 20, height: 12, fill: '#1f242c' }, s);
  el('rect', { x: X(Math.max(0.3, p - ci)), y: 22, width: Math.max(2, X(Math.min(0.7, p + ci)) - X(Math.max(0.3, p - ci))), height: 12, fill: '#5b9bd5', opacity: 0.35 }, s);
  el('line', { x1: X(0.5), x2: X(0.5), y1: 16, y2: 40, stroke: '#8a93a0', 'stroke-dasharray': '3 3' }, s);
  el('circle', { cx: X(Math.min(0.7, Math.max(0.3, p))), cy: 28, r: 6, fill: p > 0.5 + ci ? '#4fb37a' : p < 0.5 - ci ? '#d0605a' : '#d6b25e' }, s);
  text(s, 10, 12, a, { class: 'ink' });
  text(s, w - 10, 12, b, { class: 'ink', 'text-anchor': 'end' });
  text(s, X(0.5), 54, `${(100 * p).toFixed(1)}% ± ${(100 * ci).toFixed(1)} for A over ${n} games`, { 'text-anchor': 'middle', class: 'ink' });
  for (const v of [0.3, 0.4, 0.6, 0.7]) text(s, X(v), 54, `${v * 100}`, { 'text-anchor': 'middle' });
  div('', parent).appendChild(s);
  // the path, with its interval
  const pts = records.filter((r) => r.games > 0).map((r) => [r.games, r.wins / r.games]);
  if (pts.length > 4) {
    const step = Math.max(1, Math.floor(pts.length / 300));
    const ps = pts.filter((_, i) => i % step === 0 || i === pts.length - 1);
    const cw = 560, ch = 120, c = svg(cw, ch);
    const PX = scale(ps[0][0], ps[ps.length - 1][0], 34, cw - 10), PY = scale(0.3, 0.7, ch - 18, 6);
    for (const v of [0.3, 0.4, 0.5, 0.6, 0.7]) {
      el('line', { x1: 34, x2: cw - 10, y1: PY(v), y2: PY(v), class: v === 0.5 ? '' : 'grid', stroke: v === 0.5 ? '#4a525e' : '' }, c);
      text(c, 30, PY(v) + 4, `${v * 100}%`, { 'text-anchor': 'end' });
    }
    const band = ps.map(([g, q]) => [g, Math.min(0.7, q + 1.96 * Math.sqrt(q * (1 - q) / g))])
      .concat([...ps].reverse().map(([g, q]) => [g, Math.max(0.3, q - 1.96 * Math.sqrt(q * (1 - q) / g))]));
    el('path', { d: band.map((q, i) => `${i ? 'L' : 'M'}${PX(q[0]).toFixed(1)},${PY(q[1]).toFixed(1)}`).join('') + 'Z', fill: '#5b9bd5', opacity: 0.18 }, c);
    el('path', { d: ps.map((q, i) => `${i ? 'L' : 'M'}${PX(q[0]).toFixed(1)},${PY(Math.min(0.7, Math.max(0.3, q[1]))).toFixed(1)}`).join(''), fill: 'none', stroke: '#5b9bd5', 'stroke-width': 2 }, c);
    text(c, cw - 10, ch - 4, `${ps[ps.length - 1][0]} games`, { 'text-anchor': 'end' });
    div('', parent).appendChild(c);
  }
  // the last games: who won (up: A) and how (colour)
  const ticker = div('ticker', parent);
  const tail = records.slice(-120);
  let prev = records.length > tail.length ? records[records.length - tail.length - 1] : null;
  for (const r of tail) {
    const aWon = prev ? r.wins > prev.wins : r.wins > 0;
    const fin = prev ? r.games > prev.games : r.games > 0;
    const route = prof.category.includes('route') ? r.route : null;
    const i = document.createElement('i');
    i.className = fin ? (aWon ? 'up' : 'down') : 'none';
    i.style.background = ROUTE_COLORS[route] || (fin ? (aWon ? '#4fb37a' : '#d0605a') : '#4a525e');
    i.title = `${fin ? (aWon ? 'A won' : 'B won') : 'unfinished'}${route ? ', ' + (ROUTE_NAMES[route] || route) : ''}${r.turns ? `, ${r.turns} turns` : ''}`;
    ticker.appendChild(i);
    prev = r;
  }
}

// ---- a calendar of months, coloured by a number
function calendar(records, parent, key, colorBy) {
  const by = {};
  for (const r of records) if (r[key]) by[r[key]] = r;
  const months = Object.keys(by).sort();
  if (!months.length) return;
  const vals = months.map((m) => by[m][colorBy]).filter((v) => typeof v === 'number');
  const lo = Math.min(...vals), hi = Math.max(...vals);
  const y0 = +months[0].slice(0, 4), y1 = +months[months.length - 1].slice(0, 4);
  const t = document.createElement('table');
  t.className = 'cal';
  t.innerHTML = '<tr><th></th>' + 'JFMAMJJASOND'.split('').map((c) => `<th>${c}</th>`).join('') + '</tr>';
  for (let y = y0; y <= y1; y++) {
    const row = document.createElement('tr');
    row.innerHTML = `<th class="row">${y}</th>`;
    for (let m = 1; m <= 12; m++) {
      const k = `${y}-${String(m).padStart(2, '0')}`;
      const td = document.createElement('td');
      const r = by[k];
      if (r && typeof r[colorBy] === 'number') {
        const f = hi > lo ? (r[colorBy] - lo) / (hi - lo) : 1;
        td.style.background = `rgba(214,178,94,${(0.25 + 0.75 * f).toFixed(2)})`;
        td.textContent = fmtNum(r[colorBy]);
        td.title = Object.entries(r).filter(([kk]) => kk !== 't').map(([kk, v]) => `${kk}: ${v}`).join('\n');
      }
      row.appendChild(td);
    }
    t.appendChild(row);
  }
  const box = div('', parent);
  div('cal-label', box, `${colorBy} by month`);
  box.appendChild(t);
}

// ---- counts of a word field (how games ended, ...)
function counts(records, parent, key) {
  const c = {};
  for (const r of records) if (typeof r[key] === 'string') c[r[key]] = (c[r[key]] || 0) + 1;
  const total = Object.values(c).reduce((x, y) => x + y, 0);
  const entries = Object.entries(c).sort((x, y) => y[1] - x[1]);
  const box = div('counts', parent);
  div('cal-label', box, key);
  const bar = div('stack', box);
  entries.forEach(([k, n], i) => {
    const seg = document.createElement('i');
    seg.style.width = `${(100 * n / total).toFixed(2)}%`;
    seg.style.background = colorOf(k, i);
    seg.title = `${ROUTE_NAMES[k] || k}: ${n}`;
    bar.appendChild(seg);
  });
  div('legend', box, entries.map(([k, n], i) => `<span><i style="background:${colorOf(k, i)}"></i>${ROUTE_NAMES[k] || k} ${(100 * n / total).toFixed(0)}%</span>`).join(''));
}

// ---- a small line chart, x numbers or labels (months), y one field; a value per event (a game's turns) over many
// records is drawn as its running mean, which says something, rather than as noise
function mini(records, key, xKey, color, xIsLabel) {
  let pts = records.map((r, i) => [xIsLabel ? i : xKey ? r[xKey] : r.t, r[key]])
    .filter(([x, y]) => typeof x === 'number' && typeof y === 'number');
  if (pts.length > 60 && jumpy(pts)) {
    const k = Math.max(10, Math.floor(pts.length / 15));
    let sum = 0;
    pts = pts.map(([x, y], i) => { sum += y - (i >= k ? pts[i - k][1] : 0); return [x, sum / Math.min(i + 1, k)]; }).slice(k - 1);
    mini.smoothed = k;
  } else mini.smoothed = 0;
  const step = Math.max(1, Math.floor(pts.length / 400));
  const ps = pts.filter((_, i) => i % step === 0 || i === pts.length - 1);
  const w = 260, h = 120, s = svg(w, h);
  if (ps.length < 2) { text(s, 10, 20, ps.length ? fmtNum(ps[0][1]) : '', { class: 'ink' }); return s; }
  const xs = ps.map((p) => p[0]), ys = ps.map((p) => p[1]);
  const [y0, y1] = niceRange(ys, 0.12, Math.max(1e-6, (Math.max(...ys) - Math.min(...ys)) * 0.05 || Math.abs(ys[0]) * 0.02 || 0.01));
  const X = scale(Math.min(...xs), Math.max(...xs), 46, w - 10), Y = scale(y0, y1, h - 20, 8);
  for (let i = 0; i <= 2; i++) {
    const v = y0 + (y1 - y0) * i / 2;
    el('line', { x1: 46, x2: w - 10, y1: Y(v), y2: Y(v), class: 'grid' }, s);
    text(s, 42, Y(v) + 4, fmtNum(v), { 'text-anchor': 'end' });
  }
  const xl = (x) => (xIsLabel ? (records[x] || {})[xKey] || '' : xKey ? fmtNum(x) : new Date(x * 1000).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }));
  text(s, 46, h - 5, xl(xs[0]));
  text(s, w - 10, h - 5, xl(xs[xs.length - 1]), { 'text-anchor': 'end' });
  el('path', { d: ps.map((p, i) => `${i ? 'L' : 'M'}${X(p[0]).toFixed(1)},${Y(p[1]).toFixed(1)}`).join(''), fill: 'none', stroke: color, 'stroke-width': 2 }, s);
  if (ps.length <= 40) for (const p of ps) el('circle', { cx: X(p[0]), cy: Y(p[1]), r: 2.4, fill: color }, s);
  const last = ps[ps.length - 1];
  text(s, w - 10, 12, fmtNum(last[1]), { class: 'ink', 'text-anchor': 'end' });
  return s;
}

// ---- meta.grid {x, y, value, where, center}: a heatmap of value by x and y (the latest record at each cell), for a
// tuning run's grid; cells coloured by distance from center (default the mean), green above and red below
function grid(records, parent, g) {
  const rs = records.filter((r) => Object.entries(g.where || {}).every(([k, v]) => r[k] === v)
    && typeof r[g.value] === 'number' && r[g.x] !== undefined && r[g.y] !== undefined);
  if (!rs.length) return;
  const xs = [...new Set(rs.map((r) => r[g.x]))].sort((a, b) => a - b);
  const ys = [...new Set(rs.map((r) => r[g.y]))].sort((a, b) => a - b);
  const cell = {};
  for (const r of rs) cell[`${r[g.x]}|${r[g.y]}`] = r;
  const vals = rs.map((r) => r[g.value]);
  const c = g.center ?? vals.reduce((a, b) => a + b, 0) / vals.length;
  const span = Math.max(...vals.map((v) => Math.abs(v - c)), 1e-9);
  const t = document.createElement('table');
  t.className = 'heat';
  t.innerHTML = `<tr><th>${g.y} \\ ${g.x}</th>` + xs.map((x) => `<th>${x}</th>`).join('') + '</tr>'
    + ys.map((y) => `<tr><th class="row">${y}</th>` + xs.map((x) => {
      const r = cell[`${x}|${y}`];
      if (!r) return '<td style="color:var(--faint)">&middot;</td>';
      const f = (r[g.value] - c) / span;
      const bg = f >= 0 ? `rgba(79,179,122,${(0.15 + 0.6 * f).toFixed(2)})` : `rgba(208,96,90,${(0.15 - 0.6 * f).toFixed(2)})`;
      return `<td style="background:${bg}" title="${Object.entries(r).filter(([k]) => k !== 't').map(([k, v]) => `${k}: ${v}`).join('\n')}">${fmtNum(r[g.value])}</td>`;
    }).join('') + '</tr>').join('');
  const box = div('', parent);
  div('cal-label', box, g.title || `${g.value} by ${g.x} and ${g.y}`);
  box.appendChild(t);
}

// ---- meta.scatter {x, y, ci, label: [fields], ref}: one point a record, with a +-ci bar on y if given -- a tuning
// run's settings as strength against cost; the best y in gold
function scatter(records, parent, g) {
  const rs = records.filter((r) => typeof r[g.x] === 'number' && typeof r[g.y] === 'number');
  if (!rs.length) return;
  const ci = (r) => (typeof g.ci === 'number' ? g.ci : r[g.ci] || 0);
  const w = 560, h = 220, s = svg(w, h);
  const [x0, x1] = niceRange(rs.map((r) => r[g.x]), 0.1, 0.2);
  const [y0, y1] = niceRange(rs.flatMap((r) => [r[g.y] - ci(r), r[g.y] + ci(r)]).concat(g.ref ?? []), 0.08, 0.5);
  const X = scale(x0, x1, 46, w - 14), Y = scale(y0, y1, h - 26, 10);
  for (let i = 0; i <= 4; i++) {
    const v = y0 + (y1 - y0) * i / 4, u = x0 + (x1 - x0) * i / 4;
    el('line', { x1: 46, x2: w - 14, y1: Y(v), y2: Y(v), class: 'grid' }, s);
    text(s, 42, Y(v) + 4, fmtNum(v), { 'text-anchor': 'end' });
    text(s, X(u), h - 8, fmtNum(u), { 'text-anchor': 'middle' });
  }
  if (g.ref !== undefined) el('line', { x1: 46, x2: w - 14, y1: Y(g.ref), y2: Y(g.ref), stroke: '#8a93a0', 'stroke-dasharray': '4 3' }, s);
  const best = rs.reduce((a, b) => (b[g.y] > a[g.y] ? b : a));
  for (const r of rs) {
    const color = r === best ? '#d6b25e' : '#5b9bd5';
    if (ci(r)) el('line', { x1: X(r[g.x]), x2: X(r[g.x]), y1: Y(r[g.y] - ci(r)), y2: Y(r[g.y] + ci(r)), stroke: color, opacity: 0.5 }, s);
    const c = el('circle', { cx: X(r[g.x]), cy: Y(r[g.y]), r: 4.5, fill: color }, s);
    const t = el('title', {}, c);
    t.textContent = (g.label || []).map((k) => `${k} ${r[k]}`).join(', ') + ` -> ${r[g.y]}`;
  }
  text(s, w - 14, h - 8 - 12, g.x, { 'text-anchor': 'end' });
  const box = div('', parent);
  div('cal-label', box, g.title || `${g.y} against ${g.x}`);
  box.appendChild(s);
}

// whether a series moves more from one record to the next than overall: values per event, not a quantity tracked
function jumpy(pts) {
  let step = 0;
  for (let i = 1; i < pts.length; i++) step += Math.abs(pts[i][1] - pts[i - 1][1]);
  // the spread of the middle 80%: a few outliers (an unfinished game's 0 turns) would otherwise hide the jumpiness
  const ys = pts.map((p) => p[1]).sort((a, b) => a - b);
  const spread = ys[Math.floor(ys.length * 0.9)] - ys[Math.floor(ys.length * 0.1)];
  return step / (pts.length - 1) > 0.25 * spread;
}

function renderStream({ name, records: raw, meta, total }, parent) {
  const prof0 = profile(raw, meta);
  // records keyed by a month: one a month (the latest), in calendar order
  let records = raw;
  if (prof0.month && (meta.x || prof0.month) === prof0.month) {
    const by = {};
    for (const r of raw) if (r[prof0.month]) by[r[prof0.month]] = r;
    records = Object.keys(by).sort().map((k) => by[k]);
  }
  const last = raw[raw.length - 1];
  const live = Date.now() / 1000 - last.t < LIVE_SECONDS;
  const prof = profile(records, meta);
  const panel = div(`panel feed ${live ? 'live' : ''}`, parent);
  const chips = prof.constant.filter((k) => typeof last[k] === 'string' || typeof last[k] === 'number')
    .map((k) => `<span class="chip">${k} <b>${last[k]}</b></span>`).join('');
  panel.innerHTML = `<h2><span class="dot"></span>${meta.title || name}<span class="hint">${meta.hint || ''}</span>`
    + `<span class="when">${total || raw.length} records &middot; ${ago(last.t)}</span></h2>`
    + (chips ? `<div class="chips">${chips}</div>` : '');
  const body = div('body', panel);
  if (typeof last.done === 'number' && typeof last.total === 'number' && last.total > 0) progress(raw, body);
  const isMatch = prof.numeric.includes('wins') && (prof.numeric.includes('games') || prof.constant.includes('games'));
  if (isMatch) matchup(records, body, prof);
  const numeric = prof.numeric.filter((k) => !(isMatch && ['wins', 'games'].includes(k)));
  if (prof.month) calendar(records, body, prof.month, meta.color || numeric[0]);
  if (meta.grid || meta.scatter) {
    for (const g of [].concat(meta.grid || [])) grid(records, div('', body), g);
    for (const g of [].concat(meta.scatter || [])) scatter(records, div('', body), g);
  }
  const cats = prof.category.filter((k) => k !== meta.x);
  if (cats.length) { const row = div('cats', body); for (const k of cats.slice(0, 3)) counts(records, row, k); }
  // the x the charts are against: meta, else epoch/step/done, else the month, else time
  let xKey = meta.x || ['epoch', 'step', 'gen', 'done'].find((k) => prof.numeric.includes(k)) || prof.month || null;
  const xIsLabel = xKey && records.some((r) => typeof r[xKey] === 'string');
  const series = (meta.y || numeric).filter((k) => k !== xKey && numeric.includes(k)).slice(0, 12);
  if (series.length) {
    const grid = div('small-multiples', body);
    series.forEach((k, i) => {
      const cell = div('sm', grid, `<h3>${k.replace(/_/g, ' ')}</h3>`);
      cell.appendChild(mini(records, k, xKey, PALETTE[i % PALETTE.length], xIsLabel));
      if (mini.smoothed) cell.querySelector('h3').insertAdjacentHTML('beforeend', ` <span class="hint">mean of ${mini.smoothed}</span>`);
    });
  }
  if (prof.text.length) {
    const log = div('flog', body);
    for (const r of records.filter((x) => prof.text.some((k) => x[k])).slice(-6).reverse())
      div('', log, `<span>${new Date(r.t * 1000).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}</span> ${prof.text.map((k) => r[k]).filter(Boolean).join(' · ')}`);
  }
  return panel;
}

async function refreshFeeds() {
  const root = $('live');
  let index;
  try {
    index = await loadIndex();
  } catch (e) {
    root.innerHTML = `<div class="empty">No live feeds (${e}).</div>`;
    return;
  }
  if (!index.length) {
    root.innerHTML = '<div class="empty">No feeds yet: <code>python3 -I scripts/feed.py &lt;stream&gt; key=value ...</code></div>';
    return;
  }
  const now = Date.now() / 1000;
  const recent = index.filter((e) => now - e.t < RECENT_HOURS * 3600).slice(0, RECENT_MAX);
  const older = index.filter((e) => !recent.includes(e));
  const opened = new Set([...root.querySelectorAll('details.stream[open]')].map((d) => d.dataset.name));
  const earlierOpen = !!root.querySelector('details.earlier[open]');
  const drawn = await Promise.all(recent.map((e) => loadStream(e).catch(() => null)));
  root.innerHTML = '';
  const grid = div('feeds', root);
  drawn.forEach((d) => { if (d && d.records.length) renderStream(d, grid); });
  if (older.length) {
    const box = document.createElement('details');
    box.className = 'earlier';
    box.open = earlierOpen;
    box.innerHTML = `<summary>${older.length} earlier feeds</summary>`;
    for (const e of older) {
      const row = document.createElement('details');
      row.className = 'stream';
      row.dataset.name = e.name;
      row.innerHTML = `<summary><b>${(e.meta && e.meta.title) || e.name}</b> <span class="when">${e.total} records &middot; ${ago(e.t)}</span></summary>`;
      const body = div('feeds', row);
      const draw = async () => {
        if (body.childElementCount) return;
        const d = await loadStream(e).catch(() => null);
        if (d && d.records.length) renderStream(d, body);
      };
      row.addEventListener('toggle', () => { if (row.open) draw(); });
      if (opened.has(e.name)) { row.open = true; draw(); }
      box.appendChild(row);
    }
    root.appendChild(box);
  }
}
refreshFeeds();
setInterval(refreshFeeds, 15000);
