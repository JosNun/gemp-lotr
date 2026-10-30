// The dashboard: fetches data/data.json (scripts/dashboard_data.py) every minute and draws it as SVG. No libraries.
const NS = 'http://www.w3.org/2000/svg';
const $ = (id) => document.getElementById(id);
const CTX_NAMES = {
  fellowship: 'Fellowship', shadow: 'Shadow', maneuver: 'Maneuver', archery: 'Archery', skirmish: 'Skirmish',
  regroup: 'Regroup', optional: 'Responses', reconcile: 'Reconcile', move: 'Move again?', 'assign fp': 'FP assign',
  assignment: 'Assign. phase', all: 'All',
};
const ROUTE_COLORS = { site9: '#d6b25e', kill: '#d0605a', corrupt: '#a68be0' };
const ROUTE_NAMES = { site9: 'site 9', kill: 'Ring-bearer killed', corrupt: 'corrupted' };
const PALETTE = ['#5b9bd5', '#d6b25e', '#4fb37a', '#a68be0', '#d0605a', '#6cc3c9', '#e0915b', '#9aa5b1'];

function el(tag, attrs = {}, parent) {
  const e = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
  if (parent) parent.appendChild(e);
  return e;
}
function svg(w, h) { return el('svg', { viewBox: `0 0 ${w} ${h}`, width: '100%', preserveAspectRatio: 'xMinYMin meet' }); }
function text(parent, x, y, s, attrs = {}) { const t = el('text', { x, y, ...attrs }, parent); t.textContent = s; return t; }
function html(id, s) { $(id).innerHTML = s; }
function ago(t) {
  const s = Math.max(0, Date.now() / 1000 - t);
  if (s < 90) return `${Math.round(s)}s ago`;
  if (s < 5400) return `${Math.round(s / 60)} min ago`;
  if (s < 172800) return `${Math.round(s / 3600)} h ago`;
  return `${Math.round(s / 86400)} days ago`;
}
function scale(d0, d1, r0, r1) { return (v) => r0 + (v - d0) / (d1 - d0 || 1) * (r1 - r0); }
function niceRange(vals, padFrac = 0.15, minPad = 0.5) {
  let lo = Math.min(...vals), hi = Math.max(...vals);
  const pad = Math.max((hi - lo) * padFrac, minPad);
  return [lo - pad, hi + pad];
}

// ---- small line chart: series [{name, color, points: [[x, y]], dashed}]
function lineChart(series, { w = 260, h = 130, xTicks, yFmt = (v) => v.toFixed(1), refs = [], yRange, minPad = 0.5 } = {}) {
  const s = svg(w, h);
  const all = series.flatMap((x) => x.points);
  if (!all.length) return s;
  const xs = all.map((p) => p[0]), ys = all.map((p) => p[1]).concat(refs.map((r) => r.y));
  const [y0, y1] = yRange || niceRange(ys, 0.15, minPad);
  const X = scale(Math.min(...xs), Math.max(...xs), 34, w - 10), Y = scale(y0, y1, h - 20, 8);
  for (let i = 0; i <= 3; i++) {
    const v = y0 + (y1 - y0) * i / 3;
    el('line', { x1: 34, x2: w - 10, y1: Y(v), y2: Y(v), class: 'grid' }, s);
    text(s, 30, Y(v) + 4, yFmt(v), { 'text-anchor': 'end' });
  }
  for (const t of xTicks || [...new Set(xs)].filter((x) => Number.isInteger(x))) text(s, X(t), h - 5, t, { 'text-anchor': 'middle' });
  for (const r of refs) {
    el('line', { x1: 34, x2: w - 10, y1: Y(r.y), y2: Y(r.y), stroke: r.color || '#8a93a0', 'stroke-dasharray': '4 3' }, s);
  }
  for (const ser of series) {
    if (!ser.points.length) continue;
    const d = ser.points.map((p, i) => `${i ? 'L' : 'M'}${X(p[0]).toFixed(1)},${Y(p[1]).toFixed(1)}`).join('');
    el('path', { d, fill: 'none', stroke: ser.color, 'stroke-width': 2, 'stroke-dasharray': ser.dashed ? '5 3' : '' }, s);
    for (const p of ser.points) el('circle', { cx: X(p[0]), cy: Y(p[1]), r: 2.6, fill: ser.color }, s);
    const last = ser.points[ser.points.length - 1];
    text(s, Math.min(X(last[0]) + 4, w - 40), Y(last[1]) - 5, yFmt(last[1]), { class: 'ink' });
  }
  return s;
}

// ---- horizontal bars with optional 95% whiskers: rows [{label, value, ci, color, note}]
function barsH(rows, { w = 520, rowH = 26, ref, fmt = (v) => v.toFixed(1), range } = {}) {
  const h = rows.length * rowH + 26;
  const s = svg(w, h);
  if (!rows.length) return s;
  const lo = range ? range[0] : Math.min(...rows.map((r) => r.value - (r.ci || 0)), ref ?? Infinity);
  const hi = range ? range[1] : Math.max(...rows.map((r) => r.value + (r.ci || 0)), ref ?? -Infinity);
  const [d0, d1] = range || niceRange([lo, hi], 0.1, 1);
  const X = scale(d0, d1, 120, w - 60);
  for (let i = 0; i <= 4; i++) {
    const v = d0 + (d1 - d0) * i / 4;
    el('line', { x1: X(v), x2: X(v), y1: 4, y2: h - 20, class: 'grid' }, s);
    text(s, X(v), h - 5, fmt(v), { 'text-anchor': 'middle' });
  }
  if (ref !== undefined) el('line', { x1: X(ref), x2: X(ref), y1: 2, y2: h - 18, stroke: '#8a93a0', 'stroke-dasharray': '4 3' }, s);
  rows.forEach((r, i) => {
    const y = 6 + i * rowH;
    text(s, 114, y + rowH / 2 + 2, r.label, { 'text-anchor': 'end', class: r.bold ? 'ink' : '' });
    const x0 = X(Math.max(d0, Math.min(ref ?? d0, r.value))), x1 = X(Math.max(r.value, ref ?? d0));
    el('rect', { x: Math.min(x0, x1), y: y + 5, width: Math.max(2, Math.abs(x1 - x0)), height: rowH - 12, fill: r.color || '#5b9bd5', opacity: 0.85 }, s);
    if (r.ci) {
      el('line', { x1: X(r.value - r.ci), x2: X(r.value + r.ci), y1: y + rowH / 2 - 1, y2: y + rowH / 2 - 1, stroke: '#e6e8eb', 'stroke-width': 1.2 }, s);
      for (const v of [r.value - r.ci, r.value + r.ci])
        el('line', { x1: X(v), x2: X(v), y1: y + rowH / 2 - 5, y2: y + rowH / 2 + 3, stroke: '#e6e8eb', 'stroke-width': 1.2 }, s);
    }
    text(s, X(r.value + (r.ci || 0)) + 6, y + rowH / 2 + 2, fmt(r.value) + (r.note ? `  ${r.note}` : ''), { class: 'ink' });
  });
  return s;
}

// ---- stacked 100% bars: rows [{label, parts: {key: pct}}]
function stacked(rows, { w = 520, rowH = 28, colors, names } = {}) {
  const h = rows.length * rowH + 30;
  const s = svg(w, h);
  const X = scale(0, 100, 150, w - 10);
  rows.forEach((r, i) => {
    const y = 4 + i * rowH;
    text(s, 144, y + rowH / 2 + 3, r.label, { 'text-anchor': 'end', class: 'ink' });
    let acc = 0;
    for (const k of Object.keys(colors)) {
      const v = r.parts[k] || 0;
      if (!v) continue;
      el('rect', { x: X(acc), y: y + 4, width: X(acc + v) - X(acc), height: rowH - 9, fill: colors[k] }, s);
      if (v >= 7) text(s, X(acc + v / 2), y + rowH / 2 + 3, `${v.toFixed(0)}%`, { 'text-anchor': 'middle', fill: '#0f1216', style: 'fill:#0f1216;font-weight:600' });
      acc += v;
    }
  });
  let lx = 150;
  for (const k of Object.keys(colors)) {
    el('rect', { x: lx, y: h - 14, width: 10, height: 10, fill: colors[k] }, s);
    text(s, lx + 14, h - 5, names[k] || k);
    lx += 30 + (names[k] || k).length * 6.2;
  }
  return s;
}

function heatColor(diff) {
  if (diff === null || diff === undefined || isNaN(diff)) return '#1d2229';
  const a = Math.min(1, Math.abs(diff) / 6);
  return diff >= 0 ? `rgba(79,179,122,${0.12 + 0.6 * a})` : `rgba(208,96,90,${0.12 + 0.6 * a})`;
}

// ---- panels
function renderStatus(d) {
  const jobs = d.status.jobs;
  const parts = jobs.length ? jobs.map((j) => `<div class="job"><span class="dot"></span><span class="what">${j.what}</span>
      <span class="detail">${j.detail}</span><span class="el">${j.elapsed}</span></div>`)
    : ['<div class="job idle"><span class="dot"></span><span class="what">idle</span><span class="detail">nothing running</span></div>'];
  const logs = d.status.logs.map((l) => `<code><b>${l.log}</b> (${ago(l.modified)}): ${l.last.replace(/</g, '&lt;')}</code>`).join('');
  html('status', parts.join('') + (logs ? `<details class="logs"><summary>latest log lines</summary>${logs}</details>` : ''));
}

function renderKpis(d) {
  const elo = d.elo, bots = Object.keys(elo);
  const best = bots.sort((a, b) => elo[b].elo - elo[a].elo)[0];
  const agree = d.agreement;
  const bestAgree = Object.entries(agree).sort((a, b) => b[1].all - a[1].all)[0];
  const poolBest = Object.entries(d.pool).sort((a, b) => b[1] - a[1])[0];
  const lastRound = d.rounds[d.rounds.length - 1];
  const tiles = [
    { label: 'Strongest (Elo)', value: best || '—', note: best ? `${elo[best].elo.toFixed(0)} Elo` : '', gold: true },
    { label: 'Most human-like', value: bestAgree ? bestAgree[0] : '—', note: bestAgree ? `${bestAgree[1].all}% of July decisions` : '' },
    { label: 'Best against the pool', value: poolBest ? `${poolBest[1].toFixed(1)}%` : '—', note: poolBest ? poolBest[0] : '' },
    { label: 'Self-play generation', value: lastRound ? `#${lastRound.gen}` : 'not started', note: lastRound ? lastRound.name : 'the round loop comes next' },
  ];
  html('kpis', tiles.map((t) => `<div class="kpi"><div class="label">${t.label}</div>
      <div class="value ${t.gold ? 'gold' : ''}">${t.value}</div><div class="note">${t.note}</div></div>`).join(''));
}

function renderElo(d) {
  $('anchor-name').textContent = d.anchor;
  const rows = Object.entries(d.elo).sort((a, b) => b[1].elo - a[1].elo)
    .map(([b, v]) => ({ label: b, value: v.elo, ci: v.ci || 0, color: b === d.anchor ? '#8a93a0' : v.elo >= 1500 ? '#4fb37a' : '#5b9bd5', bold: b === d.anchor }));
  const body = $('elo');
  body.innerHTML = rows.length ? '' : '<div class="empty">No head-to-heads yet.</div>';
  if (rows.length) body.appendChild(barsH(rows, { ref: 1500, fmt: (v) => v.toFixed(0) }));
}

function renderPool(d) {
  const rows = Object.entries(d.pool).sort((a, b) => b[1] - a[1])
    .map(([b, v]) => ({ label: b, value: v, color: b === d.anchor ? '#8a93a0' : '#d6b25e', bold: b === d.anchor, ci: 1.6 }));
  const body = $('pool');
  body.innerHTML = rows.length ? '' : '<div class="empty">No pool results yet.</div>';
  if (rows.length) body.appendChild(barsH(rows, { ref: 50, fmt: (v) => `${v.toFixed(1)}%` }));
}

function renderAgree(d) {
  const bots = Object.keys(d.agreement).sort((a, b) => d.agreement[a]._mtime - d.agreement[b]._mtime);
  const cols = ['all', ...d.contexts, 'assignment'];
  const anchor = d.agreement[d.anchor] || {};
  let t = '<table class="heat"><tr><th></th>' + cols.map((c) => `<th>${CTX_NAMES[c] || c}</th>`).join('') + '</tr>';
  for (const b of bots) {
    const r = d.agreement[b];
    t += `<tr><th class="row">${b}</th>` + cols.map((c) => {
      const v = r[c];
      if (v === undefined) return '<td></td>';
      const diff = b === d.anchor ? null : v - (anchor[c] ?? v);
      const sign = diff === null ? '' : `<br><span style="font-size:10px;opacity:.85">${diff >= 0 ? '+' : ''}${diff.toFixed(1)}</span>`;
      return `<td style="background:${heatColor(diff)}">${v.toFixed(1)}${sign}</td>`;
    }).join('') + '</tr>';
  }
  html('agree', bots.length ? t + '</table>' : '<div class="empty">No July measurements yet.</div>');
}

function renderTraining(d) {
  const runs = { net2: d.training.filter((r) => r.run === 'net2'), judge: d.training.filter((r) => r.run === 'judge'), judge2: d.training.filter((r) => r.run === 'judge2') };
  const anchor = d.agreement[d.anchor] || {};
  const body = $('train');
  body.innerHTML = '';
  if (!runs.net2.length && !runs.judge.length && !runs.judge2.length) { body.innerHTML = '<div class="empty">No training runs logged.</div>'; return; }
  const grid = document.createElement('div');
  grid.className = 'small-multiples';
  for (const c of d.contexts) {
    const pts = runs.net2.filter((r) => r[c] !== undefined).map((r) => [r.epoch, r[c]]);
    if (!pts.length) continue;
    const cell = document.createElement('div');
    cell.className = 'sm';
    const ahead = anchor[c] !== undefined && pts[pts.length - 1][1] >= anchor[c];
    cell.innerHTML = `<h3>${CTX_NAMES[c] || c}</h3>`;
    cell.appendChild(lineChart([{ name: 'network', color: ahead ? '#4fb37a' : '#5b9bd5', points: pts }],
      { refs: anchor[c] !== undefined ? [{ y: anchor[c] }] : [] }));
    grid.appendChild(cell);
  }
  const auc = document.createElement('div');
  auc.className = 'sm';
  auc.innerHTML = '<h3>Win prediction (AUC) <span class="hint">violet: combined, gold: judge; dashed: current judge</span></h3>';
  auc.appendChild(lineChart([
    { name: 'combined', color: '#a68be0', points: runs.net2.filter((r) => r.auc).map((r) => [r.epoch, r.auc]) },
    { name: 'judge', color: '#d6b25e', points: runs.judge.filter((r) => r.auc).map((r) => [r.epoch, r.auc]) },
  ], { refs: [{ y: 0.670 }], yFmt: (v) => v.toFixed(3), minPad: 0.005 }));
  grid.appendChild(auc);
  const j2 = d.training.filter((r) => r.run === 'judge2' && r.auc);
  if (j2.length) {
    const cell = document.createElement('div');
    cell.className = 'sm';
    cell.innerHTML = '<h3>Judge, 2 positions a game <span class="hint">AUC by epoch; dashed: current judge</span></h3>';
    const ticks = [...new Set(j2.map((r) => r.epoch))].filter((e) => e % 5 === 0 || e === 1);
    cell.appendChild(lineChart([{ name: 'judge2', color: '#6cc3c9', points: j2.map((r) => [r.epoch, r.auc]) }],
      { refs: [{ y: 0.670 }], yFmt: (v) => v.toFixed(3), minPad: 0.005, xTicks: ticks }));
    grid.appendChild(cell);
  }
  body.appendChild(grid);
}

function renderEnds(d) {
  const short = { 'decided on the board': 'people (decided)', 'with inferred concessions': 'people (+ conceded)' };
  const rows = Object.entries(d.peopleEnds).map(([k, v]) => ({ label: short[k] || k, parts: v }));
  for (const r of d.rounds) if (r.ends) rows.push({ label: `gen ${r.gen}`, parts: r.ends });
  const body = $('ends');
  body.innerHTML = '';
  body.appendChild(stacked(rows, { colors: ROUTE_COLORS, names: ROUTE_NAMES }));
}

function renderH2h(d) {
  const rows = [...d.h2h].filter((r) => !r.b.startsWith('pool-')).reverse();
  if (!rows.length) { html('h2h', '<div class="empty">No head-to-heads yet.</div>'); return; }
  html('h2h', '<table class="h2h">' + rows.map((r) => {
    const cls = r.rate - r.ci > 50 ? 'win' : r.rate + r.ci < 50 ? 'loss' : 'even';
    return `<tr><td>${r.a}</td><td class="even">vs</td><td>${r.b}</td><td class="n ${cls}">${r.rate.toFixed(1)}% ± ${r.ci.toFixed(1)}</td></tr>`;
  }).join('') + '</table>');
}

function renderRounds(d) {
  const body = $('rounds');
  if (!d.rounds.length) {
    body.innerHTML = `<div class="empty">The round loop hasn't started. Each generation will add:
      <ul><li>Elo by generation against fixed anchors, and the league matrix (every generation against every other)</li>
      <li>win rate against each pool style; agreement with people per decision</li>
      <li>its games on the play-style map beside people's, coloured by generation (exploration)</li>
      <li>how its games end; style fingerprints (move again, game length, companions lost, minions per Shadow turn, unspent twilight, pass rates)</li>
      <li>cards it plays more or less than people; how decisive its choices are</li>
      <li>where the search overrules the network, and what that is worth; judge calibration; games per hour</li></ul></div>`;
    return;
  }
  body.innerHTML = '';
  const grid = document.createElement('div');
  grid.className = 'small-multiples';
  const metric = (title, key, fmt, color) => {
    const pts = d.rounds.filter((r) => r[key] !== undefined).map((r) => [r.gen, r[key]]);
    if (!pts.length) return;
    const cell = document.createElement('div');
    cell.className = 'sm';
    cell.innerHTML = `<h3>${title}</h3>`;
    cell.appendChild(lineChart([{ name: key, color, points: pts }], { yFmt: fmt }));
    grid.appendChild(cell);
  };
  metric('Elo', 'elo', (v) => v.toFixed(0), '#d6b25e');
  metric('vs its parent', 'vsPrev', (v) => `${v.toFixed(1)}%`, '#4fb37a');
  metric('vs sel10-net6', 'vsNet6', (v) => `${v.toFixed(1)}%`, '#4fb37a');
  metric('Pool mean', 'pool', (v) => `${v.toFixed(1)}%`, '#5b9bd5');
  metric('July agreement', 'agreement', (v) => `${v.toFixed(1)}%`, '#6cc3c9');
  metric('Search overrules the network', 'overrule', (v) => `${v.toFixed(1)}%`, '#a68be0');
  metric('Teacher games per hour', 'gamesPerHour', (v) => v.toFixed(0), '#9aa5b1');
  body.appendChild(grid);
  // style fingerprints: the learner's games (seat A) in the evaluation, generation by generation
  const keys = { turns: 'game length (turns)', site: 'site reached', deadCompanions: 'companions lost',
    moveAgain: 'moves again (share; people 0.23)', minionsPerShadowTurn: 'minions a Shadow turn',
    unspentTwilight: 'twilight left unspent', 'pass:fellowship': 'passes, Fellowship', 'pass:shadow': 'passes, Shadow',
    'pass:skirmish': 'passes, skirmish' };
  const styled = d.rounds.filter((r) => r.style);
  if (styled.length) {
    const h = document.createElement('h3');
    h.textContent = 'Style fingerprints by generation';
    h.style.cssText = 'margin:16px 0 8px;font-size:13px;font-weight:600';
    body.appendChild(h);
    const g2 = document.createElement('div');
    g2.className = 'small-multiples';
    for (const [k, label] of Object.entries(keys)) {
      const pts = styled.filter((r) => r.style[k] !== undefined).map((r) => [r.gen, r.style[k]]);
      if (!pts.length) continue;
      const cell = document.createElement('div');
      cell.className = 'sm';
      cell.innerHTML = `<h3>${label}</h3>`;
      cell.appendChild(lineChart([{ name: k, color: '#e0915b', points: pts }],
        { yFmt: (v) => v.toFixed(2), minPad: 0.02, h: 110 }));
      g2.appendChild(cell);
    }
    body.appendChild(g2);
  }
  const t = document.createElement('table');
  t.className = 'h2h';
  t.style.marginTop = '14px';
  t.innerHTML = '<tr><td>gen</td><td>bot</td><td>parent</td><td class="n">vs parent</td><td class="n">vs net6</td><td class="n">July</td><td class="n">adopted</td></tr>'
    + [...d.rounds].reverse().map((r) => `<tr><td>${r.gen}</td><td>${r.name}</td><td>${r.parent}</td>
      <td class="n">${r.vsPrev !== undefined ? r.vsPrev.toFixed(1) + '% ± ' + r.vsPrevCi.toFixed(1) : ''}</td>
      <td class="n">${r.vsNet6 !== undefined ? r.vsNet6.toFixed(1) + '%' : ''}</td>
      <td class="n">${r.agreement !== undefined ? r.agreement.toFixed(1) + '%' : ''}</td>
      <td class="n ${r.adopted ? 'win' : 'loss'}">${r.adopted ? 'yes' : 'no'}</td></tr>`).join('');
  body.appendChild(t);
}

async function refresh() {
  try {
    const d = await (await fetch(`data/data.json?t=${Date.now()}`)).json();
    $('updated').innerHTML = `data from <b>${ago(d.updated)}</b><br>${new Date(d.updated * 1000).toLocaleString()}`;
    renderStatus(d); renderKpis(d); renderRounds(d); renderElo(d); renderPool(d); renderAgree(d);
    renderTraining(d); renderEnds(d); renderH2h(d);
  } catch (e) {
    $('updated').textContent = `could not load data: ${e}`;
  }
}
refresh();
setInterval(refresh, 60000);
