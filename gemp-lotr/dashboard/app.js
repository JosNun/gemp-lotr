// The dashboard: fetches bots.json (hand-kept: every bot version, its cost, strength and July agreement) and
// data/data.json (scripts/dashboard_data.py: the status bar, agreement by decision) every minute; draws SVG, no libraries.
// To add a bot to the Pareto chart, add one entry to dashboard/bots.json -- its "_how_to_add" says what goes in it.
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

// ---- the Pareto panel: every bot version in bots.json, cost against strength (or July agreement), and the frontier.
const KIND_COLORS = { plain: '#5b9bd5', search: '#a68be0', network: '#6cc3c9', style: '#9aa5b1' };
const KIND_NAMES = { plain: 'plain bot', search: 'play-time search', network: 'network alone', style: 'style bot' };
const pareto = { reg: null, raw: '', y: 'strength', pinned: null, width: 0 };
const esc = (s) => String(s ?? '').replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
const fmtCost = (v) => (v >= 10 ? v.toFixed(1) : v.toFixed(2));

// Strength: each entry holds one BotVsBot result against another bot of the file; chained to the reference by adding
// the links' log-odds and their variances (a 95% interval of +-ci points at rate p is a log-odds s.e. of
// ci / 1.96 / (p (1 - p))). Cost: the game total less the opponent's side; a game against itself is halved.
function derive(reg) {
  const by = Object.fromEntries(reg.bots.map((b) => [b.id, b]));
  const ref = reg.reference, sMemo = {}, cMemo = {};
  const strength = (id, seen = []) => {
    if (id === ref) return { logit: 0, v: 0, path: [] };
    if (id in sMemo) return sMemo[id];
    const b = by[id], s = b && b.strength;
    if (!s || seen.includes(id)) return null;
    const mine = s.a === id, other = mine ? s.b : s.a;
    const p = (mine ? s.rate : 100 - s.rate) / 100;
    const up = strength(other, [...seen, id]);
    if (!up) return null;
    const se = s.ci / 196 / (p * (1 - p));
    return (sMemo[id] = { logit: up.logit + Math.log(p / (1 - p)), v: up.v + se * se, path: [other, ...up.path] });
  };
  const side = (id, seen = []) => {
    if (id in cMemo) return cMemo[id];
    const b = by[id], c = b && b.cost;
    if (!c || seen.includes(id)) return null;
    if (c.vs === id) return (cMemo[id] = { v: c.total / 2, how: `${c.total} a game against itself, halved` });
    const o = side(c.vs, [...seen, id]);
    if (!o) return (cMemo[id] = null);
    return (cMemo[id] = { v: c.total - o.v, how: `${c.total} a game against ${c.vs}, less ${c.vs}'s ${fmtCost(o.v)}` });
  };
  const sig = (x) => 100 / (1 + Math.exp(-x));
  return reg.bots.map((b, i) => {
    const s = strength(b.id), c = side(b.id);
    const st = s && {
      p: sig(s.logit), lo: sig(s.logit - 1.96 * Math.sqrt(s.v)), hi: sig(s.logit + 1.96 * Math.sqrt(s.v)),
      direct: s.path.length <= 1, path: s.path, ref: b.id === ref,
    };
    return { ...b, i, st, cs: c, july: b.july && b.july.measure === 'order-free' ? b.july.value : null };
  });
}

function frontier(pts) {
  const sorted = [...pts].sort((a, b) => a.x - b.x || b.y - a.y);
  const out = [];
  let best = -Infinity;
  for (const p of sorted) if (p.y > best) { out.push(p); best = p.y; }
  return out;
}

function popHtml(b, reg) {
  const ref = reg.reference;
  let strength = 'not measured on the test decks';
  if (b.st && b.st.ref) strength = `the reference: 50% by definition`;
  else if (b.st) {
    const s = b.strength;
    const via = b.st.path.slice(0, -1);
    const how = b.st.direct ? 'played against it directly'
      : `chained through ${via.length > 4 ? via.length + ' bots' : via.map(esc).join(' → ')} (${b.st.path.length} links)`;
    strength = `<b>${b.st.p.toFixed(1)}%</b> against ${ref} (95%: ${b.st.lo.toFixed(1)}–${b.st.hi.toFixed(1)}), ${how}.
      The result recorded: ${esc(s.a)} vs ${esc(s.b)}, A ${s.rate}% ± ${s.ci}.`;
  }
  let cost = 'not measured';
  if (b.cs) cost = `<b>${fmtCost(b.cs.v)}</b> CPU-s a game for its side (${esc(b.cs.how)})`;
  else if (b.cost) cost = `${b.cost.total} CPU-s a game against ${esc(b.cost.vs)}, both sides; ${esc(b.cost.vs)}'s own side was never measured, so off the chart`;
  const july = b.july ? `<b>${b.july.value}%</b> (${b.july.measure === 'order-free' ? 'order-forgiving measure' : 'strict measure, from before 2026-11-01'})` : 'not measured';
  const src = [['strength', b.strength && b.strength.source], ['cost', b.cost && b.cost.source], ['July', b.july && b.july.source]]
    .filter((x) => x[1]).map(([k, v]) => `<li><span>${k}:</span> ${esc(v)}</li>`).join('');
  return `<button class="pop-x" aria-label="close">×</button>
    <div class="pop-name">${esc(b.id)}</div>
    <div class="pop-meta">${esc(b.date)} · ${KIND_NAMES[b.kind] || esc(b.kind)}${b.parent ? ` · from ${esc(b.parent)}` : ''}</div>
    <p>${esc(b.what)}</p>${b.note ? `<p class="pop-note">${esc(b.note)}</p>` : ''}
    <dl><dt>Cost</dt><dd>${cost}</dd><dt>Strength</dt><dd>${strength}</dd><dt>July</dt><dd>${july}</dd></dl>
    ${src ? `<ul class="pop-src">${src}</ul>` : ''}`;
}

function closePop() {
  const p = $('pop');
  if (p) p.hidden = true;
  pareto.pinned = null;
  document.querySelectorAll('.pareto-pt.sel').forEach((e) => e.classList.remove('sel'));
}

// The popover lives on <body> at page coordinates, so it scrolls with what it points at; it is clamped to the viewport's
// width and goes below the anchor unless there is more room above.
function openPop(b, anchor, pin) {
  let p = $('pop');
  if (!p) {
    p = document.createElement('div');
    p.id = 'pop';
    p.setAttribute('role', 'dialog');
    document.body.appendChild(p);
    p.addEventListener('click', (e) => { if (e.target.closest('.pop-x')) closePop(); e.stopPropagation(); });
  }
  p.innerHTML = popHtml(b, pareto.reg);
  p.hidden = false;
  const vw = document.documentElement.clientWidth;
  const w = Math.min(360, vw - 16);
  p.style.width = `${w}px`;
  const r = anchor.getBoundingClientRect();
  const cx = r.left + r.width / 2;
  p.style.left = `${window.scrollX + Math.max(8, Math.min(cx - w / 2, vw - w - 8))}px`;
  const h = p.offsetHeight, below = window.innerHeight - r.bottom, above = r.top;
  const top = below >= h + 10 || below >= above ? r.bottom + 8 : r.top - h - 8;
  p.style.top = `${window.scrollY + Math.max(8 - window.scrollY, top)}px`;
  pareto.pinned = pin ? b.id : null;
  document.querySelectorAll('.pareto-pt').forEach((e) => e.classList.toggle('sel', e.dataset.id === b.id));
}

document.addEventListener('click', (e) => {
  if (!e.target.closest('#pop') && !e.target.closest('[data-id]')) closePop();
});
window.addEventListener('resize', () => {
  if (pareto.reg && $('pareto').clientWidth !== pareto.width) drawPareto();
});

function renderPareto(reg) {
  const raw = JSON.stringify(reg);
  if (raw === pareto.raw) return;  // the minute's refresh: redraw only when the file changed
  pareto.raw = raw;
  pareto.reg = reg;
  drawPareto();
}

function drawPareto() {
  const reg = pareto.reg, body = $('pareto');
  const bots = derive(reg);
  const byId = Object.fromEntries(bots.map((b) => [b.id, b]));
  const mode = pareto.y;
  const pts = bots.filter((b) => b.cs && b.cs.v > 0 && (mode === 'strength' ? b.st : b.july !== null))
    .map((b) => ({ b, x: b.cs.v, y: mode === 'strength' ? b.st.p : b.july, lo: mode === 'strength' ? b.st.lo : null, hi: mode === 'strength' ? b.st.hi : null }));
  const front = frontier(pts), onFront = new Set(front.map((p) => p.b.id));
  const wasOpen = body.querySelector('details.versions')?.open;

  const W = Math.max(300, body.clientWidth), small = W < 600;
  pareto.width = body.clientWidth;
  const H = small ? 360 : 440, L0 = 42, R0 = W - 12, T0 = 10, B0 = H - 38;
  const s = svg(W, H);
  const lx = pts.map((p) => Math.log10(p.x));
  const x0 = Math.min(...lx) - 0.12, x1 = Math.max(...lx) + 0.12;
  const X = (v) => scale(x0, x1, L0, R0)(Math.log10(v));
  const ys = pts.flatMap((p) => (p.lo !== null ? [p.lo, p.hi] : [p.y])).concat(mode === 'strength' ? [50] : []);
  const [y0, y1] = niceRange(ys, 0.06, mode === 'strength' ? 1 : 0.2);
  const Y = scale(y0, y1, B0, T0);
  const step = [0.5, 1, 2, 5, 10].find((st) => (y1 - y0) / st <= (small ? 6 : 9));
  for (let v = Math.ceil(y0 / step) * step; v <= y1; v += step) {
    el('line', { x1: L0, x2: R0, y1: Y(v), y2: Y(v), class: 'grid' }, s);
    text(s, L0 - 5, Y(v) + 4, `${+v.toFixed(1)}%`, { 'text-anchor': 'end' });
  }
  for (const t of [0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50]) {
    if (Math.log10(t) < x0 || Math.log10(t) > x1) continue;
    el('line', { x1: X(t), x2: X(t), y1: T0, y2: B0, class: 'grid' }, s);
    text(s, X(t), B0 + 15, t, { 'text-anchor': 'middle' });
  }
  text(s, (L0 + R0) / 2, H - 4, 'CPU-seconds a game, this bot\'s side (log scale)', { 'text-anchor': 'middle' });
  el('line', { x1: L0, x2: L0, y1: T0, y2: B0, class: 'axis' }, s);
  el('line', { x1: L0, x2: R0, y1: B0, y2: B0, class: 'axis' }, s);
  if (mode === 'strength') {
    el('line', { x1: L0, x2: R0, y1: Y(50), y2: Y(50), stroke: '#8a93a0', 'stroke-dasharray': '4 3' }, s);
    text(s, R0 - 2, Y(50) - 4, `${reg.reference} = 50%`, { 'text-anchor': 'end' });
  }
  // the frontier as steps: everything below and right of them is beaten on both counts by a frontier bot
  if (front.length) {
    let d = `M${X(front[0].x)},${B0}L${X(front[0].x)},${Y(front[0].y)}`;
    for (let i = 1; i < front.length; i++) d += `L${X(front[i].x)},${Y(front[i - 1].y)}L${X(front[i].x)},${Y(front[i].y)}`;
    d += `L${R0},${Y(front[front.length - 1].y)}`;
    el('path', { d, fill: 'none', stroke: '#d6b25e', 'stroke-width': 1.6, opacity: 0.9 }, s);
  }
  // dominated first, so the frontier draws on top
  const order = [...pts].sort((a, b) => onFront.has(a.b.id) - onFront.has(b.b.id));
  for (const p of order) {
    const f = onFront.has(p.b.id), col = KIND_COLORS[p.b.kind] || '#9aa5b1';
    const g = el('g', { class: `pareto-pt${f ? ' front' : ''}`, 'data-id': p.b.id, tabindex: 0, opacity: f ? 1 : 0.5 }, s);
    if (p.lo !== null && !p.b.st.ref) el('line', { x1: X(p.x), x2: X(p.x), y1: Y(p.lo), y2: Y(p.hi), stroke: col, 'stroke-width': 1.2 }, g);
    const hollow = mode === 'strength' && !p.b.st.direct && !p.b.st.ref;
    el('circle', { cx: X(p.x), cy: Y(p.y), r: f ? 5.5 : 4.5, fill: hollow ? '#171b21' : col,
      stroke: f ? '#d6b25e' : col, 'stroke-width': f ? 2 : 1.5, class: 'dot' }, g);
    el('circle', { cx: X(p.x), cy: Y(p.y), r: 13, fill: 'transparent' }, g);  // a tap target a finger can hit
  }
  // labels for the frontier and the reference only (the rest are named in their popover), placed greedily where they
  // cross no other label, no point and no frontier step; one moved away from its point gets a thin leader line
  const short = (id) => id.replace(/^search@/, '').replace(/^sp-m3-/, '');
  const segs = [];
  if (front.length) {
    let px = X(front[0].x), py = B0;
    const add = (x, y) => { segs.push({ x0: Math.min(px, x) - 1.5, x1: Math.max(px, x) + 1.5, y0: Math.min(py, y) - 1.5, y1: Math.max(py, y) + 1.5 }); px = x; py = y; };
    add(X(front[0].x), Y(front[0].y));
    for (let i = 1; i < front.length; i++) { add(X(front[i].x), Y(front[i - 1].y)); add(X(front[i].x), Y(front[i].y)); }
    add(R0, Y(front[front.length - 1].y));
  }
  const dots = pts.map((p) => ({ x0: X(p.x) - 6, x1: X(p.x) + 6, y0: Y(p.y) - 6, y1: Y(p.y) + 6 }));
  for (const p of pts) if (p.lo !== null) dots.push({ x0: X(p.x) - 1.5, x1: X(p.x) + 1.5, y0: Y(p.hi), y1: Y(p.lo) });  // error bars
  const boxes = [];
  const hit = (b, o) => b.x0 < o.x1 && b.x1 > o.x0 && b.y0 < o.y1 && b.y1 > o.y0;
  const free = (b) => b.x0 >= L0 + 2 && b.x1 <= R0 && b.y0 >= T0 && b.y1 <= B0 - 2
    && ![...boxes, ...dots, ...segs].some((o) => hit(b, o));
  const labelled = pts.filter((p) => onFront.has(p.b.id) || p.b.id === reg.reference).sort((a, b) => a.x - b.x);
  for (const p of labelled) {
    const name = short(p.b.id), tw = name.length * 6.1 + 4, th = 13, cx = X(p.x), cy = Y(p.y);
    let placed = null;
    for (const d of [7, 18, 30, 44, 60]) {  // further than 60 the leader lines cross the chart; such a label is left to the popover
      // above-left, above-right, below-right, below-left
      for (const [sx, sy] of [[-1, -1], [1, -1], [1, 1], [-1, 1]]) {
        const x0 = sx > 0 ? cx + d : cx - d - tw, y0 = sy > 0 ? cy + d : cy - d - th;
        const b = { x0, x1: x0 + tw, y0, y1: y0 + th };
        if (free(b)) { placed = { b, sx, sy, d }; break; }
      }
      if (placed) break;
    }
    if (!placed) continue;
    const { b, sx, sy, d } = placed;
    boxes.push(b);
    if (d > 7) el('line', { x1: cx + sx * 5, y1: cy + sy * 5, x2: sx > 0 ? b.x0 : b.x1, y2: sy > 0 ? b.y0 + 2 : b.y1 - 2,
      stroke: '#8a93a0', 'stroke-width': 0.8, 'pointer-events': 'none' }, s);
    text(s, b.x0 + 2, b.y1 - 3, name, { class: 'ink', 'pointer-events': 'none' });
  }
  s.addEventListener('click', (e) => {
    const g = e.target.closest('.pareto-pt');
    if (!g) return;
    if (pareto.pinned === g.dataset.id) closePop(); else openPop(byId[g.dataset.id], g.querySelector('.dot'), true);
  });
  s.addEventListener('pointerover', (e) => {
    const g = e.target.closest('.pareto-pt');
    if (g && e.pointerType === 'mouse' && !pareto.pinned) openPop(byId[g.dataset.id], g.querySelector('.dot'), false);
  });
  s.addEventListener('pointerout', (e) => {
    const g = e.target.closest('.pareto-pt');
    if (g && e.pointerType === 'mouse' && !pareto.pinned && !g.contains(e.relatedTarget)) closePop();
  });

  const nStr = bots.filter((b) => b.st).length, nCost = bots.filter((b) => b.cs).length;
  const caption = mode === 'strength'
    ? `<b>y</b>: win rate against ${reg.reference} on the real test decks, with its 95% bar. <b>Filled</b>: played against
      ${reg.reference} itself. <b>Hollow</b>: chained through other bots' head-to-heads (log-odds added, variances summed); the
      more links, the less it means, and gains against a parent do not always carry over.`
    : `<b>y</b>: agreement with people on held-out July 2022 (the same ~36,400 decisions for every bot), order-forgiving measure
      (from 2026-11-01). Bots measured only on the older strict measure, 4-5 points lower, are left off.`;
  body.innerHTML = `<div class="pareto-ctl" role="group" aria-label="y axis">
      <button data-y="strength" class="${mode === 'strength' ? 'on' : ''}">Strength</button>
      <button data-y="july" class="${mode === 'july' ? 'on' : ''}">Plays like people</button></div>
    <div class="legend">${Object.entries(KIND_NAMES).map(([k, n]) => `<span><i style="background:${KIND_COLORS[k]};height:8px;width:8px;border-radius:50%"></i>${n}</span>`).join('')}
      <span><i style="background:#d6b25e;height:2px"></i>Pareto frontier (gold ring: on it)</span></div>`;
  body.appendChild(s);
  const cap = document.createElement('div');
  cap.className = 'caption';
  cap.innerHTML = `<b>x</b>: CPU-seconds a game for this bot's side alone: BotVsBot's whole-game figure less the opponent's side,
    which is half that opponent's game against itself (${reg.reference}: 2.5 a game, 1.25 a side). Measured on different days
    under different load; differences under about 1.5x are not reliable. ${caption} <b>Gold steps</b>: the Pareto frontier --
    no bot is both cheaper and higher. ${pts.length} of ${bots.length} versions are on the chart (${nStr} have a strength,
    ${nCost} a cost a side); tap or hover a point for what it is.`;
  body.appendChild(cap);
  const list = document.createElement('details');
  list.className = 'versions';
  list.open = !!wasOpen;
  const shown = new Set(pts.map((p) => p.b.id));
  const rows = [...bots].sort((a, b) => b.date.localeCompare(a.date) || b.i - a.i).map((b) => {
    const st = b.st ? (b.st.ref ? 'reference' : `${b.st.p.toFixed(1)}%${b.st.direct ? '' : ' chained'}`) : 'strength not measured';
    const c = b.cs ? `${fmtCost(b.cs.v)} s` : 'cost not measured';
    return `<button data-id="${esc(b.id)}" class="${shown.has(b.id) ? '' : 'off'}"><span class="d">${b.date.slice(5)}</span>
      <span class="n">${esc(b.id)}</span><span class="v">${st} · ${c}${b.july ? ` · July ${b.july.value}%` : ''}</span></button>`;
  }).join('');
  list.innerHTML = `<summary>All versions (${bots.length}), newest first <span class="hint">dimmed: not on this chart</span></summary><div class="vlist">${rows}</div>`;
  list.addEventListener('click', (e) => {
    const r = e.target.closest('button[data-id]');
    if (!r) return;
    if (pareto.pinned === r.dataset.id) closePop(); else openPop(byId[r.dataset.id], r, true);
  });
  body.appendChild(list);
  body.querySelector('.pareto-ctl').addEventListener('click', (e) => {
    const bt = e.target.closest('button[data-y]');
    if (bt && bt.dataset.y !== pareto.y) { pareto.y = bt.dataset.y; closePop(); drawPareto(); }
  });
}

async function refresh() {
  try {
    // the registry is hand-kept and small (dashboard/bots.json); a failure there should not blank the rest
    fetch(`bots.json?t=${Date.now()}`).then((r) => r.json()).then(renderPareto)
      .catch((e) => html('pareto', `<div class="empty">could not load bots.json: ${e}</div>`));
    const d = await (await fetch(`data/data.json?t=${Date.now()}`)).json();
    $('updated').innerHTML = `data from <b>${ago(d.updated)}</b><br>${new Date(d.updated * 1000).toLocaleString()}`;
    renderStatus(d); renderAgree(d);
  } catch (e) {
    $('updated').textContent = `could not load data: ${e}`;
  }
}
refresh();
setInterval(refresh, 60000);
