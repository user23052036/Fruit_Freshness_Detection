// ═══════════════════════════════════════════════════════
// State
// ═══════════════════════════════════════════════════════
let DATA = null;
let CUR_STEP = 0;

const VEG_EMOJI = {
  apple:'🍎', banana:'🍌', capsicum:'🫑', cucumber:'🥒', potato:'🥔'
};

const STAGES = [
  {id:'s-preflight', label:'Preflight'},
  {id:'s-features',  label:'Features'},
  {id:'s-veg',       label:'Veg SVM'},
  {id:'s-fresh',     label:'Fresh SVM'},
  {id:'s-centroid',  label:'Centroid'},
  {id:'s-bounds',    label:'Bounds'},
  {id:'s-score',     label:'Score'},
  {id:'s-gate',      label:'Gate'},
];

// ═══════════════════════════════════════════════════════
// Upload handling
// ═══════════════════════════════════════════════════════

function handleFileSelect(e) {
  const f = e.target.files[0];
  if (f) doUpload(f);
}

async function doUpload(file) {
  const reader = new FileReader();
  reader.onload = e => {
    document.getElementById('previewImg').src = e.target.result;
  };
  reader.readAsDataURL(file);

  document.getElementById('loadingOverlay').style.display = 'flex';

  const fd = new FormData();
  fd.append('file', file);

  try {
    const res = await fetch('/predict', {method:'POST', body:fd});
    const json = await res.json();
    if (!res.ok) throw new Error(json.detail || 'Prediction failed');
    DATA = json;
    renderResults(json);
  } catch (err) {
    alert('Error: ' + err.message);
  } finally {
    document.getElementById('loadingOverlay').style.display = 'none';
  }
}

// ═══════════════════════════════════════════════════════
// Render result card
// ═══════════════════════════════════════════════════════

function renderResults(d) {
  document.getElementById('uploadSection').style.display = 'none';
  document.getElementById('resultSection').style.display = 'block';
  document.getElementById('tryAgainBtn').style.display = 'block';

  const em = VEG_EMOJI[d.veg] || '🥦';
  document.getElementById('vegBadge').textContent = `${em} ${d.veg || '—'}`;
  if (d.veg) {
    document.getElementById('confLine').textContent =
      `conf = ${d.veg_conf.toFixed(1)}%   gap = ${d.conf_gap.toFixed(1)}%   · ${d.norm_source} bounds`;
  }

  const sb = document.getElementById('stateBadge');
  sb.textContent = d.state;
  sb.className = `state-badge s-${d.state}`;

  if (d.score != null) {
    animateGauge(d.score);
    document.getElementById('gaugeSub').textContent =
      d.freshness_confidence_band
        ? `Confidence Band: ${d.freshness_confidence_band}`
        : d.state === 'TENTATIVE' ? 'Score valid · label withheld' : '';
  } else {
    document.getElementById('gaugeNum').textContent = '—';
  }

  const rows = document.getElementById('scoreRows');
  rows.innerHTML = '';
  const addRow = (k, v, cls) => {
    if (v == null) return;
    rows.innerHTML +=
      `<div class="rrow"><span class="rkey">${k}</span>` +
      `<span class="rval ${cls||''}">${v}</span></div>`;
  };

  if (d.fresh_label) addRow('Freshness', d.fresh_label, `c-${d.fresh_label}`);
  if (d.freshness_confidence_band) {
    const bc = 'c-' + d.freshness_confidence_band.replace(' ','');
    addRow('Confidence Band', d.freshness_confidence_band, bc);
  }
  if (d.score != null) addRow('Score', `${d.score.toFixed(2)} / 100`, '');
  if (d.raw != null)   addRow('Raw (SVM margin)', d.raw, '');
  addRow(
    'Mahal Zone',
    `${d.mahal_dist}  [${d.mahal_zone}]`,
    `c-${d.mahal_zone}`
  );

  const ws = document.getElementById('warningsSection');
  ws.innerHTML = '';
  (d.warnings || []).forEach(w => {
    ws.innerHTML += `<div class="warn-card">${w}</div>`;
  });

  if (d.pipeline) buildPipeline(d);
}

// ═══════════════════════════════════════════════════════
// Gauge animation
// ═══════════════════════════════════════════════════════

function animateGauge(score) {
  const arc    = document.getElementById('gaugeArc');
  const num    = document.getElementById('gaugeNum');
  const needle = document.getElementById('gaugeNeedle');

  let color = '#ef5350';
  if (score >= 65) color = '#00d4aa';
  else if (score >= 40) color = '#ffb74d';
  arc.style.stroke = color;

  const filled = (score / 100) * 251;
  arc.style.strokeDashoffset = 251 - filled;

  const deg = (score / 100) * 180 - 90;
  needle.setAttribute('transform', `translate(105,100) rotate(${deg})`);

  let cur = 0;
  const step = score / 45;
  const t = setInterval(() => {
    cur = Math.min(cur + step, score);
    num.textContent = cur.toFixed(1);
    if (cur >= score) { clearInterval(t); num.textContent = score.toFixed(1); }
  }, 22);
}

// ═══════════════════════════════════════════════════════
// Toggle pipeline
// ═══════════════════════════════════════════════════════

function togglePipeline() {
  const sec = document.getElementById('pipelineSection');
  const btn = document.getElementById('pipelineToggle');
  const lbl = document.getElementById('toggleLabel');
  const arr = document.getElementById('toggleArrow');

  const open = sec.style.display === 'block';
  sec.style.display = open ? 'none' : 'block';
  btn.classList.toggle('active', !open);
  lbl.textContent = open ? 'Show Pipeline' : 'Hide Pipeline';
  arr.textContent = open ? '▼' : '▲';
  if (!open) sec.scrollIntoView({behavior:'smooth', block:'nearest'});
}

// ═══════════════════════════════════════════════════════
// Build pipeline stage HTML
// ═══════════════════════════════════════════════════════

function buildPipeline(d) {
  const p = d.pipeline;
  const sc = document.getElementById('stepContent');
  sc.innerHTML = '';

  sc.innerHTML += stagePreflight(p.preflight);
  sc.innerHTML += stageFeatures(p.features);
  sc.innerHTML += stageVegSVM(d, p);
  sc.innerHTML += stageFreshSVM(d, p);
  sc.innerHTML += stageCentroid(d, p);
  sc.innerHTML += stageBounds(d, p);
  sc.innerHTML += stageScore(p.score_norm);
  sc.innerHTML += stageGate(d, p);

  buildTracker();
  CUR_STEP = 0;
  refreshStepView();
}

// ── Helper: single check row ───────────────────────────
function chk(icon, label, val, hint, badgeText, badgeClass) {
  return `
  <div class="chk">
    <span class="chk-icon">${icon}</span>
    <span class="chk-label">${label}</span>
    <span class="chk-val">${val}</span>
    <span class="chk-hint">${hint}</span>
    <span class="chk-badge ${badgeClass}">${badgeText}</span>
  </div>`;
}

// ── Stage 1: Preflight ─────────────────────────────────
function stagePreflight(pf) {
  const blur_ok  = pf.blur_pass;
  const bri_ok   = pf.brightness_pass;
  const cov_warn = pf.coverage_warn;

  return `
  <div class="step-card" id="s-preflight">
    <div class="step-title">Pre-flight Checks <span class="step-tag">stage 01</span></div>
    <div class="step-desc">Image quality gating before any ML inference. Blur and brightness are hard rejects. Coverage is a soft warning only — Otsu thresholding is unreliable on varied backgrounds.</div>
    ${chk('🔍','Focus (Laplacian)',
      `lap_var = ${pf.blur_val}`,
      `threshold ≥ ${pf.blur_thresh}`,
      blur_ok ? 'PASS' : 'FAIL',
      blur_ok ? 'b-pass' : 'b-fail')}
    ${chk('☀️','Brightness (mean px)',
      `${pf.brightness_val}`,
      `range: [${pf.brightness_min}, ${pf.brightness_max}]`,
      bri_ok ? 'PASS' : 'FAIL',
      bri_ok ? 'b-pass' : 'b-fail')}
    ${chk('🖼️','Object coverage',
      `${(pf.coverage_val*100).toFixed(1)}%`,
      `min ${(pf.coverage_min*100).toFixed(0)}%  (soft warn, not reject)`,
      cov_warn ? 'LOW COV' : 'OK',
      cov_warn ? 'b-warn' : 'b-pass')}
  </div>`;
}

// ── Stage 2: Features ──────────────────────────────────
function stageFeatures(fi) {
  return `
  <div class="step-card" id="s-features">
    <div class="step-title">Feature Extraction &amp; Preprocessing <span class="step-tag">stage 02</span></div>
    <div class="step-desc">EfficientNetB0 (ImageNet weights, no top layer) produces 1280 deep features. 32 handcrafted features (colour stats, edge density, Laplacian sharpness, luminance histogram) are appended. Three preprocessing steps reduce to 349 features — the union of top-200 by freshness-task ranking and top-200 by vegetable-task ranking, each averaged over 5 XGBoost seeds.</div>
    <div class="metric-row">
      <div class="metric-box">
        <div class="metric-label">EfficientNetB0</div>
        <div class="metric-num" style="color:var(--accent2)">${fi.deep_dims}</div>
        <div class="metric-sub">deep features</div>
      </div>
      <div class="metric-box" style="flex:0;display:flex;align-items:center;padding:0 4px;border:none;background:none">
        <span style="font-size:20px;color:var(--dim)">+</span>
      </div>
      <div class="metric-box">
        <div class="metric-label">Handcrafted</div>
        <div class="metric-num" style="color:var(--warn)">${fi.hand_dims}</div>
        <div class="metric-sub">colour / texture</div>
      </div>
      <div class="metric-box" style="flex:0;display:flex;align-items:center;padding:0 4px;border:none;background:none">
        <span style="font-size:20px;color:var(--dim)">=</span>
      </div>
      <div class="metric-box">
        <div class="metric-label">Total</div>
        <div class="metric-num" style="color:#fff">${fi.total}</div>
        <div class="metric-sub">raw features</div>
      </div>
    </div>
    ${chk('⚡','VarianceThreshold',`${fi.total} → ${fi.after_vt}`,'removes zero-variance features','DONE','b-ok')}
    ${chk('📊','StandardScaler','mean = 0,  std = 1','fit on training data only — no leakage','DONE','b-ok')}
    ${chk('🎯','XGBoost union selection',`${fi.after_vt} → ${fi.after_sel}`,'dual-task: top-200 fresh ∪ top-200 veg, 5-seed avg each','DONE','b-ok')}
  </div>`;
}

// ── Stage 3: Veg SVM ───────────────────────────────────
function stageVegSVM(d, p) {
  const sorted = Object.entries(p.veg_probs).sort((a,b)=>b[1]-a[1]);
  let bars = '';
  sorted.forEach(([veg, prob]) => {
    const top = veg === d.veg;
    bars += `
    <div class="prob-row">
      <div class="prob-hdr">
        <span style="color:${top?'var(--accent)':'var(--text)'}">${VEG_EMOJI[veg]||'🥦'} ${veg}</span>
        <span style="color:${top?'var(--accent)':'var(--dim)'}">${prob.toFixed(1)}%</span>
      </div>
      <div class="prob-bg">
        <div class="prob-fill ${top?'prob-active':'prob-passive'}"
             style="width:${Math.max(prob,.5)}%">${top?'▶':''}</div>
      </div>
    </div>`;
  });

  return `
  <div class="step-card" id="s-veg">
    <div class="step-title">Vegetable Classifier (SVM) <span class="step-tag">stage 03</span></div>
    <div class="step-desc">RBF-kernel SVM with isotonic probability calibration (CalibratedClassifierCV wrapping FrozenEstimator — SVC weights are frozen, only the isotonic calibration layer is fit on cal_val). Returns calibrated probabilities for all 5 classes. Both top-1 confidence AND the gap to second-best must exceed thresholds to use per-vegetable normalization bounds.</div>
    ${bars}
    <div style="margin-top:14px">
      ${chk('📊','Top-1 confidence',
        `${d.veg_conf.toFixed(1)}%`,
        `threshold ≥ ${p.veg_conf_thresh.toFixed(0)}%`,
        d.veg_conf >= p.veg_conf_thresh ? 'PASS' : 'BELOW THRESH',
        d.veg_conf >= p.veg_conf_thresh ? 'b-pass' : 'b-warn')}
      ${chk('📐','Confidence gap (top1 − top2)',
        `${d.conf_gap.toFixed(1)}%`,
        `threshold ≥ ${p.veg_gap_thresh.toFixed(0)}%`,
        d.conf_gap >= p.veg_gap_thresh ? 'PASS' : 'BELOW THRESH',
        d.conf_gap >= p.veg_gap_thresh ? 'b-pass' : 'b-warn')}
      ${chk('🔓','Per-veg bounds eligibility',
        p.veg_confident ? 'both checks passed' : 'one or more failed',
        'both conf + gap must pass',
        p.veg_confident ? 'ELIGIBLE' : 'FALLBACK',
        p.veg_confident ? 'b-pass' : 'b-warn')}
    </div>
  </div>`;
}

// ── Stage 4: Freshness SVM ─────────────────────────────
function stageFreshSVM(d, p) {
  const raw = p.freshness_raw;
  const pct = Math.min(Math.max((raw + 4) / 8 * 90 + 5, 5), 95);
  const dir = raw > 0 ? 'Fresh' : 'Rotten';

  return `
  <div class="step-card" id="s-fresh">
    <div class="step-title">Freshness Classifier (SVM) <span class="step-tag">stage 04</span></div>
    <div class="step-desc">RBF-kernel SVM without probability=True. Uses decision_function() — the signed distance from the separating hyperplane. Positive = fresh side, negative = rotten side. The magnitude indicates confidence.</div>
    <div class="nl-wrap">
      <div class="nl-rotten"></div>
      <div class="nl-fresh"></div>
      <div class="nl-track"></div>
      <div class="nl-zero"></div>
      <div class="nl-needle" id="nlNeedle" style="left:${pct}%"></div>
    </div>
    <div class="nl-labels">
      <span>← Rotten   −4</span>
      <span style="color:var(--dim)">boundary = 0</span>
      <span>+4   Fresh →</span>
    </div>
    <div style="margin-top:14px">
      ${chk(raw>0?'🟢':'🔴','Raw decision value',
        raw.toFixed(4),
        'positive = fresh side of hyperplane',
        dir, raw>0 ? 'b-pass' : 'b-fail')}
      ${chk('📏','Distance from boundary',
        Math.abs(raw).toFixed(4),
        p.boundary_thresh === 0
          ? `T_boundary = 0.0 — boundary gate inactive (formally selected; no margin cutoff needed)`
          : `min reliable: ${p.boundary_thresh}  (near-boundary → TENTATIVE)`,
        p.boundary_thresh === 0
          ? 'INACTIVE'
          : Math.abs(raw) >= p.boundary_thresh ? 'CLEAR' : 'NEAR BOUNDARY',
        p.boundary_thresh === 0
          ? 'b-info'
          : Math.abs(raw) >= p.boundary_thresh ? 'b-pass' : 'b-warn')}
    </div>
  </div>`;
}

// ── Stage 5: Centroid Check ────────────────────────────
function stageCentroid(d, p) {
  const c = p.centroid;
  return `
  <div class="step-card" id="s-centroid">
    <div class="step-title">Centroid Consistency Check <span class="step-tag">stage 05</span></div>
    <div class="step-desc">Runs before bounds selection. Computes ratio of (distance to predicted class centroid) ÷ (distance to second-closest centroid). A high ratio means the sample is not clearly inside the predicted cluster, even at high SVM confidence. Prevents wrong-vegetable bounds being applied silently to a misclassified sample.</div>
    ${chk('🎯','Centroid ratio  (d_pred / d_second)',
      `${c.ratio.toFixed(4)}`,
      `threshold: ${c.threshold.toFixed(4)}  ·  d_pred=${c.d_pred}  d_second=${c.d_second}`,
      c.consistent ? 'CONSISTENT' : 'INCONSISTENT',
      c.consistent ? 'b-pass' : 'b-fail')}
    ${chk('🔒','Effect on bounds',
      c.consistent ? 'centroid OK — eligible for per-veg' : 'centroid fail → forced global',
      'class_inconsistent forces global bounds regardless of veg_conf',
      c.consistent ? 'NO OVERRIDE' : 'FORCED GLOBAL',
      c.consistent ? 'b-ok' : 'b-warn')}
  </div>`;
}

// ── Stage 6: Bounds Selection ──────────────────────────
function stageBounds(d, p) {
  const b = p.bounds;
  const spread = (b.p95 - b.p5).toFixed(4);
  const srcCol = b.source === 'per-veg' ? 'var(--accent)' : 'var(--warn)';
  return `
  <div class="step-card" id="s-bounds">
    <div class="step-title">Normalization Bounds Selection <span class="step-tag">stage 06</span></div>
    <div class="step-desc">p5 and p95 percentiles of validation-set decision_function values anchor the [0,100] score scale. Computed on the FULL val set (not cal_val or thr_val) — the 50/50 cal/thr split can leave individual vegetable classes below the 50-sample minimum, causing unstable percentile estimates. Per-vegetable bounds require veg_confident AND centroid-consistent; otherwise global bounds are used.</div>
    <div class="chk" style="margin-bottom:8px;border-color:${srcCol}">
      <span class="chk-icon">📦</span>
      <span class="chk-label">Bounds source</span>
      <span class="chk-val" style="color:${srcCol}">
        ${b.source === 'per-veg' ? `per-veg  (${b.veg_name})` : 'global fallback'}
      </span>
      <span class="chk-hint">veg_confident=${p.veg_confident} · centroid_consistent=${p.centroid.consistent}</span>
      <span class="chk-badge ${b.source==='per-veg'?'b-pass':'b-warn'}">${b.source.toUpperCase()}</span>
    </div>
    ${chk('📉','p5  (5th percentile)',b.p5.toString(),`validation-set anchor for score = 0  ·  spread = ${spread}`,'VAL-ANCHORED','b-info')}
    ${chk('📈','p95 (95th percentile)',b.p95.toString(),'validation-set anchor for score = 100','VAL-ANCHORED','b-info')}
  </div>`;
}

// ── Stage 7: Score Normalization ───────────────────────
function stageScore(sn) {
  let band='Very Low', bc='var(--danger)';
  if (sn.score >= 85) { band='High';   bc='var(--ok)';     }
  else if (sn.score >= 65) { band='Medium'; bc='var(--accent)'; }
  else if (sn.score >= 40) { band='Low';    bc='var(--warn)';   }

  const bands = [
    {lo:0,  hi:40,  c:'var(--danger)'},
    {lo:40, hi:65,  c:'var(--warn)'},
    {lo:65, hi:85,  c:'var(--accent)'},
    {lo:85, hi:100, c:'var(--ok)'},
  ];
  const strip = bands.map(b => {
    const active = sn.score >= b.lo && sn.score < b.hi;
    return `<div class="bs" style="flex:${b.hi-b.lo};background:${active?b.c:'rgba(255,255,255,0.06)'}"></div>`;
  }).join('');

  return `
  <div class="step-card" id="s-score">
    <div class="step-title">Score Normalization <span class="step-tag">stage 07</span></div>
    <div class="step-desc">Maps raw SVM margin to [0, 100] using validation-anchored p5/p95. Scores are locally comparable within a vegetable class — not globally across vegetables (a banana 80 ≠ potato 80 in absolute freshness terms).</div>
    <div class="formula">
      <div><span class="f-k">score  =  clip( (raw − p5) / (p95 − p5) × 100,  0,  100 )</span></div>
      <div><span class="f-k">       =  clip( (</span><span class="f-v">${sn.raw}</span><span class="f-k"> − (</span><span class="f-v">${sn.p5}</span><span class="f-k">)) / (</span><span class="f-v">${sn.p95}</span><span class="f-k"> − (</span><span class="f-v">${sn.p5}</span><span class="f-k">)) × 100,  0,  100 )</span></div>
      <div><span class="f-k">       =  </span><span class="f-r">${sn.score.toFixed(2)}</span><span class="f-k">  / 100</span></div>
    </div>
    <div style="margin-top:16px;padding:14px;background:rgba(0,0,0,.2);border:1px solid var(--border);border-radius:8px">
      <div style="display:flex;align-items:center;gap:12px;margin-bottom:10px">
        <div style="font-size:32px;font-weight:800;color:${bc}">${sn.score.toFixed(1)}</div>
        <div>
          <div style="font-size:15px;font-weight:700;color:${bc}">${band} Confidence</div>
          <div style="font-size:10px;color:var(--dim)">≥85 High · 65 Medium · 40 Low · &lt;40 Very Low</div>
        </div>
      </div>
      <div class="band-strip">${strip}</div>
    </div>
  </div>`;
}

// ── Stage 8: Gate & Output ─────────────────────────────
function stageGate(d, p) {
  const stateCol = d.state==='RELIABLE' ? 'var(--ok)'
                 : d.state==='TENTATIVE' ? 'var(--warn)'
                 : 'var(--danger)';

  let gateRows = '';
  p.gates.forEach(g => {
    gateRows += `
    <div class="gate-row ${g.fired?'fired':'clear'}">
      <span class="gate-icon">${g.fired?'🔴':'🟢'}</span>
      <span class="gate-name">${g.name}</span>
      <span class="gate-reason">${g.reason}</span>
      <span class="gate-status" style="color:${g.fired?'var(--danger)':'var(--ok)'}">
        ${g.fired ? 'FIRED' : 'CLEAR'}
      </span>
    </div>`;
  });

  if (p.high_conf_override) {
    gateRows += `
    <div class="gate-row clear" style="border-color:var(--accent2)">
      <span class="gate-icon">⚡</span>
      <span class="gate-name" style="color:var(--accent2)">High-conf override</span>
      <span class="gate-reason">veg_conf &gt; 95%, not near boundary, not OOD, centroid OK</span>
      <span class="gate-status" style="color:var(--accent2)">ACTIVE</span>
    </div>`;
  }

  let outCells = `
  <div class="out-cell">
    <div class="oc-label">State</div>
    <div class="oc-val" style="color:${stateCol}">${d.state}</div>
  </div>
  <div class="out-cell">
    <div class="oc-label">Score</div>
    <div class="oc-val">${d.score != null ? d.score.toFixed(1) : '—'}</div>
  </div>`;

  if (d.fresh_label) {
    const fc = d.fresh_label === 'Fresh' ? 'var(--ok)' : 'var(--danger)';
    outCells += `
    <div class="out-cell">
      <div class="oc-label">Freshness</div>
      <div class="oc-val" style="color:${fc}">${d.fresh_label === 'Fresh' ? '🟢' : '🔴'} ${d.fresh_label}</div>
    </div>`;
  }
  if (d.freshness_confidence_band) {
    outCells += `
    <div class="out-cell">
      <div class="oc-label">Band</div>
      <div class="oc-val" style="color:var(--accent)">${d.freshness_confidence_band}</div>
    </div>`;
  }

  const mbar = buildMahalBar(p.mahal);

  return `
  <div class="step-card" id="s-gate">
    <div class="step-title">Reliability Gate &amp; Output <span class="step-tag">stage 08</span></div>
    <div class="step-desc">Two-level gate. Level 1 (score validity): OOD → UNRELIABLE. Augmentation instability gate is currently disabled (use_augmentation_gate=False); T_instability=36.0 is stored for future activation. Level 2 (decision validity): near boundary / low veg confidence / centroid inconsistency → TENTATIVE. High-confidence override forces RELIABLE when veg_conf &gt; 95%, not near boundary, not OOD, and centroid is consistent.</div>

    <div style="margin-bottom:6px;font-size:10px;color:var(--dim);text-transform:uppercase;letter-spacing:1px">OOD Detection</div>
    ${mbar}
    <div style="margin-top:14px;margin-bottom:6px;font-size:10px;color:var(--dim);text-transform:uppercase;letter-spacing:1px">Gate checks</div>
    ${gateRows}
    <div class="out-box" style="border-color:${stateCol}">
      <div style="font-size:10px;color:var(--dim);margin-bottom:10px;text-transform:uppercase;letter-spacing:1px">Final output</div>
      <div class="out-grid">${outCells}</div>
    </div>
  </div>`;
}

function buildMahalBar(m) {
  const cap = m.thresh_ood * 1.35;
  const cautPct  = (m.thresh_caution / cap * 100).toFixed(1);
  const oodPct   = (m.thresh_ood     / cap * 100).toFixed(1);
  const distPct  = Math.min((m.dist / cap * 100), 100).toFixed(1);
  const needleCol= m.zone==='ood'     ? 'var(--danger)'
                 : m.zone==='caution' ? 'var(--warn)'
                 : 'var(--ok)';
  return `
  <div class="mbar-wrap">
    <div class="mbar-caution" style="left:${cautPct}%;right:${(100-oodPct).toFixed(1)}%"></div>
    <div class="mbar-ood"     style="left:${oodPct}%;right:0"></div>
    <div class="mbar-needle"  style="left:${distPct}%;background:${needleCol};box-shadow:0 0 8px ${needleCol}"></div>
  </div>
  <div style="display:flex;justify-content:space-between;font-size:9px;color:var(--dim)">
    <span>0  [trusted]</span>
    <span>${m.thresh_caution}  [caution]</span>
    <span>${m.thresh_ood}  [ood]</span>
  </div>
  <div style="margin-top:6px">
    ${chk('📡','Mahalanobis dist',m.dist.toString(),`caution=${m.thresh_caution}  ·  ood=${m.thresh_ood}`,m.zone.toUpperCase(),m.zone==='trusted'?'b-pass':m.zone==='caution'?'b-warn':'b-fail')}
  </div>`;
}

// ═══════════════════════════════════════════════════════
// Step tracker
// ═══════════════════════════════════════════════════════

function buildTracker() {
  const track = document.getElementById('stepTrack');
  track.innerHTML = '';
  STAGES.forEach((s, i) => {
    track.innerHTML +=
      `<div class="step-dot" id="dot${i}" onclick="goToStep(${i})">
         <div class="dot">${i+1}</div>
         <div class="slabel">${s.label}</div>
       </div>`;
    if (i < STAGES.length - 1)
      track.innerHTML += `<div class="step-conn" id="conn${i}"></div>`;
  });
}

function refreshStepView() {
  STAGES.forEach((s, i) => {
    const el = document.getElementById(s.id);
    if (el) el.classList.toggle('visible', i === CUR_STEP);

    const dot = document.getElementById(`dot${i}`);
    if (dot) dot.className =
      'step-dot' +
      (i === CUR_STEP ? ' active' : '') +
      (i < CUR_STEP  ? ' done'   : '');

    const conn = document.getElementById(`conn${i}`);
    if (conn) conn.classList.toggle('done', i < CUR_STEP);
  });

  document.getElementById('stepCtr').textContent =
    `${CUR_STEP + 1} / ${STAGES.length}`;
  document.getElementById('prevBtn').disabled = CUR_STEP === 0;
  document.getElementById('nextBtn').disabled = CUR_STEP === STAGES.length - 1;
}

function changeStep(dir) {
  const n = CUR_STEP + dir;
  if (n >= 0 && n < STAGES.length) { CUR_STEP = n; refreshStepView(); }
}

function goToStep(n) { CUR_STEP = n; refreshStepView(); }

// ═══════════════════════════════════════════════════════
// Reset
// ═══════════════════════════════════════════════════════

function resetApp() {
  document.getElementById('uploadSection').style.display  = 'block';
  document.getElementById('resultSection').style.display  = 'none';
  document.getElementById('pipelineSection').style.display = 'none';
  document.getElementById('tryAgainBtn').style.display    = 'none';
  document.getElementById('pipelineToggle').classList.remove('active');
  document.getElementById('toggleLabel').textContent = 'Show Pipeline';
  document.getElementById('toggleArrow').textContent = '▼';
  document.getElementById('fileInput').value = '';
  DATA = null;
}

// ═══════════════════════════════════════════════════════
// Drag & drop
// ═══════════════════════════════════════════════════════

const dz = document.getElementById('dropZone');
dz.addEventListener('dragover',  e => { e.preventDefault(); dz.classList.add('drag-over'); });
dz.addEventListener('dragleave', ()  => dz.classList.remove('drag-over'));
dz.addEventListener('drop', e => {
  e.preventDefault();
  dz.classList.remove('drag-over');
  const f = e.dataTransfer.files[0];
  if (f) doUpload(f);
});