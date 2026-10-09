/* Bearing Early Fault Warning: static replay of the IMS run-to-failure tests.
 * Data comes from data/*.json, written by scripts/export_web_dashboard.py.
 * Mirrors the behaviour of app/dashboard.py (the Streamlit version). */
(function () {
  "use strict";

  var PLOTLY_FALLBACK = "https://cdn.jsdelivr.net/npm/plotly.js-dist-min@2.35.2/plotly.min.js";
  var BEARING_COLORS = { 1: "#2a78d6", 2: "#eb6834", 3: "#1baf7a", 4: "#eda100" };
  var DEFECT_COLORS = { bpfo: "#e34948", bpfi: "#4a3aa7", bsf: "#1baf7a" };
  var STATUS = {
    ok: { color: "#0f8a3c", icon: "●", label: "Healthy" },
    watch: { color: "#c98500", icon: "▲", label: "Above threshold" },
    alarm: { color: "#e34948", icon: "■", label: "ALARM" }
  };
  var LOG_TICKS = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000];
  var FRAME_MS = 100; // ~10 fps for playback and chart redraws
  var MUTED = "#52514e";
  var GRID = "#ebeae6";

  var $ = function (id) { return document.getElementById(id); };
  var el = {
    test: $("test"), detector: $("detector"), speed: $("speed"), play: $("play"), restart: $("restart"),
    pos: $("pos"), reveal: $("reveal"), readout: $("readout"), tiles: $("tiles"), score: $("scoreChart"),
    env: $("envChart"), bearingSel: $("bearingSel"), log: $("log"), truth: $("truth"), footnote: $("footnote")
  };

  var S = {
    index: null, cache: {}, data: null, test: null, detector: null, idx: 1, playing: false,
    reveal: false, revealTouched: false, bSel: 1, logKey: "", lastChart: 0, chartTimer: null,
    rafPending: false, lastTick: 0
  };

  // ------------------------------------------------------------------ helpers
  function fmt(x, d) {
    if (x === null || x === undefined || !isFinite(x)) return "n/a";
    return x.toLocaleString("en-US", { minimumFractionDigits: d, maximumFractionDigits: d });
  }
  function pad(n) { return (n < 10 ? "0" : "") + n; }
  function stamp(D, i, withSeconds) {
    var t = new Date(D.startMs + D.seconds[i] * 1000);
    var s = t.getUTCFullYear() + "-" + pad(t.getUTCMonth() + 1) + "-" + pad(t.getUTCDate()) + " " +
      pad(t.getUTCHours()) + ":" + pad(t.getUTCMinutes());
    return withSeconds ? s + ":" + pad(t.getUTCSeconds()) : s;
  }
  function esc(s) {
    return String(s).replace(/[&<>"']/g, function (c) {
      return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c];
    });
  }
  function bearings(D) { return Object.keys(D.z).map(Number).sort(function (a, b) { return a - b; }); }

  function loadScript(src) {
    return new Promise(function (resolve, reject) {
      var s = document.createElement("script");
      s.src = src; s.onload = resolve; s.onerror = function () { reject(new Error("could not load " + src)); };
      document.head.appendChild(s);
    });
  }
  function ensurePlotly() { return window.Plotly ? Promise.resolve() : loadScript(PLOTLY_FALLBACK); }

  function getJSON(url) {
    return fetch(url).then(function (r) {
      if (!r.ok) throw new Error(url + ": HTTP " + r.status);
      return r.json();
    });
  }

  // Long tests are drawn with min/max decimation over fixed buckets of snapshots so each
  // frame stays cheap; spikes survive because both extremes of every bucket are kept.
  var MAX_BUCKETS = 800;
  function decimate(x, y, size) {
    var ox = [], oy = [];
    for (var b0 = 0; b0 + size <= y.length; b0 += size) {
      var lo = -1, hi = -1;
      for (var j = b0; j < b0 + size; j++) {
        var v = y[j];
        if (v === null) continue;
        if (lo < 0 || v < y[lo]) lo = j;
        if (hi < 0 || v > y[hi]) hi = j;
      }
      if (lo < 0) { ox.push(x[b0], x[b0]); oy.push(null, null); continue; }
      var a = Math.min(lo, hi), c = Math.max(lo, hi);
      ox.push(x[a], x[c]); oy.push(y[a], y[c]);
    }
    return { x: ox, y: oy };
  }
  // Points of a series up to snapshot index m (exclusive): complete buckets + raw tail.
  function series(D, y, dec, m) {
    if (!dec) return { x: D.hours.slice(0, m), y: y.slice(0, m) };
    var k = Math.floor(m / D.bucket), cut = k * D.bucket;
    return { x: dec.x.slice(0, 2 * k).concat(D.hours.slice(cut, m)), y: dec.y.slice(0, 2 * k).concat(y.slice(cut, m)) };
  }

  // Derive everything the page needs per frame once, when a test is loaded.
  function preprocess(D) {
    D.startMs = Date.parse(D.start + "Z");
    D.hours = D.seconds.map(function (s) { return s / 3600; });
    D.endH = D.hours[D.n - 1];
    D.bucket = D.n > 2 * MAX_BUCKETS ? Math.ceil(D.n / MAX_BUCKETS) : 1;
    var dec = function (y) { return D.bucket > 1 ? decimate(D.hours, y, D.bucket) : null; };
    var bs = bearings(D);
    D.zDec = {};

    // 6-snapshot rolling mean of the envelope z-scores (causal, min_periods=1)
    D.zRoll = {};
    bs.forEach(function (b) {
      D.zRoll[b] = {};
      Object.keys(D.z[b]).forEach(function (k) {
        var a = D.z[b][k], out = new Array(a.length), sum = 0;
        for (var i = 0; i < a.length; i++) {
          sum += a[i];
          if (i >= 6) sum -= a[i - 6];
          out[i] = sum / Math.min(i + 1, 6);
        }
        D.zRoll[b][k] = out;
        D.zDec[b] = D.zDec[b] || {};
        D.zDec[b][k] = dec(out);
      });
    });

    Object.keys(D.detectors).forEach(function (det) {
      var Dd = D.detectors[det];
      Dd.events = [];
      bs.forEach(function (b) {
        var B = Dd.bearings[b];
        var n = B.alarm.length, al = new Uint8Array(n), ep = new Int32Array(n);
        B.firstAlarm = -1;
        for (var i = 0; i < n; i++) {
          al[i] = B.alarm.charCodeAt(i) === 49 ? 1 : 0;
          if (al[i]) {
            ep[i] = i > 0 && al[i - 1] ? ep[i - 1] : i;
            if (ep[i] === i) Dd.events.push({ b: b, i: i });
            if (B.firstAlarm < 0) B.firstAlarm = i;
          } else ep[i] = -1;
        }
        B.al = al; B.ep = ep;
        B.dec = dec(B.ratio);
      });
      Dd.events.sort(function (x, y) { return x.i - y.i || x.b - y.b; });
    });
    return D;
  }

  function loadTest(name) {
    if (S.cache[name]) return Promise.resolve(S.cache[name]);
    var meta = S.index.tests.filter(function (t) { return t.name === name; })[0];
    return getJSON(meta.file).then(function (D) { S.cache[name] = preprocess(D); return D; });
  }

  // Envelope diagnosis: defect with the highest mean z over the last 6 snapshots seen.
  function diagnose(D, b, i) {
    var best = null, bestV = -Infinity;
    S.index.defects.forEach(function (d) {
      var a = D.z[b][d.key], lo = Math.max(0, i - 5), s = 0;
      for (var j = lo; j <= i; j++) s += a[j];
      var v = s / (i - lo + 1);
      if (v > bestV) { bestV = v; best = d; }
    });
    return { label: best.label, z: bestV };
  }

  // ------------------------------------------------------------------ rendering
  function renderReadout(D, i) {
    el.readout.innerHTML = "<b>" + esc(D.test) + "</b> · detector <b>" + esc(S.detector) + "</b> · snapshot " +
      (i + 1).toLocaleString("en-US") + "/" + D.n.toLocaleString("en-US") + " · <b>" + fmt(D.hours[i], 1) +
      " h</b> since start (" + stamp(D, i, false) + ")" + (i < D.calib_end ? " · <i>before monitoring starts</i>" : "");
  }

  function renderTiles(D, i) {
    var Dd = D.detectors[S.detector], bs = bearings(D);
    // The first bearing to raise an alarm is the likely source: once it degrades, its
    // vibration travels through the shaft and housing and lifts its neighbours too.
    var source = null, srcI = Infinity;
    bs.forEach(function (b) {
      var f = Dd.bearings[b].firstAlarm;
      if (f >= 0 && f <= i && f < srcI) { srcI = f; source = b; }
    });

    var html = bs.map(function (b) {
      var B = Dd.bearings[b], ratio = B.ratio[i], alarm = B.al[i] === 1;
      var state = alarm ? "alarm" : (ratio !== null && ratio > 1 ? "watch" : "ok");
      var st = STATUS[state];
      var since = alarm ? " since " + fmt(D.hours[B.ep[i]], 1) + " h" : "";
      var diag = "";
      if (alarm) {
        var dg = diagnose(D, b, i);
        diag = dg.z > 3 ? "likely <b>" + esc(dg.label) + "</b> (z=" + fmt(dg.z, 1) + ")" : "no clear defect signature";
      }
      var gt = "";
      if (S.reveal) {
        var f = D.failed[b];
        gt = '<div class="gt' + (f ? " failed" : "") + '">ground truth: ' + esc(f || "survived") + "</div>";
      }
      return '<div class="tile" style="border-left-color:' + st.color + '">' +
        '<div class="who"><span class="dot" style="background:' + BEARING_COLORS[b] + '"></span>Bearing ' + b +
        (b === source ? ' <span class="badge" title="First bearing to raise an alarm: the likely source of the fault">first to alarm</span>' : "") +
        "</div>" +
        '<div class="state" style="color:' + st.color + '">' + st.icon + " " + st.label + "</div>" +
        '<div class="line">score / threshold <b>' + fmt(ratio, 2) + "</b>" + since + "</div>" +
        '<div class="diag">' + diag + "</div>" + gt + "</div>";
    }).join("");
    el.tiles.innerHTML = html;
  }

  function renderLog(D, i) {
    var Dd = D.detectors[S.detector], evs = [];
    for (var k = 0; k < Dd.events.length && Dd.events[k].i <= i; k++) evs.push(Dd.events[k]);
    var key = D.test + "|" + S.detector + "|" + evs.length;
    if (key === S.logKey) return;
    S.logKey = key;
    if (!evs.length) { el.log.innerHTML = '<div class="empty">No alarms yet.</div>'; return; }
    el.log.innerHTML = '<table><thead><tr><th>Bearing</th><th class="num">Raised at</th><th>Time</th></tr></thead><tbody>' +
      evs.map(function (e) {
        return '<tr><td><span class="dot" style="background:' + BEARING_COLORS[e.b] + '"></span>' + e.b + "</td>" +
          '<td class="num">' + fmt(D.hours[e.i], 1) + " h</td><td>" + stamp(D, e.i, true) + "</td></tr>";
      }).join("") + "</tbody></table>";
  }

  function renderTruth(D) {
    if (!S.reveal) { el.truth.hidden = true; return; }
    var fails = Object.keys(D.failed).map(function (b) { return "bearing " + b + " " + esc(D.failed[b]); });
    var sm = (D.summary || {})[S.detector] || {};
    var lines = Object.keys(D.failed).map(function (b) {
      var r = sm[b];
      if (!r || r.persistent_alarm_h === null) return "<p>" + esc(S.detector) + " never raised a lasting alarm on bearing " + b + ".</p>";
      return "<p>" + esc(S.detector) + ": the alarm on bearing " + b + " stayed on from <b>" + fmt(r.persistent_alarm_h, 1) +
        " h</b>, giving <b>" + fmt(r.lead_time_h, 1) + " h</b> of warning before the end" +
        (r.false_alarm_episodes ? " (" + r.false_alarm_episodes + " false alarm episode" + (r.false_alarm_episodes > 1 ? "s" : "") + " earlier)" : "") + ".</p>";
    });
    el.truth.innerHTML = "<p><b>Ground truth:</b> test ends at <b>" + fmt(D.duration_h, 0) + " h</b> with " + fails.join(", ") + ".</p>" + lines.join("");
    el.truth.hidden = false;
  }

  function renderBearingSel(D) {
    el.bearingSel.innerHTML = bearings(D).map(function (b) {
      return '<button type="button" role="radio" aria-checked="' + (b === S.bSel) + '" data-b="' + b + '">' +
        '<span class="dot" style="background:' + BEARING_COLORS[b] + '"></span>' + b + "</button>";
    }).join("");
  }

  function isNarrow() { return window.matchMedia("(max-width: 560px)").matches; }

  function baseLayout(D, extra) {
    var narrow = isNarrow();
    var L = {
      paper_bgcolor: "#ffffff", plot_bgcolor: "#ffffff",
      font: { family: getComputedStyle(document.body).fontFamily, size: narrow ? 11 : 12, color: "#1f1e1c" },
      margin: { l: narrow ? 44 : 56, r: 10, t: narrow ? 44 : 34, b: 44 },
      xaxis: { title: { text: "hours since start", font: { color: MUTED } }, range: [0, D.endH], gridcolor: GRID,
        zeroline: false, linecolor: "#d6d5d0", fixedrange: narrow },
      yaxis: { gridcolor: GRID, zeroline: false, linecolor: "#d6d5d0", fixedrange: narrow, automargin: true },
      legend: { orientation: "h", x: 0, y: 1.02, yanchor: "bottom", font: { size: narrow ? 10 : 12 } },
      hovermode: "closest", dragmode: narrow ? false : "zoom",
      hoverlabel: { font: { family: getComputedStyle(document.body).fontFamily } }
    };
    for (var k in extra) L[k] = Object.assign({}, L[k] || {}, extra[k]);
    return L;
  }
  var CONFIG = { responsive: true, displayModeBar: false, scrollZoom: false };

  // Over a wide log range Plotly labels every minor tick; label only 1-2-5 per decade then.
  function wideRange(traces) {
    var lo = Infinity, hi = -Infinity;
    traces.forEach(function (t) {
      for (var k = 0; k < t.y.length; k++) { var v = t.y[k]; if (v > 0) { if (v < lo) lo = v; if (v > hi) hi = v; } }
    });
    return hi / lo > 20;
  }

  function renderScoreChart(D, i) {
    var Dd = D.detectors[S.detector];
    var traces = bearings(D).map(function (b) {
      var B = Dd.bearings[b], sr = series(D, B.ratio, B.dec, i + 1);
      return {
        type: "scatter", mode: "lines", name: "Bearing " + b, x: sr.x, y: sr.y,
        line: { color: BEARING_COLORS[b], width: 1.4 },
        hovertemplate: "Bearing " + b + "<br>%{x:.1f} h<br>score / threshold %{y:.2f}<extra></extra>"
      };
    });
    var tE = D.hours[D.train_end], cE = D.hours[D.calib_end];
    var layout = baseLayout(D, {
      yaxis: Object.assign({ title: { text: "score / threshold", font: { color: MUTED } }, type: Dd.log_axis ? "log" : "linear" },
        Dd.log_axis && wideRange(traces) ? { tickmode: "array", tickvals: LOG_TICKS, ticktext: LOG_TICKS.map(String) } : {}),
      uirevision: D.test + "|" + S.detector
    });
    layout.shapes = [
      { type: "rect", xref: "x", yref: "paper", x0: 0, x1: tE, y0: 0, y1: 1, fillcolor: "#8a8984", opacity: 0.08, line: { width: 0 }, layer: "below" },
      { type: "rect", xref: "x", yref: "paper", x0: tE, x1: cE, y0: 0, y1: 1, fillcolor: "#8a8984", opacity: 0.16, line: { width: 0 }, layer: "below" },
      { type: "line", xref: "paper", yref: "y", x0: 0, x1: 1, y0: 1, y1: 1, line: { color: MUTED, width: 1.2, dash: "dash" } }
    ];
    layout.annotations = [
      { x: 0, xref: "x", y: 1, yref: "paper", text: "train", showarrow: false, xanchor: "left", yanchor: "top", font: { size: 11, color: MUTED } },
      { x: tE, xref: "x", y: 1, yref: "paper", text: "calibration", showarrow: false, xanchor: "left", yanchor: "top", font: { size: 11, color: MUTED } }
    ];
    Plotly.react(el.score, traces, layout, CONFIG);
  }

  function renderEnvChart(D, i) {
    var b = S.bSel;
    var traces = S.index.defects.map(function (d) {
      var sr = series(D, D.zRoll[b][d.key], D.zDec[b][d.key], i + 1);
      return {
        type: "scatter", mode: "lines", name: d.label + " (" + Math.round(d.hz) + " Hz)", x: sr.x,
        y: sr.y, line: { color: DEFECT_COLORS[d.key], width: 1.4 },
        hovertemplate: d.label + "<br>%{x:.1f} h<br>z = %{y:.1f}<extra></extra>"
      };
    });
    var layout = baseLayout(D, {
      yaxis: { title: { text: "z-score vs. healthy", font: { color: MUTED } } },
      uirevision: D.test + "|" + b
    });
    layout.shapes = [{ type: "line", xref: "paper", yref: "y", x0: 0, x1: 1, y0: 3, y1: 3, line: { color: MUTED, width: 1.2, dash: "dot" } }];
    Plotly.react(el.env, traces, layout, CONFIG);
  }

  function renderCharts() {
    var D = S.data;
    if (!D) return;
    S.lastChart = performance.now();
    renderScoreChart(D, S.idx - 1);
    renderEnvChart(D, S.idx - 1);
  }

  // Charts are redrawn at most every FRAME_MS, with a trailing redraw so the last position always shows.
  function scheduleCharts(force) {
    var wait = FRAME_MS - (performance.now() - S.lastChart);
    if (force || wait <= 0) {
      clearTimeout(S.chartTimer); S.chartTimer = null;
      renderCharts();
    } else if (!S.chartTimer) {
      S.chartTimer = setTimeout(function () { S.chartTimer = null; renderCharts(); }, wait);
    }
  }

  function render(forceCharts) {
    var D = S.data;
    if (!D) return;
    var i = S.idx - 1;
    el.pos.value = S.idx;
    renderReadout(D, i);
    renderTiles(D, i);
    renderLog(D, i);
    renderTruth(D);
    scheduleCharts(forceCharts);
  }

  function requestRender() {
    if (S.rafPending) return;
    S.rafPending = true;
    requestAnimationFrame(function () { S.rafPending = false; render(false); });
  }

  // ------------------------------------------------------------------ playback
  function setPlaying(on) {
    S.playing = on;
    el.play.setAttribute("aria-pressed", String(on));
    el.play.querySelector(".ico").innerHTML = on ? "&#10074;&#10074;" : "&#9654;";
    el.play.querySelector(".lbl").textContent = on ? "Pause" : "Play";
    if (on) { S.lastTick = 0; requestAnimationFrame(tick); }
  }

  function atEnd() {
    if (!S.revealTouched && !S.reveal) setReveal(true);
  }

  function tick(t) {
    if (!S.playing) return;
    if (!S.lastTick) S.lastTick = t;
    if (t - S.lastTick >= FRAME_MS) {
      // advance by wall-clock time so playback speed holds even if a frame renders slowly
      var steps = Math.min(5, Math.floor((t - S.lastTick) / FRAME_MS));
      S.lastTick = steps === 5 ? t : S.lastTick + steps * FRAME_MS;
      var n = S.data.n;
      S.idx = Math.min(n, S.idx + steps * Number(el.speed.value));
      if (S.idx >= n) { setPlaying(false); atEnd(); render(true); return; }
      render(false);
    }
    requestAnimationFrame(tick);
  }

  function setReveal(on) {
    S.reveal = on;
    el.reveal.checked = on;
    if (on) {
      var failed = Object.keys(S.data.failed).map(Number);
      if (failed.length) S.bSel = Math.min.apply(null, failed);
    }
    renderBearingSel(S.data);
    render(true);
  }

  // ------------------------------------------------------------------ wiring
  function selectTest(name) {
    setPlaying(false);
    el.test.disabled = true;
    return loadTest(name).then(function (D) {
      S.test = name; S.data = D; S.logKey = "";
      S.idx = D.first; // playback starts where monitoring starts
      S.reveal = false; S.revealTouched = false; el.reveal.checked = false;
      S.bSel = 1;
      el.pos.max = D.n; el.pos.min = 1;
      renderBearingSel(D);
      render(true);
    }).finally(function () { el.test.disabled = false; });
  }

  function wire() {
    el.test.addEventListener("change", function () { selectTest(el.test.value); });
    el.detector.addEventListener("change", function () { S.detector = el.detector.value; S.logKey = ""; render(true); });
    el.play.addEventListener("click", function () {
      if (!S.playing && S.idx >= S.data.n) S.idx = S.data.first;
      setPlaying(!S.playing);
    });
    el.restart.addEventListener("click", function () { S.idx = S.data.first; render(true); });
    el.pos.addEventListener("input", function () {
      S.idx = Number(el.pos.value);
      if (S.idx >= S.data.n) atEnd();
      requestRender();
    });
    el.reveal.addEventListener("change", function () { S.revealTouched = true; setReveal(el.reveal.checked); });
    el.bearingSel.addEventListener("click", function (e) {
      var btn = e.target.closest("button[data-b]");
      if (!btn) return;
      S.bSel = Number(btn.getAttribute("data-b"));
      renderBearingSel(S.data);
      renderEnvChart(S.data, S.idx - 1);
    });
    document.addEventListener("keydown", function (e) {
      var tag = (e.target.tagName || "").toLowerCase();
      if (e.code === "Space" && tag !== "button" && tag !== "select" && tag !== "input") {
        e.preventDefault(); el.play.click();
      }
    });
    var lastNarrow = isNarrow();
    window.addEventListener("resize", function () {
      if (isNarrow() !== lastNarrow) { lastNarrow = isNarrow(); render(true); }
    });
  }

  function init() {
    Promise.all([getJSON("data/index.json"), ensurePlotly()]).then(function (r) {
      var idx = r[0];
      S.index = idx;
      el.test.innerHTML = idx.tests.map(function (t) {
        return '<option value="' + esc(t.name) + '">' + esc(t.name) + " (" + esc(t.folder) + ")</option>";
      }).join("");
      el.detector.innerHTML = idx.detectors.map(function (d) {
        return '<option value="' + esc(d) + '">' + esc(d) + "</option>";
      }).join("");
      el.test.value = idx.default_test;
      el.detector.value = idx.default_detector;
      S.detector = idx.default_detector;
      el.footnote.textContent = "Detectors were fitted on the first " + Math.round(idx.split.train_end * 100) +
        " % of each test's snapshots and calibrated on the next " + Math.round((idx.split.calib_end - idx.split.train_end) * 100) +
        " %. An alarm needs " + idx.alarm.k + " of the last " + idx.alarm.n + " snapshots above threshold. " +
        "Playback starts where monitoring starts; drag the slider left to see the training period. " +
        "The first bearing to raise an alarm is flagged as the likely source, because a failing bearing's vibration " +
        "travels through the shaft and housing and lifts its neighbours too.";
      wire();
      return selectTest(idx.default_test);
    }).catch(function (err) {
      el.readout.innerHTML = '<span class="error">Could not load the dashboard: ' + esc(err.message) + "</span>";
      if (window.console) console.error(err);
    });
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init);
  else init();
})();
