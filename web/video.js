// Video analysis: per-person emotion statistics for a local video file.
// The video is decoded and analysed in this browser tab; nothing is uploaded.

const $ = (id) => document.getElementById(id);
const COLORS = { angry: "#e34948", disgust: "#1baf7a", fear: "#4a3aa7", happy: "#eda100",
  neutral: "#2a78d6", sad: "#8a94a6", surprise: "#e87ba4", unsure: "#3b4252" };
const MAX_WIDTH = 1280;   // frames are scaled down to this width before analysis
const MAX_GAP_S = 2.0;    // a person missing for longer than this starts a new track
const job = { id: 0, url: null, result: null };

const fmt = (s) => `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, "0")}`;
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const center = (b) => [b[0] + b[2] / 2, b[1] + b[3] / 2];
const iou = (p, q) => {
  const x1 = Math.max(p[0], q[0]), y1 = Math.max(p[1], q[1]);
  const x2 = Math.min(p[0] + p[2], q[0] + q[2]), y2 = Math.min(p[1] + p[3], q[1] + q[3]);
  const inter = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
  return inter / (p[2] * p[3] + q[2] * q[3] - inter);
};

function seek(video, t) {
  return new Promise((resolve) => {
    video.addEventListener("seeked", resolve, { once: true });
    video.currentTime = t;
  });
}

// Link this frame's faces to the people already found: overlapping boxes, or a nearby box
// of similar size if the person was briefly missing.
function assign(people, faces, t) {
  const used = new Set();
  for (const f of faces) {
    let best = null, bestScore = 0;
    for (const p of people) {
      if (used.has(p) || t - p.lastT > MAX_GAP_S) continue;
      const o = iou(p.box, f.box);
      const [ax, ay] = center(p.box), [bx, by] = center(f.box);
      const near = Math.hypot(ax - bx, ay - by) < 0.6 * Math.max(p.box[2], f.box[2])
        && Math.abs(Math.log(p.box[2] / f.box[2])) < 0.4;
      const score = o > 0.2 ? 1 + o : near ? 0.5 : 0;
      if (score > bestScore) { best = p; bestScore = score; }
    }
    if (!best) {
      best = { id: people.length + 1, samples: [], thumb: null, thumbSize: 0 };
      people.push(best);
    }
    used.add(best);
    best.box = f.box;
    best.lastT = t;
    f.person = best;
  }
}

function thumbnail(canvas, box) {
  const side = Math.max(box[2], box[3]) * 1.4, c = document.createElement("canvas");
  c.width = c.height = 96;
  c.getContext("2d").drawImage(canvas, box[0] + box[2] / 2 - side / 2, box[1] + box[3] / 2 - side / 2, side, side, 0, 0, 96, 96);
  return c.toDataURL("image/jpeg", 0.85);
}

async function analyze() {
  const id = ++job.id;
  const ed = window.emotionDetecter;
  const emotions = ed.state.manifest.emotions;
  const fps = Number($("vaRate").value);
  const thorough = $("vaThorough").checked;
  const video = document.createElement("video");
  video.muted = true;
  video.src = job.url;
  await new Promise((r, e) => { video.onloadedmetadata = r; video.onerror = () => e(new Error("This video format cannot be decoded")); });
  const scale = Math.min(1, MAX_WIDTH / video.videoWidth);
  const canvas = document.createElement("canvas");
  canvas.width = Math.round(video.videoWidth * scale);
  canvas.height = Math.round(video.videoHeight * scale);
  const ctx = canvas.getContext("2d", { willReadFrequently: true });

  const step = 1 / fps, people = [];
  let rejected = 0, frames = 0;
  const started = performance.now();
  for (let t = 0; t < video.duration; t += step) {
    if (id !== job.id) return;  // cancelled or restarted
    await seek(video, t);
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    const r = await ed.analyzeStill(canvas, { thorough });
    rejected += r.rejected;
    frames++;
    assign(people, r.faces, t);
    for (const f of r.faces) {
      const top = f.probs.indexOf(Math.max(...f.probs));
      f.person.samples.push({ t, emotion: f.unsure ? "unsure" : emotions[top], confidence: f.probs[top], probs: f.probs });
      if (f.box[2] > f.person.thumbSize) { f.person.thumb = thumbnail(canvas, f.box); f.person.thumbSize = f.box[2]; }
    }
    const done = Math.min(1, (t + step) / video.duration);
    const eta = (performance.now() - started) / done * (1 - done) / 1000;
    $("vaProgress").style.width = `${(done * 100).toFixed(1)}%`;
    $("vaStatus").textContent = `Analysing ${fmt(t)} / ${fmt(video.duration)} · ${people.length} people so far · about ${Math.ceil(eta)} s left`;
    await sleep(0);  // keep the page responsive
  }

  // people seen in only one or two samples are usually detector noise
  const kept = people.filter((p) => p.samples.length >= Math.max(3, fps));
  kept.forEach((p, i) => { p.n = i + 1; });
  job.result = { people: kept, step, duration: video.duration, emotions, fps, frames, rejected };
  $("vaStatus").textContent = `Done: ${frames} frames analysed in ${((performance.now() - started) / 1000).toFixed(0)} s · ` +
    `${kept.length} ${kept.length === 1 ? "person" : "people"} · ${rejected} non-face detections ignored`;
  $("vaAnalyze").textContent = "Analyse again";
  $("vaExport").hidden = !kept.length;
  render(job.result);
}

function render({ people, step, duration, emotions }) {
  const labels = [...emotions, "unsure"];
  $("vaResults").innerHTML = people.length ? "" : `<p class="game-intro">No faces found in this video.</p>`;
  for (const p of people) {
    const seconds = Object.fromEntries(labels.map((e) => [e, 0]));
    for (const s of p.samples) seconds[s.emotion] += step;
    const onScreen = p.samples.length * step;
    const ranked = labels.filter((e) => seconds[e] > 0).sort((a, b) => seconds[b] - seconds[a]);
    const dominant = ranked.find((e) => e !== "unsure") || "unsure";
    const card = document.createElement("div");
    card.className = "va-person";
    card.innerHTML =
      `<div class="va-head"><img src="${p.thumb}" alt="Person ${p.n}"><div>` +
      `<div class="va-name">Person ${p.n}</div>` +
      `<div class="va-sub">on screen ${fmt(onScreen)} · mostly <b>${dominant}</b></div></div></div>` +
      `<div class="va-stack">${ranked.map((e) => `<span style="width:${(seconds[e] / onScreen * 100).toFixed(2)}%;` +
        `background:${COLORS[e]}" title="${e}: ${seconds[e].toFixed(1)} s"></span>`).join("")}</div>` +
      `<table class="va-table">${ranked.map((e) => `<tr><td><i style="background:${COLORS[e]}"></i>${e}</td>` +
        `<td>${seconds[e].toFixed(1)} s</td><td>${(seconds[e] / onScreen * 100).toFixed(1)}%</td></tr>`).join("")}</table>` +
      `<canvas class="va-timeline" title="Click to jump to that moment"></canvas>`;
    $("vaResults").append(card);
    drawTimeline(card.querySelector("canvas"), p, step, duration);
  }
  $("vaLegend").innerHTML = labels.map((e) => `<span><i style="background:${COLORS[e]}"></i>${e}</span>`).join("");
  $("vaLegend").hidden = !people.length;
}

function drawTimeline(canvas, person, step, duration) {
  const w = canvas.clientWidth || 600, h = 18, dpr = window.devicePixelRatio || 1;
  canvas.width = w * dpr;
  canvas.height = h * dpr;
  const ctx = canvas.getContext("2d");
  ctx.scale(dpr, dpr);
  ctx.fillStyle = "rgba(148, 163, 184, 0.15)";
  ctx.fillRect(0, 0, w, h);
  for (const s of person.samples) {
    ctx.fillStyle = COLORS[s.emotion];
    ctx.fillRect((s.t / duration) * w, 0, Math.max(1, (step / duration) * w + 0.5), h);
  }
  canvas.onclick = (e) => {
    const r = canvas.getBoundingClientRect();
    $("vaPreview").currentTime = ((e.clientX - r.left) / r.width) * duration;
    $("vaPreview").play();
  };
}

function download(name, text, type) {
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([text], { type }));
  a.download = name;
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
}

function exportResults(kind) {
  const r = job.result;
  if (!r) return;
  if (kind === "csv") {
    const rows = [["person", "time_s", "emotion", "confidence", ...r.emotions.map((e) => `p_${e}`)]];
    for (const p of r.people) {
      for (const s of p.samples) rows.push([p.n, s.t.toFixed(2), s.emotion, s.confidence.toFixed(3), ...s.probs.map((v) => v.toFixed(3))]);
    }
    download("emotions.csv", rows.map((row) => row.join(",")).join("\n"), "text/csv");
  } else {
    const summary = r.people.map((p) => {
      const seconds = {};
      for (const s of p.samples) seconds[s.emotion] = (seconds[s.emotion] || 0) + r.step;
      return { person: p.n, on_screen_s: +(p.samples.length * r.step).toFixed(2),
        seconds: Object.fromEntries(Object.entries(seconds).map(([k, v]) => [k, +v.toFixed(2)])) };
    });
    download("emotions.json", JSON.stringify({ duration_s: r.duration, sample_rate_fps: r.fps,
      model: window.emotionDetecter.state.model.version, people: summary }, null, 2), "application/json");
  }
}

$("vaFile").addEventListener("change", () => {
  const file = $("vaFile").files[0];
  if (!file) return;
  job.id++;  // stop any running analysis
  if (job.url) URL.revokeObjectURL(job.url);
  job.url = URL.createObjectURL(file);
  $("vaPreview").src = job.url;
  $("vaPreview").hidden = false;
  $("vaAnalyze").disabled = false;
  $("vaAnalyze").textContent = "Analyse";
  $("vaResults").innerHTML = "";
  $("vaLegend").hidden = true;
  $("vaExport").hidden = true;
  $("vaStatus").textContent = `${file.name} · ready`;
  $("vaProgress").style.width = "0%";
});
$("vaAnalyze").addEventListener("click", () => analyze().catch((e) => { $("vaStatus").textContent = `Error: ${e.message}`; }));
$("vaCsv").addEventListener("click", () => exportResults("csv"));
$("vaJson").addEventListener("click", () => exportResults("json"));
