// emotionDetecter live: face detection (MediaPipe) -> alignment -> emotion model (ONNX Runtime Web).
// Everything runs in the browser; no frame ever leaves the device.
import { FaceDetector, FilesetResolver } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.1.0/vision_bundle.mjs";
import * as ort from "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/ort.all.min.mjs";

const MEDIAPIPE = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.1.0/wasm";
const DETECTOR_MODEL =
  "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite";
ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/";

const SUPERSAMPLE = 4;     // faces are cut out at 4x and box-averaged down, like a camera sensor would
const SMOOTH_MS = 300;     // time constant of the probability smoothing
const TRACK_TTL_MS = 600;  // forget a face that has not been seen for this long
const MIN_SCORE = 0.5;     // face detector confidence
const MAX_FACES = 4;

const $ = (id) => document.getElementById(id);
const ui = {
  video: $("video"), still: $("still"), overlay: $("overlay"), viewport: $("viewport"), empty: $("empty"),
  camera: $("camera"), file: $("file"), smooth: $("smooth"), showCrops: $("showCrops"), stats: $("stats"),
  models: $("models"), verdict: $("verdict"), bars: $("bars"), crops: $("crops"), cropsRow: $("cropsRow"),
};
const state = {
  manifest: null, model: null, sessions: new Map(), detectors: {}, mode: "idle", stream: null,
  tracks: [], nextId: 1, busy: false, fps: 0, lastTick: 0, inferMs: 0, faces: 0, photo: null,
};

// ---------------------------------------------------------------- alignment (same math as src/align.py)

// Least-squares similarity transform (scale, rotation, translation) mapping src points onto dst points.
// Returned as canvas setTransform() arguments: x' = a*x + c*y + e, y' = b*x + d*y + f.
export function fitSimilarity(src, dst) {
  const n = src.length;
  let sx = 0, sy = 0, dx = 0, dy = 0;
  for (let i = 0; i < n; i++) { sx += src[i][0]; sy += src[i][1]; dx += dst[i][0]; dy += dst[i][1]; }
  sx /= n; sy /= n; dx /= n; dy /= n;
  let num = 0, cross = 0, varSrc = 0;
  for (let i = 0; i < n; i++) {
    const px = src[i][0] - sx, py = src[i][1] - sy, qx = dst[i][0] - dx, qy = dst[i][1] - dy;
    num += px * qx + py * qy;
    cross += px * qy - py * qx;
    varSrc += px * px + py * py;
  }
  const c = num / varSrc, s = cross / varSrc;  // scale*cos, scale*sin
  return { a: c, b: s, c: -s, d: c, e: dx - (c * sx - s * sy), f: dy - (s * sx + c * sy) };
}

const work = document.createElement("canvas");
const workCtx = work.getContext("2d", { willReadFrequently: true });

// Training (OpenCV) extends the image border outward when a crop reaches past the edge.
// Do the same here: copy the image onto a larger canvas and stretch its outermost pixels.
function replicatePad(source, width, height, pad) {
  const c = document.createElement("canvas");
  c.width = width + 2 * pad;
  c.height = height + 2 * pad;
  const ctx = c.getContext("2d");
  ctx.imageSmoothingEnabled = false;
  ctx.drawImage(source, pad, pad, width, height);
  ctx.drawImage(source, 0, 0, 1, height, 0, pad, pad, height);                        // left
  ctx.drawImage(source, width - 1, 0, 1, height, pad + width, pad, pad, height);      // right
  ctx.drawImage(c, 0, pad, c.width, 1, 0, 0, c.width, pad);                           // top
  ctx.drawImage(c, 0, pad + height - 1, c.width, 1, 0, pad + height, c.width, pad);   // bottom
  return c;
}

// Source-image area a crop needs: the template square mapped back through the alignment.
function cropReach(m, size) {
  const det = m.a * m.d - m.b * m.c;
  return [[0, 0], [size, 0], [0, size], [size, size]].map(([x, y]) => {
    const u = x - m.e, v = y - m.f;
    return [(m.d * u - m.c * v) / det, (-m.b * u + m.a * v) / det];
  });
}

// Cut one face out of `source`, aligned to the model's template. Returns size*size gray values (0-255).
// Training used OpenCV (pixel centers at integer coordinates); the half-pixel shifts below make the
// browser sample exactly the same spots.
export function extractFace(source, keypoints, model) {
  const S = model.size, K = SUPERSAMPLE, big = S * K;
  let m = fitSimilarity(keypoints, model.template);
  const width = source.videoWidth || source.width, height = source.videoHeight || source.height;
  const reach = cropReach(m, S);
  const overflow = Math.max(0, ...reach.map(([x, y]) => Math.max(-x, -y, x - width, y - height)));
  if (overflow > 0) {
    const pad = Math.ceil(overflow) + 2;
    source = replicatePad(source, width, height, pad);
    m = fitSimilarity(keypoints.map(([x, y]) => [x + pad, y + pad]), model.template);
  }
  work.width = big;
  work.height = big;
  workCtx.fillStyle = "#808080";
  workCtx.fillRect(0, 0, big, big);
  workCtx.imageSmoothingEnabled = true;
  workCtx.imageSmoothingQuality = "medium";
  workCtx.setTransform(m.a * K, m.b * K, m.c * K, m.d * K,
    (m.e - 0.5 * (m.a + m.c) + 0.5) * K, (m.f - 0.5 * (m.b + m.d) + 0.5) * K);
  workCtx.drawImage(source, 0, 0);
  workCtx.setTransform(1, 0, 0, 1, 0, 0);
  const px = workCtx.getImageData(0, 0, big, big).data;
  const gray = new Float32Array(S * S);
  for (let y = 0; y < S; y++) {
    for (let x = 0; x < S; x++) {
      let sum = 0;
      for (let j = 0; j < K; j++) {
        let i = ((y * K + j) * big + x * K) * 4;
        for (let k = 0; k < K; k++, i += 4) sum += 0.299 * px[i] + 0.587 * px[i + 1] + 0.114 * px[i + 2];
      }
      gray[y * S + x] = sum / (K * K);
    }
  }
  return gray;
}

// ---------------------------------------------------------------- model

async function loadSession(entry) {
  if (!state.sessions.has(entry.version)) {
    let session, backend = "WebGPU";
    try {
      if (!navigator.gpu) throw new Error("no WebGPU");
      session = await ort.InferenceSession.create(entry.file, { executionProviders: ["webgpu"] });
    } catch {
      backend = "WASM";
      session = await ort.InferenceSession.create(entry.file, { executionProviders: ["wasm"] });
    }
    state.sessions.set(entry.version, { session, backend });
  }
  return state.sessions.get(entry.version);
}

const softmax = (z, o) => {
  let max = -Infinity;
  for (let k = 0; k < 7; k++) max = Math.max(max, z[o + k]);
  const e = Array.from({ length: 7 }, (_, k) => Math.exp(z[o + k] - max));
  const sum = e.reduce((a, b) => a + b, 0);
  return e.map((v) => v / sum);
};

// Classify aligned crops; every face is also classified mirrored and the two answers averaged.
async function classify(crops, model) {
  const S = model.size, plane = S * S, n = crops.length;
  const data = new Float32Array(2 * n * plane);
  crops.forEach((g, i) => {
    const o = 2 * i * plane, of = o + plane;
    for (let y = 0; y < S; y++) {
      for (let x = 0; x < S; x++) {
        const v = (g[y * S + x] / 255 - model.mean) / model.std;
        data[o + y * S + x] = v;
        data[of + y * S + (S - 1 - x)] = v;
      }
    }
  });
  const t0 = performance.now();
  const out = await model.session.run({ input: new ort.Tensor("float32", data, [2 * n, 1, S, S]) });
  state.inferMs = 0.8 * state.inferMs + 0.2 * (performance.now() - t0);
  const z = out.logits.data;
  return crops.map((_, i) => {
    const a = softmax(z, 14 * i), b = softmax(z, 14 * i + 7);
    return a.map((p, k) => (p + b[k]) / 2);
  });
}

// ---------------------------------------------------------------- tracking + smoothing

const iou = (p, q) => {
  const x1 = Math.max(p[0], q[0]), y1 = Math.max(p[1], q[1]);
  const x2 = Math.min(p[0] + p[2], q[0] + q[2]), y2 = Math.min(p[1] + p[3], q[1] + q[3]);
  const inter = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
  return inter / (p[2] * p[3] + q[2] * q[3] - inter);
};

// Each face keeps a running average of its probabilities, so a single noisy frame cannot flip the label.
function track(faces, probs, now, smooth) {
  const used = new Set();
  faces.forEach((face, i) => {
    let best = null, bestIou = 0.3;
    for (const t of state.tracks) {
      const o = iou(t.box, face.box);
      if (!used.has(t) && o > bestIou) { best = t; bestIou = o; }
    }
    if (best) {
      const alpha = smooth ? 1 - Math.exp(-(now - best.seen) / SMOOTH_MS) : 1;
      best.probs = best.probs.map((p, k) => p + alpha * (probs[i][k] - p));
      best.box = face.box;
      best.seen = now;
    } else {
      best = { id: state.nextId++, box: face.box, probs: probs[i].slice(), seen: now };
      state.tracks.push(best);
    }
    used.add(best);
    face.track = best;
  });
  state.tracks = state.tracks.filter((t) => now - t.seen < TRACK_TTL_MS);
}

// ---------------------------------------------------------------- pipeline

async function processFrame(source, width, height, detections, mirrored, smooth) {
  const now = performance.now();
  const faces = detections
    .filter((d) => d.categories[0].score >= MIN_SCORE && d.keypoints.length >= 4)
    .map((d) => ({
      box: [d.boundingBox.originX, d.boundingBox.originY, d.boundingBox.width, d.boundingBox.height],
      kp: d.keypoints.slice(0, 4).map((k) => [k.x * width, k.y * height]),
    }))
    .sort((p, q) => q.box[2] * q.box[3] - p.box[2] * p.box[3])
    .slice(0, MAX_FACES);
  const model = state.model;
  let crops = [], mainProbs = null;
  if (faces.length) {
    crops = faces.map((f) => extractFace(source, f.kp, model));
    const probs = await classify(crops, model);
    if (model !== state.model) return;  // the user switched models meanwhile
    track(faces, probs, now, smooth);
    mainProbs = probs[0];  // largest face, unsmoothed: games need the fastest reaction
  } else {
    state.tracks = state.tracks.filter((t) => now - t.seen < TRACK_TTL_MS);
  }
  state.faces = faces.length;
  if (state.mode === "camera") {
    window.dispatchEvent(new CustomEvent("emotion", { detail: {
      probs: mainProbs, smoothed: faces[0]?.track.probs ?? null, box: faces[0]?.box ?? null,
      width, height, source, emotions: state.manifest.emotions, emoji: state.manifest.emoji,
    } }));
  }
  if (source === ui.video || source === ui.still) drawOverlay(faces, width, height, mirrored);
  renderPanel(faces[0]?.track, faces.length);
  renderCrops(crops, model.size);
  tickFps(now);
}

// One loop per camera target: moving the camera starts a new loop and the old one ends here.
async function cameraLoop(id) {
  if (id !== state.loopId || state.mode !== "camera" || !state.source) return;
  const v = state.source;
  if (v.readyState >= 2 && !state.busy) {
    state.busy = true;
    try {
      const result = state.detectors.video.detectForVideo(v, performance.now());
      await processFrame(v, v.videoWidth, v.videoHeight, result.detections, true, ui.smooth.checked);
    } finally {
      state.busy = false;
    }
  }
  if (v.requestVideoFrameCallback) v.requestVideoFrameCallback(() => cameraLoop(id));
  else requestAnimationFrame(() => cameraLoop(id));
}

// Photos: faces filling the whole picture are hard for the detector, so retry on a padded copy.
function detectInImage(canvas) {
  let result = state.detectors.image.detect(canvas);
  if (result.detections.length) return result.detections;
  const pad = Math.round(0.25 * Math.max(canvas.width, canvas.height));
  const padded = document.createElement("canvas");
  padded.width = canvas.width + 2 * pad;
  padded.height = canvas.height + 2 * pad;
  const ctx = padded.getContext("2d");
  ctx.fillStyle = "#808080";
  ctx.fillRect(0, 0, padded.width, padded.height);
  ctx.drawImage(canvas, pad, pad);
  result = state.detectors.image.detect(padded);
  return result.detections.map((d) => ({
    ...d,
    boundingBox: { ...d.boundingBox, originX: d.boundingBox.originX - pad, originY: d.boundingBox.originY - pad },
    keypoints: d.keypoints.map((k) => ({
      ...k,
      x: (k.x * padded.width - pad) / canvas.width,
      y: (k.y * padded.height - pad) / canvas.height,
    })),
  }));
}

async function runPhoto(bitmap = state.photo) {
  if (!bitmap) return;
  state.photo = bitmap;
  const c = ui.still;
  c.width = bitmap.width;
  c.height = bitmap.height;
  c.getContext("2d").drawImage(bitmap, 0, 0);
  setAspect(bitmap.width, bitmap.height);
  state.tracks = [];
  await processFrame(c, c.width, c.height, detectInImage(c), false, false);
}

// ---------------------------------------------------------------- drawing

function setAspect(w, h) {
  ui.viewport.style.aspectRatio = `${w} / ${h}`;
}

function drawOverlay(faces, width, height, mirrored) {
  const rect = ui.overlay.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  ui.overlay.width = Math.round(rect.width * dpr);
  ui.overlay.height = Math.round(rect.height * dpr);
  const ctx = ui.overlay.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, rect.width, rect.height);
  const scale = Math.min(rect.width / width, rect.height / height);
  const ox = (rect.width - width * scale) / 2, oy = (rect.height - height * scale) / 2;
  const { emotions, emoji } = state.manifest;

  for (const face of faces) {
    let [x, y, w, h] = face.box;
    if (mirrored) x = width - x - w;
    x = ox + x * scale; y = oy + y * scale; w *= scale; h *= scale;
    const grad = ctx.createLinearGradient(x, y, x + w, y + h);
    grad.addColorStop(0, "#8b5cf6");
    grad.addColorStop(1, "#06b6d4");
    ctx.lineWidth = 3;
    ctx.strokeStyle = grad;
    ctx.beginPath();
    ctx.roundRect(x, y, w, h, 14);
    ctx.stroke();

    const p = face.track.probs, top = p.indexOf(Math.max(...p));
    const label = `${emoji[top]} ${emotions[top]} ${Math.round(p[top] * 100)}%`;
    ctx.font = "600 15px -apple-system, BlinkMacSystemFont, Inter, sans-serif";
    const tw = ctx.measureText(label).width + 20, th = 28;
    const ly = y - th - 8 >= 0 ? y - th - 8 : y + 8;
    ctx.fillStyle = "rgba(15, 23, 42, 0.82)";
    ctx.beginPath();
    ctx.roundRect(x, ly, tw, th, 999);
    ctx.fill();
    ctx.fillStyle = "#fff";
    ctx.textBaseline = "middle";
    ctx.fillText(label, x + 10, ly + th / 2 + 1);
  }
}

function buildBars() {
  const { emotions, emoji } = state.manifest;
  ui.bars.innerHTML = "";
  emotions.forEach((e, i) => {
    const li = document.createElement("li");
    li.innerHTML = `<span>${emoji[i]}</span><span class="name">${e}</span>` +
      `<span class="track"><span class="fill"></span></span><span class="value">0%</span>`;
    ui.bars.append(li);
  });
}

function renderPanel(track, nFaces) {
  const { emotions, emoji } = state.manifest;
  const rows = ui.bars.children;
  if (!track) {
    for (const li of rows) {
      li.classList.remove("best");
      li.querySelector(".fill").style.width = "0%";
      li.querySelector(".value").textContent = "–";
    }
    ui.verdict.innerHTML = `<div class="verdict-emoji">·</div><div><div class="verdict-label">No face</div>` +
      `<div class="verdict-sub">Look at the camera, or try another photo</div></div>`;
    return;
  }
  const p = track.probs, top = p.indexOf(Math.max(...p));
  [...rows].forEach((li, i) => {
    li.classList.toggle("best", i === top);
    li.querySelector(".fill").style.width = `${(p[i] * 100).toFixed(1)}%`;
    li.querySelector(".value").textContent = `${Math.round(p[i] * 100)}%`;
  });
  const others = nFaces > 1 ? ` · largest of ${nFaces} faces` : "";
  ui.verdict.innerHTML = `<div class="verdict-emoji">${emoji[top]}</div><div><div class="verdict-label">${emotions[top]}</div>` +
    `<div class="verdict-sub">${Math.round(p[top] * 100)}% confident · ${state.model.version}${others}</div></div>`;
}

function renderCrops(crops, size) {
  if (!ui.showCrops.checked) return;
  ui.cropsRow.innerHTML = "";
  for (const g of crops) {
    const c = document.createElement("canvas");
    c.width = size;
    c.height = size;
    const img = c.getContext("2d").createImageData(size, size);
    g.forEach((v, i) => img.data.set([v, v, v, 255], i * 4));
    c.getContext("2d").putImageData(img, 0, 0);
    ui.cropsRow.append(c);
  }
}

function tickFps(now) {
  if (state.lastTick) state.fps = 0.9 * state.fps + 0.1 * (1000 / Math.max(1, now - state.lastTick));
  state.lastTick = now;
  updateStats();
}

function updateStats() {
  const s = state.sessions.get(state.model?.version);
  const parts = [s ? s.backend : "…"];
  if (state.mode === "camera") parts.push(`${state.fps.toFixed(0)} fps`);
  if (state.inferMs) parts.push(`model ${state.inferMs.toFixed(1)} ms`);
  parts.push(`${state.faces} face${state.faces === 1 ? "" : "s"}`);
  ui.stats.textContent = parts.join(" · ");
}

// ---------------------------------------------------------------- controls

function buildModelChips() {
  for (const m of state.manifest.models) {
    const chip = document.createElement("button");
    chip.className = "chip";
    chip.setAttribute("role", "radio");
    chip.dataset.version = m.version;
    chip.textContent = m.version;
    chip.dataset.model = m.version;
    chip.addEventListener("click", () => selectModel(m.version));
    ui.models.append(chip);
  }
}

async function selectModel(version) {
  const entry = state.manifest.models.find((m) => m.version === version);
  ui.stats.textContent = `loading ${version}…`;
  const { session } = await loadSession(entry);
  state.model = { ...entry, session };
  state.tracks = [];  // never blend probabilities from two different models
  for (const chip of ui.models.children) chip.setAttribute("aria-checked", String(chip.dataset.version === version));
  updateStats();
  if (state.mode === "photo") await runPhoto();
}

// The camera feeds one <video> at a time: the main stage, or a game's own camera view.
// Only that element is shown and analysed, so the stage goes dark while a game plays.
async function startCamera(target = ui.video) {
  if (!state.stream) {
    try {
      state.stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: "user" }, audio: false,
      });
    } catch (err) {
      ui.empty.hidden = false;
      ui.empty.innerHTML = `<div class="empty-icon">📷</div><p>Camera not available: ${err.message}</p>` +
        `<p class="muted">Allow camera access in the address bar, or upload a photo instead.</p>`;
      return;
    }
  }
  if (state.source && state.source !== target) state.source.srcObject = null;
  target.srcObject = state.stream;
  await target.play();
  state.source = target;
  const onStage = target === ui.video;
  ui.video.classList.add("mirrored");
  ui.video.hidden = !onStage;
  ui.still.hidden = true;
  ui.empty.hidden = onStage;
  if (onStage) {
    setAspect(ui.video.videoWidth, ui.video.videoHeight);
  } else {
    ui.empty.innerHTML = `<div class="empty-icon">🎮</div><p>The camera is being used by the game below.</p>` +
      `<p class="muted">Press “Start camera” to bring it back here.</p>`;
    drawOverlay([], 1, 1, false);
  }
  ui.camera.textContent = onStage ? "Stop camera" : "Start camera";
  state.tracks = [];
  state.mode = "camera";
  state.loopId = (state.loopId || 0) + 1;
  cameraLoop(state.loopId);
}

function stopCamera() {
  if (state.source && state.source !== ui.video) {  // the camera was in a game: reset the stage message
    ui.empty.innerHTML = `<div class="empty-icon">🙂</div><p>Turn on your camera or drop a photo here.</p>` +
      `<p class="muted">Everything runs on this device. No image or video is uploaded.</p>`;
  }
  state.stream?.getTracks().forEach((t) => t.stop());
  if (state.source) state.source.srcObject = null;
  state.stream = null;
  state.source = null;
  state.mode = "idle";
  ui.camera.textContent = "Start camera";
  window.dispatchEvent(new CustomEvent("camerastop"));
}

async function openPhoto(file) {
  if (!file || !file.type.startsWith("image/")) return;
  stopCamera();
  state.mode = "photo";
  ui.video.hidden = true;
  ui.still.hidden = false;
  ui.empty.hidden = true;
  await runPhoto(await createImageBitmap(file));
}

ui.camera.addEventListener("click", () =>
  (state.mode === "camera" && state.source === ui.video ? stopCamera() : startCamera(ui.video)));
ui.file.addEventListener("change", () => openPhoto(ui.file.files[0]));
ui.showCrops.addEventListener("change", () => {
  ui.crops.hidden = !ui.showCrops.checked;
  if (state.mode === "photo") runPhoto();
});
ui.viewport.addEventListener("dragover", (e) => { e.preventDefault(); ui.viewport.classList.add("dragging"); });
ui.viewport.addEventListener("dragleave", () => ui.viewport.classList.remove("dragging"));
ui.viewport.addEventListener("drop", (e) => {
  e.preventDefault();
  ui.viewport.classList.remove("dragging");
  openPhoto(e.dataTransfer.files[0]);
});
window.addEventListener("resize", () => state.mode === "photo" && runPhoto());

// ---------------------------------------------------------------- model tooltip

const esc = (t) => String(t).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);

function modelTooltipHtml(m) {
  const stat = (label, value) => (value ? `<div><b>${value}</b><span>${label}</span></div>` : "");
  const pctOrNull = (x) => (x == null ? null : `${(x * 100).toFixed(1)}%`);
  return `<div class="tip-head"><span class="tip-name">${esc(m.version)}</span><span class="tip-tag">${esc(m.tagline)}</span></div>` +
    (m.summary ? `<p class="tip-summary">${esc(m.summary)}</p>` : "") +
    `<ul class="tip-list">${m.changes.map((c) => `<li>${esc(c)}</li>`).join("")}</ul>` +
    `<div class="tip-stats">` +
    stat("FER+ test", pctOrNull(m.test_ferplus)) + stat("RAF-DB test", pctOrNull(m.test_rafdb)) +
    stat("webcam sim", pctOrNull(m.webcam)) +
    stat("train images", m.train_images ? m.train_images.toLocaleString("en") : null) +
    stat("training time", m.train_time) +
    `</div>`;
}

const tipEl = document.createElement("div");
tipEl.className = "model-tip";
document.body.append(tipEl);

function showTip(target) {
  const info = state.manifest?.about.find((m) => m.version === target.dataset.model);
  if (!info) return;
  tipEl.innerHTML = modelTooltipHtml(info);
  tipEl.classList.add("show");
  const r = target.getBoundingClientRect(), w = tipEl.offsetWidth, h = tipEl.offsetHeight;
  const left = Math.max(12, Math.min(r.left, window.innerWidth - w - 12));
  const top = r.bottom + 10 + h < window.innerHeight ? r.bottom + 10 : Math.max(12, r.top - h - 10);
  tipEl.style.left = `${left}px`;
  tipEl.style.top = `${top}px`;
}

document.addEventListener("mouseover", (e) => {
  const target = e.target.closest("[data-model]");
  if (target) showTip(target);
});
document.addEventListener("mouseout", (e) => {
  const from = e.target.closest("[data-model]");
  if (from && !from.contains(e.relatedTarget)) tipEl.classList.remove("show");
});

// ---------------------------------------------------------------- about the models

const pct = (x) => (x == null ? "—" : `${(x * 100).toFixed(1)}%`);

function renderAbout() {
  const cards = $("modelCards"), charts = $("charts");
  for (const m of state.manifest.about) {
    const card = document.createElement("div");
    card.className = "mcard";
    card.dataset.model = m.version;
    card.innerHTML = `<div class="mcard-head"><span class="mcard-name">${m.version}</span>` +
      (m.version === state.manifest.default ? `<span class="badge">default</span>` : "") + `</div>` +
      `<div class="mcard-metrics">` +
      `<div><b>${pct(m.test_ferplus)}</b><span>FER+ test</span></div>` +
      `<div><b>${pct(m.test_rafdb)}</b><span>RAF-DB test</span></div>` +
      `<div><b>${pct(m.webcam)}</b><span>webcam sim</span></div>` +
      `<div><b>${m.train_images ? (m.train_images / 1000).toFixed(1) + "k" : "—"}</b><span>train images</span></div>` +
      `</div>`;
    cards.append(card);
  }
  for (const c of state.manifest.charts) {
    const fig = document.createElement("figure");
    fig.innerHTML = `<img src="${c.file}" alt="${c.caption}" loading="lazy"><figcaption>${c.caption}</figcaption>`;
    charts.append(fig);
  }
}

// ---------------------------------------------------------------- start

async function init() {
  state.manifest = await (await fetch("models/manifest.json")).json();
  buildModelChips();
  buildBars();
  renderPanel(null, 0);
  renderAbout();
  const vision = await FilesetResolver.forVisionTasks(MEDIAPIPE);
  const options = (runningMode) => ({
    baseOptions: { modelAssetPath: DETECTOR_MODEL, delegate: "GPU" }, runningMode, minDetectionConfidence: MIN_SCORE,
  });
  state.detectors.video = await FaceDetector.createFromOptions(vision, options("VIDEO"));
  state.detectors.image = await FaceDetector.createFromOptions(vision, options("IMAGE"));
  await selectModel(state.manifest.default);
  ui.camera.disabled = false;
}

// Used by games.js (camera control) and by tests (photo pipeline on an image URL).
window.emotionDetecter = {
  state, fitSimilarity, extractFace, selectModel,
  startCamera: (target) => startCamera(target),
  stopCamera: () => stopCamera(),
  get stream() { return state.stream; },
  get ready() { return !ui.camera.disabled; },
  async photoFromUrl(url) {
    state.mode = "photo";
    ui.video.hidden = true;
    ui.still.hidden = false;
    ui.empty.hidden = true;
    await runPhoto(await createImageBitmap(await (await fetch(url)).blob()));
    return state.tracks.map((t) => t.probs);
  },
};

init().catch((err) => {
  ui.stats.textContent = `error: ${err.message}`;
  console.error(err);
});
