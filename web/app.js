// emotionDetecter live: face detection (MediaPipe) -> alignment -> emotion model (ONNX Runtime Web).
// Everything runs in the browser; no frame ever leaves the device.
import { FaceDetector, FaceLandmarker, FilesetResolver } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.1.0/vision_bundle.mjs";
import * as ort from "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/ort.all.min.mjs";

const MEDIAPIPE = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.1.0/wasm";
const DETECTOR_MODEL =
  "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite";
ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/";

const SUPERSAMPLE = 4;     // faces are cut out at 4x and box-averaged down, like a camera sensor would
const SMOOTH_MS = 300;     // time constant of the probability smoothing
const TRACK_TTL_MS = 600;  // forget a face that has not been seen for this long
const MIN_SCORE = 0.5;     // face detector confidence
const MAX_FACES = 6;
const MIN_FACE_PX = 28;    // smaller faces carry too little detail to read an expression
const VERIFY_MS = 1000;    // re-check each tracked face with the landmark model this often
const MIN_HITS = 2;        // a live face must appear in 2 frames before it gets a label
const UNSURE_BELOW = 0.4;  // below this confidence the app says "unsure" instead of guessing
const MAX_YAW = 0.55;      // heads turned further than this are too sideways to judge
const LANDMARKER_MODEL =
  "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task";

const $ = (id) => document.getElementById(id);
const ui = {
  video: $("video"), still: $("still"), overlay: $("overlay"), viewport: $("viewport"), empty: $("empty"),
  camera: $("camera"), file: $("file"), smooth: $("smooth"), showCrops: $("showCrops"), stats: $("stats"),
  models: $("models"), verdict: $("verdict"), bars: $("bars"), crops: $("crops"), cropsRow: $("cropsRow"),
  people: $("people"),
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

// ---------------------------------------------------------------- finding real faces

const iou = (p, q) => {
  const x1 = Math.max(p[0], q[0]), y1 = Math.max(p[1], q[1]);
  const x2 = Math.min(p[0] + p[2], q[0] + q[2]), y2 = Math.min(p[1] + p[3], q[1] + q[3]);
  const inter = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
  return inter / (p[2] * p[3] + q[2] * q[3] - inter);
};

// Same rule as src/align.py plausible(): the keypoints must be laid out like a face.
function plausible(kp, box) {
  const [re, le, nose, mouth] = kp;
  const eye = Math.hypot(le[0] - re[0], le[1] - re[1]);
  const mid = [(le[0] + re[0]) / 2, (le[1] + re[1]) / 2];
  const size = Math.max(box[2], box[3]);
  return eye > 0.12 * size && eye < 0.9 * size
    && Math.hypot(mouth[0] - mid[0], mouth[1] - mid[1]) > 0.4 * eye
    && Math.hypot(nose[0] - mid[0], nose[1] - mid[1]) < 1.5 * eye;
}

function toCandidates(detections, width, height) {
  return detections
    .filter((d) => d.categories[0].score >= MIN_SCORE && d.keypoints.length >= 4)
    .map((d) => ({
      box: [d.boundingBox.originX, d.boundingBox.originY, d.boundingBox.width, d.boundingBox.height],
      kp: d.keypoints.slice(0, 4).map((k) => [k.x * width, k.y * height]),
      score: d.categories[0].score,
    }))
    .filter((f) => f.box[2] >= MIN_FACE_PX && plausible(f.kp, f.box));
}

// Second opinion: cut out the area around a candidate and ask MediaPipe's 478-point face
// landmark model whether there really is a face there. Posters, toys, patterns, ears and
// other look-alikes usually fail this. Also estimates how far the head is turned (yaw).
const verifyCanvas = document.createElement("canvas");
verifyCanvas.width = verifyCanvas.height = 256;
const verifyCtx = verifyCanvas.getContext("2d", { willReadFrequently: true });

function verifyFace(source, box) {
  const side = 2.2 * Math.max(box[2], box[3]);
  const cx = box[0] + box[2] / 2, cy = box[1] + box[3] / 2;
  verifyCtx.setTransform(1, 0, 0, 1, 0, 0);
  verifyCtx.fillStyle = "#808080";
  verifyCtx.fillRect(0, 0, 256, 256);
  verifyCtx.drawImage(source, cx - side / 2, cy - side / 2, side, side, 0, 0, 256, 256);
  const result = state.landmarker.detect(verifyCanvas);
  for (const lm of result.faceLandmarks) {
    const xs = lm.map((p) => p.x), ys = lm.map((p) => p.y);
    const w = Math.max(...xs) - Math.min(...xs), mx = (Math.max(...xs) + Math.min(...xs)) / 2;
    const my = (Math.max(...ys) + Math.min(...ys)) / 2;
    // the face we were asked about sits in the middle of the cut-out at about 1/2.2 of its size
    if (Math.abs(mx - 0.5) < 0.2 && Math.abs(my - 0.5) < 0.22 && w > 0.2 && w < 0.85) {
      const nose = lm[1], left = lm[263], right = lm[33];
      const eyes = Math.hypot(left.x - right.x, left.y - right.y);
      return { ok: true, yaw: (nose.x - (left.x + right.x) / 2) / Math.max(eyes, 1e-6) };
    }
  }
  return { ok: false, yaw: 0 };
}

// Third opinion: the face gate, a small CNN trained to tell human faces from animals,
// cartoons and emoji (scripts/train_gate.py). It looks at the same aligned crop as the
// emotion model. Without web/models/gate.onnx this check is skipped.
async function gatePasses(source, faces) {
  if (!state.gate || !faces.length) return faces.map(() => true);
  const g = state.gate, S = g.size, plane = S * S;
  const data = new Float32Array(faces.length * plane);
  faces.forEach((f, i) => {
    const crop = extractFace(source, f.kp, g);
    for (let j = 0; j < plane; j++) data[i * plane + j] = (crop[j] / 255 - g.mean) / g.std;
  });
  const out = await g.session.run({ input: new ort.Tensor("float32", data, [faces.length, 1, S, S]) });
  return Array.from(out.logit.data, (z) => 1 / (1 + Math.exp(-z)) >= g.threshold);
}

// The detector looks at a 128 x 128 version of the picture, so a face that is small in a big
// frame shrinks to a few pixels and is missed. Pictures are therefore also scanned in
// overlapping square tiles at two scales (half and a quarter of the short side), where those
// faces appear large. Used for photos and video analysis, where accuracy beats speed.
function detectTiles(canvas, detector, scales = [0.5, 0.28]) {
  const out = [...detector.detect(canvas).detections];
  const W = canvas.width, H = canvas.height;
  const tile = document.createElement("canvas");
  const ctx = tile.getContext("2d");
  for (const frac of scales) {
    const t = Math.round(Math.min(W, H) * frac);
    if (t < 140) continue;  // the tile itself would be tiny: nothing to gain
    tile.width = tile.height = t;
    const step = Math.round(t * 0.66);
    const xs = [], ys = [];
    for (let x = 0; x + t < W; x += step) xs.push(x);
    for (let y = 0; y + t < H; y += step) ys.push(y);
    xs.push(W - t);
    ys.push(H - t);
    for (const oy of ys) {
      for (const ox of xs) {
        ctx.drawImage(canvas, ox, oy, t, t, 0, 0, t, t);
        for (const d of detector.detect(tile).detections) {
          out.push({
            ...d,
            boundingBox: { ...d.boundingBox, originX: d.boundingBox.originX + ox, originY: d.boundingBox.originY + oy },
            keypoints: d.keypoints.map((k) => ({ ...k, x: (k.x * t + ox) / W, y: (k.y * t + oy) / H })),
          });
        }
      }
    }
  }
  // merge duplicates found by several scans, keeping the most confident one
  out.sort((p, q) => q.categories[0].score - p.categories[0].score);
  const kept = [];
  const box = (d) => [d.boundingBox.originX, d.boundingBox.originY, d.boundingBox.width, d.boundingBox.height];
  for (const d of out) if (!kept.some((k) => iou(box(k), box(d)) > 0.35)) kept.push(d);
  return kept;
}

// Photos: faces filling the whole picture are hard for the detector, so retry on a padded copy.
function detectInImage(canvas, scales) {
  const found = detectTiles(canvas, state.detectors.image, scales);
  if (found.length) return found;
  const pad = Math.round(0.25 * Math.max(canvas.width, canvas.height));
  const padded = document.createElement("canvas");
  padded.width = canvas.width + 2 * pad;
  padded.height = canvas.height + 2 * pad;
  const ctx = padded.getContext("2d");
  ctx.fillStyle = "#808080";
  ctx.fillRect(0, 0, padded.width, padded.height);
  ctx.drawImage(canvas, pad, pad);
  return state.detectors.image.detect(padded).detections.map((d) => ({
    ...d,
    boundingBox: { ...d.boundingBox, originX: d.boundingBox.originX - pad, originY: d.boundingBox.originY - pad },
    keypoints: d.keypoints.map((k) => ({
      ...k,
      x: (k.x * padded.width - pad) / canvas.width,
      y: (k.y * padded.height - pad) / canvas.height,
    })),
  }));
}

const isUnsure = (probs, yaw) => Math.max(...probs) < UNSURE_BELOW || Math.abs(yaw) > MAX_YAW;

// ---------------------------------------------------------------- tracking + smoothing

// Candidates are matched to the people already being tracked; each person keeps a running
// average of their probabilities so a single noisy frame cannot flip the label.
function assignTracks(faces, now) {
  const used = new Set();
  for (const face of faces) {
    let best = null, bestIou = 0.3;
    for (const t of state.tracks) {
      const o = iou(t.box, face.box);
      if (!used.has(t) && o > bestIou) { best = t; bestIou = o; }
    }
    if (!best) {
      best = { id: state.nextId++, hits: 0, probs: null };
      state.tracks.push(best);
    }
    Object.assign(best, { box: face.box, hits: best.hits + 1 });
    used.add(best);
    face.track = best;
  }
}

// ---------------------------------------------------------------- pipeline

async function processFrame(source, width, height, detections, mirrored, smooth, live = true) {
  const now = performance.now();
  const candidates = toCandidates(detections, width, height);
  assignTracks(candidates, now);
  const toVerify = candidates.filter((f) => f.track.verifiedAt === undefined || now - f.track.verifiedAt > VERIFY_MS);
  for (const f of toVerify) Object.assign(f.track, verifyFace(source, f.box), { verifiedAt: now });
  const passed = toVerify.filter((f) => f.track.ok);
  const gate = await gatePasses(source, passed);
  passed.forEach((f, i) => { f.track.ok = gate[i]; });
  for (const f of toVerify) f.track.verified = f.track.ok;
  // shown = verified faces; live faces must also have been seen in a couple of frames
  const faces = candidates
    .filter((f) => f.track.verified && (!live || f.track.hits >= MIN_HITS))
    .sort((p, q) => q.box[2] * q.box[3] - p.box[2] * p.box[3])
    .slice(0, MAX_FACES);
  state.rejected = candidates.filter((f) => !f.track.verified).length;

  const model = state.model;
  let crops = [], mainProbs = null;
  if (faces.length) {
    crops = faces.map((f) => extractFace(source, f.kp, model));
    const probs = await classify(crops, model);
    if (model !== state.model) return;  // the user switched models meanwhile
    faces.forEach((f, i) => {
      const t = f.track;
      const alpha = smooth && t.probs ? 1 - Math.exp(-(now - t.seen) / SMOOTH_MS) : 1;
      t.probs = t.probs ? t.probs.map((p, k) => p + alpha * (probs[i][k] - p)) : probs[i].slice();
      t.seen = now;
      t.unsure = isUnsure(t.probs, t.yaw);
    });
    mainProbs = probs[0];  // largest face, unsmoothed: games need the fastest reaction
  }
  for (const f of candidates) f.track.seenAny = now;
  state.tracks = state.tracks.filter((t) => now - (t.seenAny ?? now) < TRACK_TTL_MS);
  state.faces = faces.length;
  if (state.mode === "camera") {
    window.dispatchEvent(new CustomEvent("emotion", { detail: {
      probs: mainProbs, smoothed: faces[0]?.track.probs ?? null, box: faces[0]?.box ?? null,
      width, height, source, emotions: state.manifest.emotions, emoji: state.manifest.emoji,
    } }));
  }
  if (source === ui.video || source === ui.still) drawOverlay(faces, width, height, mirrored);
  renderPanel(faces[0]?.track, faces.length);
  renderPeople(faces);
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

async function runPhoto(bitmap = state.photo) {
  if (!bitmap) return;
  state.photo = bitmap;
  const c = ui.still;
  c.width = bitmap.width;
  c.height = bitmap.height;
  c.getContext("2d").drawImage(bitmap, 0, 0);
  setAspect(bitmap.width, bitmap.height);
  state.tracks = [];
  await processFrame(c, c.width, c.height, detectInImage(c), false, false, false);
}

// For video analysis (video.js): every verified face in one still frame, with its probabilities.
// thorough = also scan small tiles (finds far-away faces, ~4x slower).
async function analyzeStill(canvas, { thorough = true } = {}) {
  const scales = thorough ? [0.5, 0.28] : [0.5];
  const candidates = toCandidates(detectInImage(canvas, scales), canvas.width, canvas.height);
  let faces = [];
  for (const f of candidates) {
    const v = verifyFace(canvas, f.box);
    if (v.ok) faces.push({ ...f, yaw: v.yaw });
  }
  const gate = await gatePasses(canvas, faces);
  faces = faces.filter((_, i) => gate[i]);
  const rejected = candidates.length - faces.length;
  if (!faces.length) return { faces, rejected };
  const probs = await classify(faces.map((f) => extractFace(canvas, f.kp, state.model)), state.model);
  faces.forEach((f, i) => { f.probs = probs[i]; f.unsure = isUnsure(probs[i], f.yaw); });
  return { faces, rejected };
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
    const who = faces.length > 1 ? `#${personNumber(face.track)} ` : "";
    const label = face.track.unsure ? `${who}❔ unsure` : `${who}${emoji[top]} ${emotions[top]} ${Math.round(p[top] * 100)}%`;
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
  const others = nFaces > 1 ? ` · person #${personNumber(track)}, largest of ${nFaces}` : "";
  const unsure = track.unsure ? " · unsure (head turned or unclear)" : "";
  ui.verdict.innerHTML = `<div class="verdict-emoji">${track.unsure ? "❔" : emoji[top]}</div><div>` +
    `<div class="verdict-label">${emotions[top]}</div>` +
    `<div class="verdict-sub">${Math.round(p[top] * 100)}% confident · ${state.model.version}${others}${unsure}</div></div>`;
}

// Stable, small person numbers (1, 2, 3 …) for the tracks currently on screen.
function personNumber(track) {
  if (!state.personIds) state.personIds = new Map();
  if (!state.personIds.has(track.id)) {
    const used = new Set(state.tracks.filter((t) => state.personIds.has(t.id)).map((t) => state.personIds.get(t.id)));
    let n = 1;
    while (used.has(n)) n++;
    state.personIds.set(track.id, n);
  }
  return state.personIds.get(track.id);
}

// With more than one person on screen, list everyone with their current emotion.
function renderPeople(faces) {
  const { emotions, emoji } = state.manifest;
  ui.people.hidden = faces.length < 2;
  if (faces.length < 2) return;
  ui.people.innerHTML = `<div class="crops-title">Everyone on screen</div>` + faces
    .map((f) => [personNumber(f.track), f.track])
    .sort((p, q) => p[0] - q[0])
    .map(([n, t]) => {
      const top = t.probs.indexOf(Math.max(...t.probs));
      return `<div class="person"><span class="person-n">#${n}</span>` +
        `<span class="person-e">${t.unsure ? "❔" : emoji[top]}</span>` +
        `<span class="person-l">${t.unsure ? "unsure" : emotions[top]}</span>` +
        `<span class="person-v">${t.unsure ? "" : Math.round(t.probs[top] * 100) + "%"}</span></div>`;
    }).join("");
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
  if (state.rejected) parts.push(`${state.rejected} ignored (not a face)`);
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
  state.landmarker = await FaceLandmarker.createFromOptions(vision, {
    baseOptions: { modelAssetPath: LANDMARKER_MODEL, delegate: "GPU" },
    runningMode: "IMAGE", numFaces: 3, minFaceDetectionConfidence: 0.5, minFacePresenceConfidence: 0.5,
  });
  if (state.manifest.gate) {
    try {
      const session = await ort.InferenceSession.create(state.manifest.gate.file, {
        executionProviders: navigator.gpu ? ["webgpu", "wasm"] : ["wasm"] });
      state.gate = { ...state.manifest.gate, session };
    } catch (err) {
      console.warn("face gate not loaded, continuing without it", err);
    }
  }
  await selectModel(state.manifest.default);
  ui.camera.disabled = false;
}

// Used by games.js (camera control) and by tests (photo pipeline on an image URL).
window.emotionDetecter = {
  state, fitSimilarity, extractFace, selectModel,
  startCamera: (target) => startCamera(target),
  analyzeStill,
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
