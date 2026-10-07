// Two small games driven by the live emotion stream from app.js ("emotion" events).
//   Reaction Dino — a cube wearing your face: smile to jump, look surprised to duck.
//   Actor Game    — act the emotion on screen; the model scores how convincing it was.
// While a game runs, the camera is moved from the main stage to the game's own camera view.

const HAPPY = "happy", SURPRISE = "surprise";
const $ = (id) => document.getElementById(id);

// ---------------------------------------------------------------- shared: emotion stream + camera

const live = { probs: null, smoothed: null, box: null, width: 1, height: 1, source: null, emotions: [], emoji: [], at: 0 };
window.addEventListener("emotion", (e) => Object.assign(live, e.detail, { at: performance.now() }));
window.addEventListener("camerastop", () => Object.assign(live, { probs: null, box: null, source: null }));

const prob = (name) => (live.probs ? live.probs[live.emotions.indexOf(name)] : 0);
const faceVisible = () => Boolean(live.probs) && performance.now() - live.at < 600;

async function useCamera(video) {
  const ed = window.emotionDetecter;
  for (let i = 0; i < 100 && !ed?.ready; i++) await new Promise((r) => setTimeout(r, 100));
  await ed.startCamera(video);
  return Boolean(ed.stream);
}

function loadBest(key) {
  try { return Number(localStorage.getItem(key)) || 0; } catch { return 0; }
}
function saveBest(key, value) {
  try { localStorage.setItem(key, String(value)); } catch { /* storage unavailable */ }
}

function fitCanvas(canvas, width, height) {
  const dpr = window.devicePixelRatio || 1;
  if (canvas.width !== Math.round(width * dpr)) canvas.width = Math.round(width * dpr);
  if (canvas.height !== Math.round(height * dpr)) canvas.height = Math.round(height * dpr);
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return ctx;
}

const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

// A still copy of the current camera frame and what the model saw in it.
function snapshot() {
  const v = live.source;
  if (!v || v.readyState < 2) return null;
  const image = document.createElement("canvas");
  image.width = v.videoWidth;
  image.height = v.videoHeight;
  image.getContext("2d").drawImage(v, 0, 0);
  return { image, box: faceVisible() ? live.box?.slice() : null, width: live.width, height: live.height,
    probs: (live.smoothed || live.probs)?.slice() };
}

const liveView = () => ({ image: null, box: faceVisible() ? live.box : null, width: live.width, height: live.height,
  probs: live.smoothed || live.probs });

// Game camera view: mirrored video with the face box and the current emotion, like the main stage.
// With a frozen snapshot it shows that still frame instead (the moment the player lost).
function drawCamOverlay(canvas, video, frozen = null) {
  const rect = canvas.getBoundingClientRect();
  const ctx = fitCanvas(canvas, rect.width, rect.height);
  ctx.clearRect(0, 0, rect.width, rect.height);
  if (!frozen && live.source !== video) return;
  const view = frozen || liveView();
  // the camera view uses object-fit: cover, so scale to fill and crop the overflow
  const scale = Math.max(rect.width / view.width, rect.height / view.height);
  const ox = (rect.width - view.width * scale) / 2, oy = (rect.height - view.height * scale) / 2;
  if (frozen) {
    ctx.save();
    ctx.translate(rect.width, 0);
    ctx.scale(-1, 1);
    ctx.drawImage(frozen.image, ox, oy, view.width * scale, view.height * scale);
    ctx.restore();
  }
  if (!view.box || !view.probs) return;
  let [x, y, w, h] = view.box;
  x = view.width - x - w;  // mirrored
  x = ox + x * scale; y = oy + y * scale; w *= scale; h *= scale;
  ctx.lineWidth = 2.5;
  const grad = ctx.createLinearGradient(x, y, x + w, y + h);
  grad.addColorStop(0, "#8b5cf6");
  grad.addColorStop(1, "#06b6d4");
  ctx.strokeStyle = grad;
  ctx.beginPath(); ctx.roundRect(x, y, w, h, 10); ctx.stroke();
  const p = view.probs, top = p.indexOf(Math.max(...p));
  const label = `${live.emoji[top]} ${live.emotions[top]} ${Math.round(p[top] * 100)}%`;
  ctx.font = "600 12px -apple-system, Inter, sans-serif";
  const tw = ctx.measureText(label).width + 14;
  const ly = y - 26 >= 0 ? y - 26 : y + 4;
  ctx.fillStyle = "rgba(15, 23, 42, 0.82)";
  ctx.beginPath(); ctx.roundRect(x, ly, tw, 22, 999); ctx.fill();
  ctx.fillStyle = "#fff";
  ctx.textBaseline = "middle";
  ctx.fillText(label, x + 7, ly + 11.5);
}

// ---------------------------------------------------------------- Reaction Dino (a cube with your face)

const W = 900, H = 260, GROUND = 214, CUBE = 56;
const GRAVITY = 2400, JUMP_SPEED = 860, THRESHOLD = 0.5;

const dino = {
  ctx: null, state: "ready", score: 0, best: loadBest("dino-best"), speed: 0,
  y: 0, vy: 0, ducking: false, obstacles: [], nextSpawn: 0, last: 0, wasSmiling: false, keys: {},
};

function dinoReset(state) {
  Object.assign(dino, { state, frozen: null, score: 0, speed: 380, y: 0, vy: 0, ducking: false,
    obstacles: [], nextSpawn: 1.2, wasSmiling: true });  // a smile held at start does not jump
}

function cubeBox() {
  const ducked = dino.ducking && dino.y === 0;
  const h = ducked ? 32 : CUBE, w = ducked ? 66 : CUBE;
  return { x: 70, y: GROUND - dino.y - h, w, h };
}

function spawnObstacle() {
  const bird = dino.score > 120 && Math.random() < 0.38;
  dino.obstacles.push(bird
    ? { kind: "bird", x: W + 20, y: GROUND - 66, w: 44, h: 26 }
    : { kind: "cactus", x: W + 20, y: GROUND - 46, w: 26 + Math.random() * 14, h: 46 });
}

function dinoStep(dt) {
  const smiling = prob(HAPPY) > THRESHOLD || dino.keys.jump;
  const surprised = prob(SURPRISE) > THRESHOLD || dino.keys.duck;
  if (dino.state !== "playing") return;
  if (smiling && !dino.wasSmiling && dino.y === 0) dino.vy = JUMP_SPEED;  // each new smile = one jump
  dino.wasSmiling = smiling;
  dino.ducking = surprised;

  dino.vy -= (dino.ducking ? GRAVITY * 1.8 : GRAVITY) * dt;  // ducking in the air drops faster
  dino.y = Math.max(0, dino.y + dino.vy * dt);
  if (dino.y === 0) dino.vy = Math.max(0, dino.vy);

  dino.speed += 9 * dt;
  dino.score += dino.speed * dt * 0.05;
  dino.nextSpawn -= dt;
  if (dino.nextSpawn <= 0) {
    spawnObstacle();
    dino.nextSpawn = 0.85 + Math.random() * 0.9 * (420 / dino.speed);
  }
  for (const o of dino.obstacles) o.x -= dino.speed * dt;
  dino.obstacles = dino.obstacles.filter((o) => o.x + o.w > -10);

  const c = cubeBox();
  const hit = dino.obstacles.some((o) => c.x + 4 < o.x + o.w && c.x + c.w - 4 > o.x && c.y + 4 < o.y + o.h && c.y + c.h > o.y + 4);
  if (hit) {
    dino.state = "over";
    dino.frozen = snapshot();  // keep the player's face exactly as it was when they lost
    window.emotionDetecter.stopCamera();
    if (Math.floor(dino.score) > dino.best) {
      dino.best = Math.floor(dino.score);
      saveBest("dino-best", dino.best);
    }
    $("dinoStart").textContent = "Restart";
  }
}

// The cube's face: the player's face cut out of the camera (mirrored), or an emoji without a camera.
function drawCube(ctx, c) {
  ctx.save();
  ctx.beginPath();
  ctx.roundRect(c.x, c.y, c.w, c.h, 10);
  ctx.clip();
  const frozen = dino.state === "over" ? dino.frozen : null;
  const image = frozen ? frozen.image : live.source;
  const box = frozen ? frozen.box : (faceVisible() ? live.box : null);
  if (image && box && (frozen || image.readyState >= 2)) {
    const [bx, by, bw, bh] = box;
    const side = Math.max(bw, bh) * 1.25, cx = bx + bw / 2, cy = by + bh / 2;
    ctx.translate(c.x + c.w, c.y);
    ctx.scale(-1, 1);
    ctx.drawImage(image, cx - side / 2, cy - side / 2, side, side, 0, 0, c.w, c.h);
  } else {
    ctx.fillStyle = css("--card-2");
    ctx.fillRect(c.x, c.y, c.w, c.h);
    ctx.font = `${Math.min(c.w, c.h) * 0.7}px serif`;
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText("🙂", c.x + c.w / 2, c.y + c.h / 2 + 2);
  }
  ctx.restore();
  const grad = ctx.createLinearGradient(c.x, c.y, c.x + c.w, c.y + c.h);
  grad.addColorStop(0, "#8b5cf6");
  grad.addColorStop(1, "#06b6d4");
  ctx.lineWidth = 3;
  ctx.strokeStyle = grad;
  ctx.beginPath();
  ctx.roundRect(c.x, c.y, c.w, c.h, 10);
  ctx.stroke();
}

function dinoDraw() {
  const rect = $("dinoCanvas").getBoundingClientRect();
  const ctx = fitCanvas($("dinoCanvas"), rect.width, rect.height);
  ctx.scale(rect.width / W, rect.height / H);  // draw in a fixed 900 x 260 world
  const text = css("--text"), muted = css("--muted");
  ctx.clearRect(0, 0, W, H);
  ctx.strokeStyle = muted;
  ctx.globalAlpha = 0.5;
  ctx.beginPath(); ctx.moveTo(0, GROUND + 1); ctx.lineTo(W, GROUND + 1); ctx.stroke();
  ctx.globalAlpha = 1;

  drawCube(ctx, cubeBox());

  ctx.textAlign = "left";
  ctx.textBaseline = "bottom";
  for (const o of dino.obstacles) {
    ctx.font = o.kind === "bird" ? "34px serif" : `${o.h + 6}px serif`;
    ctx.fillText(o.kind === "bird" ? "🦅" : "🌵", o.x - 4, o.y + o.h + 4);
  }

  ctx.fillStyle = text;
  ctx.font = "600 15px ui-monospace, Menlo, monospace";
  ctx.textAlign = "right";
  ctx.textBaseline = "top";
  ctx.fillText(`${String(Math.floor(dino.score)).padStart(5, "0")}   best ${String(dino.best).padStart(5, "0")}`, W - 16, 14);
  if (dino.state !== "playing") {
    ctx.textAlign = "center";
    ctx.font = "700 22px -apple-system, Inter, sans-serif";
    ctx.fillText(dino.state === "over" ? "Game over" : "Press Start", W / 2, 72);
    ctx.font = "15px -apple-system, Inter, sans-serif";
    ctx.fillStyle = muted;
    ctx.fillText(dino.state === "over" ? `Score ${Math.floor(dino.score)} · press Restart`
      : "😄 smile = jump    😮 surprise = duck    (or ↑ / ↓ keys)", W / 2, 104);
  }
}

function meter(id, value) {
  $(id).style.width = `${Math.round(value * 100)}%`;
  $(id).parentElement.classList.toggle("on", value > THRESHOLD);
}

async function dinoStart() {
  $("dinoStart").disabled = true;
  const ok = await useCamera($("dinoCam"));
  $("dinoStart").disabled = false;
  $("dinoHint").textContent = ok ? "" : "Allow camera access to play with your face (↑ / ↓ keys also work).";
  dinoReset("playing");
  $("dinoStart").textContent = "Restart";
}

// ---------------------------------------------------------------- Actor Game

const ROUNDS = 5, COUNTDOWN = 3, PERFORM = 3, WINDOW = 6;
const EMOJI = { angry: "😠", disgust: "🤢", fear: "😨", happy: "😄", neutral: "😐", sad: "😢", surprise: "😮" };
const actor = { run: 0, best: loadBest("actor-best") };
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

function rating(avg) {
  if (avg >= 85) return "🏆 Oscar-worthy";
  if (avg >= 65) return "🎬 Leading role";
  if (avg >= 45) return "🎭 Supporting role";
  return "🎟️ Audience member";
}

// Returns the round's score, or null if a restart cancelled this run.
async function actorRound(run, target, index) {
  $("actorStage").innerHTML = `<div class="actor-round">Round ${index + 1} of ${ROUNDS}</div>` +
    `<div class="actor-emoji">${EMOJI[target]}</div><div class="actor-target">${target}</div>` +
    `<div class="actor-count" id="actorCount"></div><div class="actor-meter"><span id="actorMeter"></span></div>`;
  for (let c = COUNTDOWN; c > 0; c--) {
    if (run !== actor.run) return null;
    $("actorCount").textContent = `Get ready… ${c}`;
    await sleep(1000);
  }
  // Score = the most convincing moment: highest short-window average of the target probability.
  const recent = [];
  let best = 0;
  const end = performance.now() + PERFORM * 1000;
  while (performance.now() < end) {
    if (run !== actor.run) return null;
    $("actorCount").textContent = `Act! ${Math.ceil((end - performance.now()) / 1000)}`;
    recent.push(faceVisible() ? prob(target) : 0);
    if (recent.length > WINDOW) recent.shift();
    const avg = recent.reduce((a, b) => a + b, 0) / recent.length;
    best = Math.max(best, avg);
    $("actorMeter").style.width = `${Math.round(avg * 100)}%`;
    await sleep(80);
  }
  return Math.round(best * 100);
}

async function actorPlay() {
  const run = ++actor.run;  // pressing Restart starts a new run; the old one stops at its next check
  $("actorStart").textContent = "Restart";
  if (!(await useCamera($("actorCam")))) {
    $("actorStage").innerHTML = `<div class="actor-emoji">📷</div><div class="actor-target">Camera needed</div>` +
      `<div class="actor-best">Allow camera access in the address bar, then press Start again.</div>`;
    $("actorStart").textContent = "Start";
    return;
  }
  const pool = Object.keys(EMOJI);
  const targets = Array.from({ length: ROUNDS }, () => pool.splice(Math.floor(Math.random() * pool.length), 1)[0]);
  const scores = [];
  for (const [i, t] of targets.entries()) {
    const s = await actorRound(run, t, i);
    if (s === null) return;
    scores.push(s);
  }
  const total = scores.reduce((a, b) => a + b, 0);
  if (total > actor.best) {
    actor.best = total;
    saveBest("actor-best", total);
  }
  $("actorStage").innerHTML = `<div class="actor-result">${rating(total / ROUNDS)}</div>` +
    `<div class="actor-total">${total} <span>/ ${ROUNDS * 100}</span></div>` +
    `<ul class="actor-scores">${targets.map((t, i) => `<li><span>${EMOJI[t]} ${t}</span>` +
      `<span class="bar"><span style="width:${scores[i]}%"></span></span><b>${scores[i]}</b></li>`).join("")}</ul>` +
    `<div class="actor-best">Best: ${actor.best} / ${ROUNDS * 100}</div>`;
  $("actorStart").textContent = "Play again";
}

// ---------------------------------------------------------------- loop, tabs, controls

function frame(now) {
  const dt = Math.min(0.05, (now - (dino.last || now)) / 1000);
  dino.last = now;
  if (!document.hidden) {
    if ($("game-dino").classList.contains("active")) {
      dinoStep(dt);
      dinoDraw();
      meter("dinoSmile", prob(HAPPY));
      meter("dinoSurprise", prob(SURPRISE));
      drawCamOverlay($("dinoCamOverlay"), $("dinoCam"), dino.state === "over" ? dino.frozen : null);
      if (dino.state === "over") $("dinoHint").textContent = "Camera off · press Restart to play again";
      else if (live.source === $("dinoCam")) $("dinoHint").textContent = faceVisible() ? "" : "No face — look at the camera";
    } else {
      drawCamOverlay($("actorCamOverlay"), $("actorCam"));
    }
  }
  requestAnimationFrame(frame);
}

function showGame(name) {
  for (const tab of document.querySelectorAll(".game-tab")) tab.setAttribute("aria-selected", String(tab.dataset.game === name));
  for (const panel of document.querySelectorAll(".game")) panel.classList.toggle("active", panel.id === `game-${name}`);
  // keep the camera with the game on screen if a game already has it
  const cams = { dino: $("dinoCam"), actor: $("actorCam") };
  if (live.source && Object.values(cams).includes(live.source) && live.source !== cams[name]) {
    window.emotionDetecter.startCamera(cams[name]);
  }
  if (name !== "actor") actor.run++;  // leaving the Actor Game stops a running round
}

document.querySelectorAll(".game-tab").forEach((tab) => tab.addEventListener("click", () => showGame(tab.dataset.game)));
$("dinoStart").addEventListener("click", dinoStart);
$("actorStart").addEventListener("click", actorPlay);

const keyMap = { ArrowUp: "jump", " ": "jump", ArrowDown: "duck" };
window.addEventListener("keydown", (e) => {
  if (!keyMap[e.key] || !$("game-dino").classList.contains("active") || dino.state !== "playing") return;
  e.preventDefault();
  dino.keys[keyMap[e.key]] = true;
});
window.addEventListener("keyup", (e) => { if (keyMap[e.key]) dino.keys[keyMap[e.key]] = false; });

window.emotionGames = { dino, actor, live, step: dinoStep, draw: dinoDraw };  // test hook

dinoReset("ready");
$("actorBest").textContent = actor.best ? `Best: ${actor.best} / ${ROUNDS * 100}` : "";
showGame("dino");
requestAnimationFrame(frame);
