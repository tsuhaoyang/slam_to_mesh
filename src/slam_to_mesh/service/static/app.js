// slam_to_mesh multi-view UI.
// Checkboxes choose which representations to show; each gets its own synced
// viewer pane. Mesh reps decimate (quad = re-remesh, triangle = QEM); point
// clouds downsample (voxel). All independent. See docs/spec_flexible_io.md.

import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";

const $ = (id) => document.getElementById(id);
const loader = new GLTFLoader();

// ---- representation registry ----------------------------------------------
// kind: how to render (surface | wireframe | points); mesh: is it a mesh rep.
const REPS = {
  triangle:      { label: "Triangle mesh",        kind: "surface",   mesh: true },
  triangle_dec:  { label: "Triangle — decimated",  kind: "surface",   mesh: true },
  quad:          { label: "Quad mesh",             kind: "wireframe", mesh: true },
  quad_dec:      { label: "Quad — decimated",      kind: "wireframe", mesh: true },
  pointcloud:    { label: "Point cloud",           kind: "points",    mesh: false },
  pointcloud_ds: { label: "Point cloud — downsample", kind: "points", mesh: false },
};

// ---- viewer factory --------------------------------------------------------
function makeViewer(el) {
  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(window.devicePixelRatio);
  el.appendChild(renderer.domElement);
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x2a2a2e);
  const camera = new THREE.PerspectiveCamera(45, 1, 0.001, 100000);
  camera.position.set(0, 0, 3);
  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  scene.add(new THREE.HemisphereLight(0xffffff, 0x444455, 1.0));
  const dir = new THREE.DirectionalLight(0xffffff, 1.2);
  dir.position.set(1, 2, 3);
  scene.add(dir);
  let content = null;

  function resize() {
    const w = el.clientWidth, h = el.clientHeight;
    if (!w || !h) return;
    renderer.setSize(w, h, false);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
  }
  function frame(obj) {
    const box = new THREE.Box3().setFromObject(obj);
    const size = box.getSize(new THREE.Vector3());
    const c = box.getCenter(new THREE.Vector3());
    obj.position.sub(c);
    const d = Math.max(size.x, size.y, size.z) || 1;
    camera.position.set(0, 0, d * 2.2);
    camera.near = d / 100; camera.far = d * 100;
    camera.updateProjectionMatrix();
    controls.target.set(0, 0, 0); controls.update();
  }
  function clear() {
    if (content) {
      scene.remove(content);
      content.traverse?.((o) => { o.geometry?.dispose?.(); o.material?.dispose?.(); });
      content = null;
    }
  }
  function setObject(obj) { clear(); content = obj; scene.add(obj); frame(obj); }
  function render() { controls.update(); renderer.render(scene, camera); }
  window.addEventListener("resize", resize);
  return { el, scene, camera, controls, resize, render, setObject };
}

// ---- pane management -------------------------------------------------------
let panes = {}; // rep -> {viewer, labelEl, badgeEl}
let rafStarted = false;

function rebuildPanes(reps) {
  // Dispose old panes.
  $("panes").innerHTML = "";
  panes = {};
  const container = $("panes");
  container.classList.toggle("rows-2", reps.length > 2);

  for (const rep of reps) {
    const pane = document.createElement("div");
    pane.className = "viewpane";
    const label = document.createElement("div");
    label.className = "viewlabel";
    label.textContent = REPS[rep].label;
    const badge = document.createElement("span");
    badge.className = "badge";
    label.appendChild(badge);
    const view = document.createElement("div");
    view.className = "viewer";
    pane.appendChild(label); pane.appendChild(view);
    container.appendChild(pane);
    panes[rep] = { viewer: makeViewer(view), badgeEl: badge };
  }
  // Resize after layout settles, then sync cameras among mesh panes.
  setTimeout(() => Object.values(panes).forEach((p) => p.viewer.resize()), 0);
  wireCameraSync();
  if (!rafStarted) { rafStarted = true; animate(); }
}

function animate() {
  requestAnimationFrame(animate);
  for (const p of Object.values(panes)) p.viewer.render();
}

// Sync cameras across all currently-shown panes (drag one → all move).
let syncing = false;
function wireCameraSync() {
  const list = Object.values(panes).map((p) => p.viewer);
  for (const v of list) {
    v.controls.addEventListener("change", () => {
      if (syncing) return;
      syncing = true;
      for (const o of list) {
        if (o === v) continue;
        o.camera.position.copy(v.camera.position);
        o.camera.quaternion.copy(v.camera.quaternion);
        o.controls.target.copy(v.controls.target);
        o.camera.updateProjectionMatrix();
      }
      syncing = false;
    });
  }
}

// ---- gating (Q1) -----------------------------------------------------------
function checkedReps() {
  return [...document.querySelectorAll(".show:checked")].map((c) => c.value);
}

function applyGating() {
  const checked = checkedReps();
  const meshChecked = checked.some((r) => REPS[r].mesh);
  const dsChecked = checked.includes("pointcloud_ds");
  const pcOnly = checked.length === 1 && checked[0] === "pointcloud";
  const hasPC = state.hasPointcloud;

  document.querySelectorAll(".show").forEach((cb) => {
    const v = cb.value;
    if (v === "pointcloud_ds") {
      // Enabled only when point cloud is the sole selection (or already on).
      cb.disabled = !hasPC || (!pcOnly && !cb.checked);
    } else if (REPS[v].mesh) {
      // Mesh reps disabled while downsample mode is active.
      cb.disabled = dsChecked;
    } else if (v === "pointcloud") {
      cb.disabled = !hasPC;
    }
  });

  let note = "";
  if (!hasPC) note = "This job has no point cloud (mesh input).";
  else if (dsChecked) note = "Point-cloud downsample mode: mesh views disabled.";
  else if (meshChecked) note = "‘Point cloud — downsample’ needs point cloud alone.";
  $("gateNote").textContent = note;

  // Contextual sliders.
  $("grpTri").hidden = !checked.includes("triangle_dec");
  $("grpQuad").hidden = !checked.includes("quad_dec");
  $("grpPts").hidden = !checked.includes("pointcloud_ds");
}

// ---- state -----------------------------------------------------------------
const state = { jobId: null, inputFaces: null, hasPointcloud: false, pointCount: null };
const setStatus = (m) => { $("status").textContent = m || ""; };

async function loadJob() {
  state.jobId = $("job").value.trim();
  if (!state.jobId) { setStatus("enter a job id"); return; }
  setStatus("loading job…");
  const r = await fetch(`/jobs/${state.jobId}`);
  if (!r.ok) { setStatus("job not found"); return; }
  const info = await r.json();
  state.inputFaces = info.input_faces ?? info.stages?.ingest?.metrics?.faces ?? null;
  state.hasPointcloud = !!info.has_pointcloud;
  // If it's a point cloud job, default to showing the point cloud.
  if (state.hasPointcloud) {
    document.querySelector('.show[value="pointcloud"]').checked = true;
  }
  applyGating();
  updateSliderLabels();
  refreshPcTools();
  setStatus("loaded");
  apply();
}

// ---- loaders per representation --------------------------------------------
function loadGlbInto(viewer, url) {
  return new Promise((res, rej) =>
    loader.load(url, (g) => { viewer.setObject(g.scene); res(); }, undefined, rej));
}
async function loadWireInto(viewer, url) {
  const r = await fetch(url); if (!r.ok) throw new Error("wire");
  const { positions, edges } = await r.json();
  const geo = new THREE.BufferGeometry();
  geo.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
  geo.setIndex(edges);
  viewer.setObject(new THREE.LineSegments(geo, new THREE.LineBasicMaterial({ color: 0x8fd6a8 })));
}
async function loadPointsInto(viewer, url, body) {
  const r = body
    ? await fetch(url, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) })
    : await fetch(url);
  if (!r.ok) throw new Error("points");
  const data = await r.json();
  const positions = data.positions;
  const geo = new THREE.BufferGeometry();
  geo.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
  const mat = new THREE.PointsMaterial({ color: 0x8fb3ff, size: 0.01, sizeAttenuation: true });
  viewer.setObject(new THREE.Points(geo, mat));
  return positions.length / 3;
}

// ---- slider helpers --------------------------------------------------------
function pctToFaces(pct) {
  return state.inputFaces ? Math.max(200, Math.round(state.inputFaces * pct / 100)) : null;
}
function updateSliderLabels() {
  const tf = pctToFaces(+$("triSlider").value);
  $("triPct").textContent = `${$("triSlider").value}%`;
  $("triFaces").textContent = tf ? `${tf} faces` : "—";
  const qf = pctToFaces(+$("quadSlider").value);
  $("quadPct").textContent = `${$("quadSlider").value}%`;
  $("quadFaces").textContent = qf ? `${qf} faces` : "—";
  $("ptsPct").textContent = `${$("ptsSlider").value}%`;
  const pc = state.pointCount ? Math.max(1, Math.round(state.pointCount * $("ptsSlider").value / 100)) : null;
  $("ptsCount").textContent = pc ? `${pc} pts` : "—";
}

// ---- apply: build the panes for the current selection ----------------------
async function apply() {
  if (!state.jobId) { setStatus("load a job first"); return; }
  const reps = checkedReps();
  if (reps.length === 0) { setStatus("pick something to show"); rebuildPanes([]); return; }
  rebuildPanes(reps);
  setStatus("building…");
  $("apply").disabled = true;
  const jid = state.jobId;
  try {
    for (const rep of reps) {
      const { viewer, badgeEl } = panes[rep];
      if (rep === "triangle") {
        await loadGlbInto(viewer, `/jobs/${jid}/source.glb`);
        badgeEl.textContent = state.inputFaces ? `${state.inputFaces} faces` : "";
      } else if (rep === "triangle_dec") {
        const tf = pctToFaces(+$("triSlider").value);
        const res = await postJson(`/jobs/${jid}/tri-lod`, { target_faces: tf });
        await loadGlbInto(viewer, res.glb_url);
        badgeEl.textContent = `${res.lod.actual_faces} faces`;
      } else if (rep === "quad" || rep === "quad_dec") {
        const pct = rep === "quad_dec" ? +$("quadSlider").value : 100;
        const tf = pctToFaces(pct);
        const res = await postJson(`/jobs/${jid}/lod`, { target_faces: tf });
        const base = `/jobs/${jid}/lod/${res.lod.target_faces}`;
        await loadWireInto(viewer, `${base}/wire.json`);
        badgeEl.textContent = `${res.lod.actual_faces} faces`;
      } else if (rep === "pointcloud") {
        const n = await loadPointsInto(viewer, `/jobs/${jid}/pointcloud.json`);
        state.pointCount = n;
        badgeEl.textContent = `${n} pts`;
      } else if (rep === "pointcloud_ds") {
        const pct = +$("ptsSlider").value;
        const target = state.pointCount ? Math.max(1, Math.round(state.pointCount * pct / 100)) : 2000;
        const n = await loadPointsInto(viewer, `/jobs/${jid}/pointcloud/downsample`, { target_points: target });
        badgeEl.textContent = `${n} pts`;
      }
    }
    updateSliderLabels();
    setStatus("done");
  } catch (e) {
    setStatus("build failed");
  } finally {
    $("apply").disabled = false;
  }
}

async function postJson(url, body) {
  const r = await fetch(url, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
  if (!r.ok) throw new Error(`${url} ${r.status}`);
  return r.json();
}

function refreshPcTools() {
  // Generate offered for mesh jobs without a point cloud; download when present.
  $("genPts").hidden = state.hasPointcloud || !state.jobId;
  $("dlPts").hidden = !state.hasPointcloud;
}

async function generatePoints() {
  if (!state.jobId) return;
  setStatus("generating point cloud…");
  $("genPts").disabled = true;
  try {
    const res = await postJson(`/jobs/${state.jobId}/pointcloud/generate`, { n: 20000 });
    state.hasPointcloud = true;
    state.pointCount = res.stats.sampled_points;
    document.querySelector('.show[value="pointcloud"]').checked = true;
    applyGating();
    refreshPcTools();
    setStatus(`generated ${res.stats.sampled_points} points`);
    apply();
  } catch {
    setStatus("generate failed");
  } finally {
    $("genPts").disabled = false;
  }
}

function downloadPoints() {
  if (!state.jobId) return;
  // Downsampled if that pane is active, else the original.
  const ds = checkedReps().includes("pointcloud_ds");
  const a = document.createElement("a");
  a.href = `/jobs/${state.jobId}/pointcloud/download?downsampled=${ds ? 1 : 0}`;
  a.click();
}

async function uploadJob() {
  const f = $("file").files[0];
  if (!f) { $("upNote").textContent = "choose a file first"; return; }
  const fd = new FormData();
  fd.append("file", f);
  fd.append("target_faces", String(parseInt($("upTarget").value, 10) || 8000));
  fd.append("frames", String(parseInt($("upFrames").value, 10) || 40));
  fd.append("backend", "quadriflow");
  fd.append("formats", "glb,obj");

  $("upload").disabled = true;
  $("upNote").textContent = "uploading…";
  try {
    const r = await fetch("/jobs", { method: "POST", body: fd });
    if (r.status === 503) { $("upNote").textContent = "image input unavailable (TripoSR not installed)"; return; }
    if (r.status === 400) { $("upNote").textContent = "unsupported file format"; return; }
    if (!r.ok) { $("upNote").textContent = `upload failed (${r.status})`; return; }
    const { job_id } = await r.json();
    $("job").value = job_id;
    // Poll until the pipeline completes (photogrammetry can take minutes).
    for (let i = 0; i < 300; i++) {
      await new Promise((res) => setTimeout(res, 2000));
      const s = await (await fetch(`/jobs/${job_id}`)).json();
      $("upNote").textContent = `processing… (${s.status})`;
      if (s.status === "completed") { $("upNote").textContent = `done: ${job_id}`; loadJob(); return; }
      if (s.status === "failed") { $("upNote").textContent = "processing failed"; return; }
    }
    $("upNote").textContent = "still processing — load manually later";
  } catch {
    $("upNote").textContent = "upload error";
  } finally {
    $("upload").disabled = false;
  }
}

// ---- wiring ----------------------------------------------------------------
$("upload").addEventListener("click", uploadJob);
$("load").addEventListener("click", loadJob);
$("apply").addEventListener("click", apply);
$("genPts").addEventListener("click", generatePoints);
$("dlPts").addEventListener("click", downloadPoints);
document.querySelectorAll(".show").forEach((cb) =>
  cb.addEventListener("change", () => { applyGating(); }));
["triSlider", "quadSlider", "ptsSlider"].forEach((id) => {
  $(id).addEventListener("input", updateSliderLabels);
  $(id).addEventListener("change", apply);
});
applyGating();
updateSliderLabels();
refreshPcTools();
