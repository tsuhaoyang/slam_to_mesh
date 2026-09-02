// Interactive quad decimation UI.
// The SAME decimated LOD is shown two ways, updated together when the slider
// changes: left = shaded surface, right = real quad wireframe (polygon edges
// from the backend, not triangle diagonals). See docs/spec_interactive_decimation.md.

import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";

const $ = (id) => document.getElementById(id);
const loader = new GLTFLoader();

function makeViewer(el) {
  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(window.devicePixelRatio);
  el.appendChild(renderer.domElement);

  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x2a2a2e);

  const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 5000);
  camera.position.set(0, 0, 3);

  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;

  scene.add(new THREE.HemisphereLight(0xffffff, 0x444455, 1.0));
  const dir = new THREE.DirectionalLight(0xffffff, 1.2);
  dir.position.set(1, 2, 3);
  scene.add(dir);

  let content = null; // current object group

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
    const center = box.getCenter(new THREE.Vector3());
    obj.position.sub(center);
    const maxDim = Math.max(size.x, size.y, size.z) || 1;
    camera.position.set(0, 0, maxDim * 2.2);
    camera.near = maxDim / 100;
    camera.far = maxDim * 100;
    camera.updateProjectionMatrix();
    controls.target.set(0, 0, 0);
    controls.update();
    return maxDim;
  }

  function clear() {
    if (content) {
      scene.remove(content);
      content.traverse?.((o) => {
        if (o.geometry) o.geometry.dispose();
        if (o.material) o.material.dispose?.();
      });
      content = null;
    }
  }

  function setObject(obj, doFrame = true) {
    clear();
    content = obj;
    scene.add(obj);
    if (doFrame) frame(obj);
  }

  function render() { controls.update(); renderer.render(scene, camera); }

  window.addEventListener("resize", resize);
  resize();
  return { scene, camera, controls, resize, render, frame, setObject };
}

const left = makeViewer($("viewerSrc"));   // shaded surface
const right = makeViewer($("viewerLod"));  // quad wireframe

// Keep the two cameras in sync: dragging one rotates both.
let syncing = false;
function sync(from, to) {
  if (syncing) return;
  syncing = true;
  to.camera.position.copy(from.camera.position);
  to.camera.quaternion.copy(from.camera.quaternion);
  to.controls.target.copy(from.controls.target);
  to.camera.updateProjectionMatrix();
  syncing = false;
}
left.controls.addEventListener("change", () => sync(left, right));
right.controls.addEventListener("change", () => sync(right, left));

function animate() {
  requestAnimationFrame(animate);
  left.render();
  right.render();
}
animate();
setTimeout(() => { left.resize(); right.resize(); }, 0);

// --- API glue ---------------------------------------------------------------
let jobId = null;
let inputFaces = null;
const setStatus = (m) => { $("status").textContent = m || ""; };

async function loadJob() {
  jobId = $("job").value.trim();
  if (!jobId) { setStatus("enter a job id"); return; }
  setStatus("loading job…");
  const r = await fetch(`/jobs/${jobId}`);
  if (!r.ok) { setStatus("job not found"); return; }
  const info = await r.json();
  inputFaces = info.stages?.ingest?.metrics?.faces ?? null;
  updateTargetLabel();
  setStatus("job loaded — pick a level");
  applyLod();
}

function currentTargetFaces() {
  const pct = parseInt($("slider").value, 10);
  if (inputFaces) return Math.max(200, Math.round((inputFaces * pct) / 100));
  return null;
}

function updateTargetLabel() {
  const pct = parseInt($("slider").value, 10);
  $("pct").textContent = `${pct}%`;
  const tf = currentTargetFaces();
  $("targetFaces").textContent = tf ? `${tf} faces` : "—";
}

// Load the shaded glb into the left viewer.
function loadShaded(url) {
  return new Promise((resolve, reject) => {
    loader.load(url, (gltf) => { left.setObject(gltf.scene); resolve(); }, undefined, reject);
  });
}

// Build a quad-wireframe object from the backend's positions+edges and show it
// in the right viewer. Reuses the left camera framing so both align.
async function loadWire(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error("wire fetch failed");
  const { positions, edges } = await r.json();
  const geo = new THREE.BufferGeometry();
  geo.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
  geo.setIndex(edges);
  const mat = new THREE.LineBasicMaterial({ color: 0x8fd6a8 });
  const lines = new THREE.LineSegments(geo, mat);
  right.setObject(lines);
}

async function applyLod() {
  if (!jobId) { setStatus("load a job first"); return; }
  const pct = parseInt($("slider").value, 10);
  const bake = $("bake").checked;
  const body = inputFaces
    ? { target_faces: currentTargetFaces(), bake }
    : { ratio: pct / 100, bake };

  setStatus("building LOD…");
  $("apply").disabled = true;
  try {
    const r = await fetch(`/jobs/${jobId}/lod`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!r.ok) { setStatus(`error ${r.status}`); return; }
    const data = await r.json();
    const lod = data.lod;
    updateMetrics(lod);
    $("srcFaces").textContent = `${lod.actual_faces} faces`;
    $("lodFaces").textContent = `${lod.actual_faces} faces`;

    const base = `/jobs/${jobId}/lod/${lod.target_faces}`;
    const q = lod.baked ? "?baked=1" : "";
    await Promise.all([
      loadShaded(`${base}/model.glb${q}`),
      loadWire(`${base}/wire.json${q}`),
    ]);
    sync(left, right); // align cameras after both load
    setStatus("done");
  } catch {
    setStatus("request failed");
  } finally {
    $("apply").disabled = false;
  }
}

function updateMetrics(l) {
  $("mFaces").textContent = l.actual_faces;
  $("mQuad").textContent = `${(l.quad_ratio * 100).toFixed(0)}%`;
  $("mErr").textContent = `${l.mean_dist_pct_bbox.toFixed(3)}% bbox`;
  $("mHaus").textContent = `${l.hausdorff_pct_bbox.toFixed(2)}% bbox`;
}

async function exportLod() {
  if (!jobId) { setStatus("load a job first"); return; }
  const formats = [...document.querySelectorAll(".fmt:checked")].map((c) => c.value);
  if (formats.length === 0) { setStatus("pick a format"); return; }
  const body = inputFaces
    ? { target_faces: currentTargetFaces(), bake: $("bake").checked, formats }
    : { ratio: parseInt($("slider").value, 10) / 100, bake: $("bake").checked, formats };
  setStatus("exporting…");
  const r = await fetch(`/jobs/${jobId}/export-lod`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!r.ok) { setStatus(`export error ${r.status}`); return; }
  const blob = await r.blob();
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url; a.download = `${jobId}_lod.zip`; a.click();
  URL.revokeObjectURL(url);
  setStatus("exported");
}

$("load").addEventListener("click", loadJob);
$("slider").addEventListener("input", updateTargetLabel);
$("slider").addEventListener("change", applyLod);
$("apply").addEventListener("click", applyLod);
$("bake").addEventListener("change", applyLod);
$("export").addEventListener("click", exportLod);
updateTargetLabel();
