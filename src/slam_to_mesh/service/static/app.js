// Interactive quad decimation UI.
// Loads a job's LOD glb, lets the user pick a face target via a slider,
// re-requests the LOD from the API, and previews it (rotatable). See
// docs/spec_interactive_decimation.md.

import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";

const $ = (id) => document.getElementById(id);

// --- Three.js scene ---------------------------------------------------------
const viewerEl = $("viewer");
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(window.devicePixelRatio);
viewerEl.appendChild(renderer.domElement);

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

let currentMesh = null;
const loader = new GLTFLoader();

function resize() {
  const w = viewerEl.clientWidth;
  const h = viewerEl.clientHeight;
  renderer.setSize(w, h, false);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
}
window.addEventListener("resize", resize);
resize();

function animate() {
  requestAnimationFrame(animate);
  controls.update();
  renderer.render(scene, camera);
}
animate();

// Frame the loaded object so it fills the view.
function frameObject(obj) {
  const box = new THREE.Box3().setFromObject(obj);
  const size = box.getSize(new THREE.Vector3());
  const center = box.getCenter(new THREE.Vector3());
  obj.position.sub(center); // recenter at origin
  const maxDim = Math.max(size.x, size.y, size.z) || 1;
  const dist = maxDim * 2.2;
  camera.position.set(0, 0, dist);
  camera.near = maxDim / 100;
  camera.far = maxDim * 100;
  camera.updateProjectionMatrix();
  controls.target.set(0, 0, 0);
  controls.update();
}

function setMesh(gltf) {
  if (currentMesh) {
    scene.remove(currentMesh);
    currentMesh.traverse((o) => {
      if (o.geometry) o.geometry.dispose();
      if (o.material) o.material.dispose?.();
    });
  }
  currentMesh = gltf.scene;
  scene.add(currentMesh);
  frameObject(currentMesh);
}

// --- API glue ---------------------------------------------------------------
let jobId = null;
let inputFaces = null; // for % → target mapping

function setStatus(msg) { $("status").textContent = msg || ""; }

async function loadJob() {
  jobId = $("job").value.trim();
  if (!jobId) { setStatus("enter a job id"); return; }
  setStatus("loading job…");
  const r = await fetch(`/jobs/${jobId}`);
  if (!r.ok) { setStatus("job not found"); return; }
  const info = await r.json();
  // Pull the original input face count from the ingest stage metrics.
  inputFaces = info.stages?.ingest?.metrics?.faces
            ?? info.stages?.ingest?.metrics?.vertices
            ?? null;
  updateTargetLabel();
  setStatus("job loaded — pick a level and Apply");
  applyLod(); // show an initial LOD
}

function currentTargetFaces() {
  const pct = parseInt($("slider").value, 10);
  if (inputFaces) return Math.max(200, Math.round(inputFaces * pct / 100));
  return null; // fall back to ratio on the server
}

function updateTargetLabel() {
  const pct = parseInt($("slider").value, 10);
  $("pct").textContent = `${pct}%`;
  const tf = currentTargetFaces();
  $("targetFaces").textContent = tf ? `${tf} faces` : "—";
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
    updateMetrics(data.lod);
    loader.load(
      data.glb_url,
      (gltf) => { setMesh(gltf); setStatus("done"); },
      undefined,
      () => setStatus("failed to load glb"),
    );
  } catch (e) {
    setStatus("request failed");
  } finally {
    $("apply").disabled = false;
  }
}

function updateMetrics(lod) {
  $("mFaces").textContent = lod.actual_faces;
  $("mQuad").textContent = `${(lod.quad_ratio * 100).toFixed(0)}%`;
  $("mErr").textContent = `${lod.mean_dist_pct_bbox.toFixed(3)}% bbox`;
  $("mHaus").textContent = `${lod.hausdorff_pct_bbox.toFixed(2)}% bbox`;
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
  a.href = url;
  a.download = `${jobId}_lod.zip`;
  a.click();
  URL.revokeObjectURL(url);
  setStatus("exported");
}

// --- wiring -----------------------------------------------------------------
$("load").addEventListener("click", loadJob);
$("slider").addEventListener("input", updateTargetLabel);   // live label
$("slider").addEventListener("change", applyLod);           // fire on release
$("apply").addEventListener("click", applyLod);
$("bake").addEventListener("change", applyLod);
$("export").addEventListener("click", exportLod);
updateTargetLabel();
