/**
 * Real-time 3D Avatar Kinematic Controller and UI Controller
 * Bunkai MoCap Live
 * Solves hierarchical bone orientations directly in Three.js space for Karateka_Armature.
 * Supports individual finger tracking (15 bones per hand) and upright head posture.
 */

// --- THREE.JS SCENE SETUP ---
const container = document.getElementById('viewport-3d');
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x111215);

const camera = new THREE.PerspectiveCamera(45, container.clientWidth / container.clientHeight, 0.1, 100);
camera.position.set(0, 1.25, 3.2);

const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
renderer.setSize(container.clientWidth, container.clientHeight);
renderer.setPixelRatio(window.devicePixelRatio);
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.15;
container.appendChild(renderer.domElement);

const controls = new THREE.OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.05;
controls.target.set(0, 0.95, 0);

// Lighting (Dojo Atmosphere)
const hemiLight = new THREE.HemisphereLight(0xffffff, 0x3d414d, 0.85);
hemiLight.position.set(0, 20, 0);
scene.add(hemiLight);

const keyLight = new THREE.DirectionalLight(0xffffff, 1.2);
keyLight.position.set(2.5, 6, 4);
keyLight.castShadow = true;
keyLight.shadow.mapSize.width = 1024;
keyLight.shadow.mapSize.height = 1024;
scene.add(keyLight);

const fillLight = new THREE.DirectionalLight(0x70a1ff, 0.4);
fillLight.position.set(-3, 4, -2);
scene.add(fillLight);

// Tatami Floor Grid
const grid = new THREE.GridHelper(6, 24, 0xff334b, 0x2e3340);
grid.position.y = 0;
scene.add(grid);

// State
let avatarModel = null;
let skeletonHelper = null;
const boneMap = {};
let isModelReady = false;
let smoothRate = 0.35;

// Finger Bone Names (3 segments per finger, 5 fingers per hand = 30 finger bones)
const FINGER_TYPES = ["Thumb", "Index", "Middle", "Ring", "Pinky"];

// --- LOAD 3D AVATAR ---
const gltfLoader = new THREE.GLTFLoader();
const modelUrl = "/static/models/karateka_lowpoly.glb";

gltfLoader.load(modelUrl, (gltf) => {
  avatarModel = gltf.scene;
  scene.add(avatarModel);

  // Initialize bones and cache their rest-pose geometric directions in Three.js world space
  avatarModel.updateMatrixWorld(true);

  avatarModel.traverse((node) => {
    if (node.isBone) {
      boneMap[node.name] = node;

      // 1. Initial rest local quaternion
      node.restLocalQ = node.quaternion.clone();

      // 2. Initial rest world quaternion
      node.restWorldQ = new THREE.Quaternion();
      node.getWorldQuaternion(node.restWorldQ);

      // 3. In glTF exported from Blender, every bone extends along its local +Y axis:
      // Computing rest world pointing vector:
      node.restWorldDir = new THREE.Vector3(0, 1, 0).applyQuaternion(node.restWorldQ).normalize();
    }
    if (node.isMesh) {
      node.castShadow = true;
      node.receiveShadow = true;
    }
  });

  skeletonHelper = new THREE.SkeletonHelper(avatarModel);
  skeletonHelper.visible = false;
  scene.add(skeletonHelper);

  isModelReady = true;
  document.getElementById('loading-modal').classList.add('hidden');
  console.log("Karateka avatar ready. Total bones initialized:", Object.keys(boneMap).length);
}, undefined, (err) => {
  console.error("Error loading GLB avatar:", err);
  document.getElementById('loading-modal').innerHTML = `
    <div style="color: #ff334b; font-weight: bold;">Ошибка загрузки 3D модели:</div>
    <div>${err.message || 'Проверьте файл static/models/karateka_lowpoly.glb'}</div>
  `;
});

/**
 * Universal Hierarchical Alignment:
 * Rotates a bone so that its longitudinal axis points along targetDir in world space,
 * taking into account its parent bone's current orientation.
 */
function alignBone(bone, targetDir, alpha) {
  if (!bone || !bone.restWorldDir || !targetDir || targetDir.lengthSq() < 1e-5) return;

  const rotWorld = new THREE.Quaternion().setFromUnitVectors(bone.restWorldDir, targetDir.clone().normalize());
  const targetWorldQ = rotWorld.multiply(bone.restWorldQ);

  const parentWorldQ = new THREE.Quaternion();
  if (bone.parent) {
    bone.parent.getWorldQuaternion(parentWorldQ);
  }

  const targetLocalQ = parentWorldQ.clone().invert().multiply(targetWorldQ);
  bone.quaternion.slerp(targetLocalQ, alpha);
  bone.updateMatrixWorld(true);
}

/**
 * Reset all bones to canonical rest T-pose.
 */
function resetPose() {
  for (const bone of Object.values(boneMap)) {
    if (bone.restLocalQ) {
      bone.quaternion.copy(bone.restLocalQ);
    }
  }
  if (avatarModel) avatarModel.updateMatrixWorld(true);
}

/**
 * Apply individual finger curl angles for all 5 fingers.
 */
function applyFingerCurls(side, fingerData, defaultClenched) {
  for (const ftype of FINGER_TYPES) {
    const key = ftype.toLowerCase();
    let curl = 0.0;
    if (fingerData && fingerData.detected && typeof fingerData[key] === 'number') {
      curl = fingerData[key];
    } else {
      curl = defaultClenched ? 0.95 : 0.0;
    }

    // Curl angles: Index/Middle/Ring/Pinky flex around +X
    for (let idx = 1; idx <= 3; idx++) {
      const b = boneMap[`${ftype}${idx}.${side}`];
      if (b && b.restLocalQ) {
        if (ftype === "Thumb") {
          // Thumb has compound flexion
          const ang = curl * (idx === 1 ? 0.6 : 0.9);
          const flexQ = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(1, 0, 0), ang);
          b.quaternion.copy(b.restLocalQ.clone().multiply(flexQ));
        } else {
          const maxAng = idx === 2 ? 1.6 : (idx === 1 ? 1.4 : 1.1);
          const ang = curl * maxAng;
          const flexQ = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(1, 0, 0), ang);
          b.quaternion.copy(b.restLocalQ.clone().multiply(flexQ));
        }
      }
    }
  }
}

// --- WEBSOCKET REAL-TIME MOTION STREAM ---
let ws = null;

function connectWebSocket() {
  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  const wsUrl = `${protocol}//${window.location.host}/ws`;
  ws = new WebSocket(wsUrl);

  ws.onopen = () => {
    console.log("WebSocket connected to MoCap stream.");
  };

  ws.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data);
      handlePoseData(data);
    } catch (e) {
      console.warn("WS parse error:", e);
    }
  };

  ws.onclose = () => {
    setTimeout(connectWebSocket, 1000);
  };
}

const statusDot = document.getElementById('status-dot');
const statusText = document.getElementById('status-text');
const hudFps = document.getElementById('hud-fps');
const telemFps = document.getElementById('telem-fps');
const telemStatus = document.getElementById('telem-status');
const telemBones = document.getElementById('telem-bones');
const checkFists = document.getElementById('check-fists');

function handlePoseData(data) {
  hudFps.textContent = `FPS: ${data.fps || 0}`;
  telemFps.textContent = `${data.fps || 0} fps`;

  if (data.detected) {
    statusDot.className = "dot-status";
    statusText.textContent = "Захват позы активен";
    telemStatus.textContent = "Зеркалит в реальном времени";
    telemStatus.style.color = "#38ef7d";
  } else {
    statusDot.className = "dot-status searching";
    statusText.textContent = "Поиск спортсмена...";
    telemStatus.textContent = "Ожидание человека в кадре";
    telemStatus.style.color = "#f59e0b";
  }

  if (!isModelReady || !data.detected || !data.landmarks) return;

  const pts = data.landmarks;
  const vis = data.visibility || {};
  const alpha = Math.max(0.2, 1.0 - smoothRate);
  let activeCount = 0;

  // Vector helper
  const getVec = (name) => pts[name] ? new THREE.Vector3(pts[name][0], pts[name][1], pts[name][2]) : null;

  const pShL = getVec("left_shoulder");
  const pShR = getVec("right_shoulder");
  const pElL = getVec("left_elbow");
  const pElR = getVec("right_elbow");
  const pWrL = getVec("left_wrist");
  const pWrR = getVec("right_wrist");

  const pHipL = getVec("left_hip");
  const pHipR = getVec("right_hip");
  const pKnL = getVec("left_knee");
  const pKnR = getVec("right_knee");
  const pAnL = getVec("left_ankle");
  const pAnR = getVec("right_ankle");

  // 1. Torso / Spine Tilt
  const minTorsoVis = 0.15;
  const boneSpine = boneMap["Spine"];
  if (boneSpine && boneSpine.restLocalQ) {
    if (pShL && pShR && pHipL && pHipR && (vis.left_shoulder || 0) > minTorsoVis && (vis.right_shoulder || 0) > minTorsoVis && (vis.left_hip || 0) > minTorsoVis) {
      const midSh = pShL.clone().add(pShR).multiplyScalar(0.5);
      const midHp = pHipL.clone().add(pHipR).multiplyScalar(0.5);
      const torsoDir = midSh.sub(midHp).normalize();
      alignBone(boneSpine, torsoDir, alpha * 0.5);
      activeCount++;
    } else {
      boneSpine.quaternion.slerp(boneSpine.restLocalQ, 0.08);
    }
  }

  // 2. Upright Head (No drooping; calibrated Metsuke posture)
  if (data.head && boneMap["Head"] && boneMap["Head"].restLocalQ) {
    const h = data.head;
    const euler = new THREE.Euler(h.pitch || 0, h.yaw || 0, h.roll || 0, 'YXZ');
    const headRotQ = new THREE.Quaternion().setFromEuler(euler);
    const targetHeadLocal = boneMap["Head"].restLocalQ.clone().multiply(headRotQ);
    boneMap["Head"].quaternion.slerp(targetHeadLocal, alpha * 0.85);
    activeCount++;
  }

  // 3. Left Arm (Avatar's Left Arm, on viewer's right side)
  const minArmVis = 0.10;
  const canUpperL = pShL && pElL && (vis.left_shoulder || 0) > minArmVis && (vis.left_elbow || 0) > minArmVis;
  if (canUpperL && boneMap["UpperArm.L"]) {
    const dirUpperL = pElL.clone().sub(pShL).normalize();
    alignBone(boneMap["UpperArm.L"], dirUpperL, alpha);
    activeCount++;

    if (pWrL && (vis.left_wrist || 0) > minArmVis && boneMap["LowerArm.L"]) {
      const dirLowerL = pWrL.clone().sub(pElL).normalize();
      alignBone(boneMap["LowerArm.L"], dirLowerL, alpha);
      activeCount++;
    } else if (boneMap["LowerArm.L"]) {
      alignBone(boneMap["LowerArm.L"], dirUpperL, alpha * 0.5);
    }
  } else if (boneMap["UpperArm.L"] && boneMap["UpperArm.L"].restLocalQ) {
    boneMap["UpperArm.L"].quaternion.slerp(boneMap["UpperArm.L"].restLocalQ, 0.08);
    if (boneMap["LowerArm.L"] && boneMap["LowerArm.L"].restLocalQ) {
      boneMap["LowerArm.L"].quaternion.slerp(boneMap["LowerArm.L"].restLocalQ, 0.08);
    }
  }

  // 4. Right Arm (Avatar's Right Arm, on viewer's left side)
  const canUpperR = pShR && pElR && (vis.right_shoulder || 0) > minArmVis && (vis.right_elbow || 0) > minArmVis;
  if (canUpperR && boneMap["UpperArm.R"]) {
    const dirUpperR = pElR.clone().sub(pShR).normalize();
    alignBone(boneMap["UpperArm.R"], dirUpperR, alpha);
    activeCount++;

    if (pWrR && (vis.right_wrist || 0) > minArmVis && boneMap["LowerArm.R"]) {
      const dirLowerR = pWrR.clone().sub(pElR).normalize();
      alignBone(boneMap["LowerArm.R"], dirLowerR, alpha);
      activeCount++;
    } else if (boneMap["LowerArm.R"]) {
      alignBone(boneMap["LowerArm.R"], dirUpperR, alpha * 0.5);
    }
  } else if (boneMap["UpperArm.R"] && boneMap["UpperArm.R"].restLocalQ) {
    boneMap["UpperArm.R"].quaternion.slerp(boneMap["UpperArm.R"].restLocalQ, 0.08);
    if (boneMap["LowerArm.R"] && boneMap["LowerArm.R"].restLocalQ) {
      boneMap["LowerArm.R"].quaternion.slerp(boneMap["LowerArm.R"].restLocalQ, 0.08);
    }
  }

  // 5. Legs (Tracking when visible; stable karate stance when seated)
  const minLegVis = 0.12;
  const isStandingL = pKnL && pHipL && (vis.left_knee || 0) > minLegVis;
  if (isStandingL && boneMap["UpperLeg.L"]) {
    const dirThighL = pKnL.clone().sub(pHipL).normalize();
    alignBone(boneMap["UpperLeg.L"], dirThighL, alpha);
    activeCount++;

    if (pAnL && (vis.left_ankle || 0) > minLegVis && boneMap["LowerLeg.L"]) {
      const dirShinL = pAnL.clone().sub(pKnL).normalize();
      alignBone(boneMap["LowerLeg.L"], dirShinL, alpha);
      activeCount++;
    }
  } else {
    if (boneMap["UpperLeg.L"] && boneMap["UpperLeg.L"].restLocalQ) {
      boneMap["UpperLeg.L"].quaternion.slerp(boneMap["UpperLeg.L"].restLocalQ, 0.1);
    }
    if (boneMap["LowerLeg.L"] && boneMap["LowerLeg.L"].restLocalQ) {
      boneMap["LowerLeg.L"].quaternion.slerp(boneMap["LowerLeg.L"].restLocalQ, 0.1);
    }
  }

  const isStandingR = pKnR && pHipR && (vis.right_knee || 0) > minLegVis;
  if (isStandingR && boneMap["UpperLeg.R"]) {
    const dirThighR = pKnR.clone().sub(pHipR).normalize();
    alignBone(boneMap["UpperLeg.R"], dirThighR, alpha);
    activeCount++;

    if (pAnR && (vis.right_ankle || 0) > minLegVis && boneMap["LowerLeg.R"]) {
      const dirShinR = pAnR.clone().sub(pKnR).normalize();
      alignBone(boneMap["LowerLeg.R"], dirShinR, alpha);
      activeCount++;
    }
  } else {
    if (boneMap["UpperLeg.R"] && boneMap["UpperLeg.R"].restLocalQ) {
      boneMap["UpperLeg.R"].quaternion.slerp(boneMap["UpperLeg.R"].restLocalQ, 0.1);
    }
    if (boneMap["LowerLeg.R"] && boneMap["LowerLeg.R"].restLocalQ) {
      boneMap["LowerLeg.R"].quaternion.slerp(boneMap["LowerLeg.R"].restLocalQ, 0.1);
    }
  }

  // 6. Fingers (All 30 finger bones: 15 on Left, 15 on Right)
  const defaultFists = checkFists.checked;
  const fLeft = data.fingers ? data.fingers.left : null;
  const fRight = data.fingers ? data.fingers.right : null;

  applyFingerCurls("L", fLeft, defaultFists);
  applyFingerCurls("R", fRight, defaultFists);
  activeCount += 30;

  // Propagate all bone transforms through hierarchy
  if (avatarModel) {
    avatarModel.updateMatrixWorld(true);
  }

  telemBones.textContent = `${activeCount} звеньев (вкл. 30 костей пальцев)`;
}

connectWebSocket();

// --- UI EVENT HANDLERS ---
const btnWebcam = document.getElementById('btn-webcam');
const btnSample = document.getElementById('btn-sample');
const btnUpload = document.getElementById('btn-upload');
const btnPause = document.getElementById('btn-pause');
const btnRecord = document.getElementById('btn-record');
const fileInput = document.getElementById('file-upload-input');
const checkMirror = document.getElementById('check-mirror');
const checkSkeleton = document.getElementById('check-skeleton');
const checkWireframe = document.getElementById('check-wireframe');
const smoothSlider = document.getElementById('smooth-slider');
const smoothVal = document.getElementById('smooth-val');

function updateSourceBtn(activeBtn) {
  [btnWebcam, btnSample, btnUpload].forEach(b => b.classList.remove('active'));
  activeBtn.classList.add('active');
}

btnWebcam.addEventListener('click', async () => {
  await fetch('/api/source/webcam?camera_index=0', { method: 'POST' });
  updateSourceBtn(btnWebcam);
  checkMirror.checked = true;
});

btnSample.addEventListener('click', async () => {
  await fetch('/api/source/sample_video', { method: 'POST' });
  updateSourceBtn(btnSample);
  checkMirror.checked = false;
});

btnUpload.addEventListener('click', () => {
  fileInput.click();
});

fileInput.addEventListener('change', async () => {
  if (!fileInput.files || fileInput.files.length === 0) return;
  const file = fileInput.files[0];
  const formData = new FormData();
  formData.append('file', file);
  document.getElementById('loading-modal').classList.remove('hidden');
  document.getElementById('loading-text').textContent = `Загрузка ${file.name}...`;

  try {
    const res = await fetch('/api/upload_video', { method: 'POST', body: formData });
    const json = await res.json();
    if (json.status === 'ok') {
      updateSourceBtn(btnUpload);
      checkMirror.checked = false;
    }
  } catch (e) {
    alert("Ошибка загрузки видео: " + e.message);
  } finally {
    document.getElementById('loading-modal').classList.add('hidden');
  }
});

btnPause.addEventListener('click', async () => {
  const res = await fetch('/api/toggle_pause', { method: 'POST' });
  const data = await res.json();
  btnPause.textContent = data.paused ? "▶ Возобновить" : "⏸ Пауза";
});

let isRecording = false;
btnRecord.addEventListener('click', async () => {
  if (!isRecording) {
    await fetch('/api/record/start', { method: 'POST' });
    isRecording = true;
    btnRecord.classList.add('recording');
    btnRecord.textContent = "⏹ Стоп и сохранить";
  } else {
    btnRecord.textContent = "Сохранение...";
    const res = await fetch('/api/record/stop', { method: 'POST' });
    const data = await res.json();
    isRecording = false;
    btnRecord.classList.remove('recording');
    btnRecord.textContent = "⏺ Начать запись";

    if (data.status === 'ok') {
      alert(`Движение успешно записано в JSON!\n\nФайл: ${data.filename}\nПуть: ${data.json_path}`);
    } else {
      alert("Запись остановлена (нет кадров).");
    }
  }
});

checkMirror.addEventListener('change', async () => {
  await fetch('/api/settings', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ mirror: checkMirror.checked })
  });
});

smoothSlider.addEventListener('input', async () => {
  smoothRate = parseFloat(smoothSlider.value);
  smoothVal.textContent = smoothRate.toFixed(2);
  await fetch('/api/settings', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ smooth: smoothRate })
  });
});

checkSkeleton.addEventListener('change', () => {
  if (skeletonHelper) skeletonHelper.visible = checkSkeleton.checked;
});

checkWireframe.addEventListener('change', () => {
  if (!avatarModel) return;
  avatarModel.traverse((node) => {
    if (node.isMesh && node.material) {
      node.material.wireframe = checkWireframe.checked;
    }
  });
});

document.getElementById('btn-reset-pose').addEventListener('click', () => {
  resetPose();
});

document.getElementById('btn-cam-front').addEventListener('click', () => {
  camera.position.set(0, 1.25, 3.2);
  controls.target.set(0, 0.95, 0);
});

document.getElementById('btn-cam-side').addEventListener('click', () => {
  camera.position.set(3.2, 1.25, 0);
  controls.target.set(0, 0.95, 0);
});

document.getElementById('btn-cam-34').addEventListener('click', () => {
  camera.position.set(2.0, 1.4, 2.5);
  controls.target.set(0, 0.95, 0);
});

// Calibration Controls
async function triggerCalibration() {
  try {
    await fetch('/api/calibrate', { method: 'POST' });
    console.log("Postural baseline calibrated.");
    const origText = telemStatus.textContent;
    telemStatus.textContent = "🎯 Осанка откалибрована (голова прямо)!";
    telemStatus.style.color = "#2ed573";
    setTimeout(() => { telemStatus.textContent = origText; }, 2500);
  } catch (e) {
    console.warn("Calibration error:", e);
  }
}

const btnCalib = document.getElementById('btn-calibrate');
if (btnCalib) btnCalib.addEventListener('click', triggerCalibration);

const btnCalibVp = document.getElementById('btn-calibrate-vp');
if (btnCalibVp) btnCalibVp.addEventListener('click', triggerCalibration);

window.addEventListener('keydown', (e) => {
  if (e.key === 'c' || e.key === 'C' || e.key === 'с' || e.key === 'С') {
    triggerCalibration();
  }
});

// Resize handler
window.addEventListener('resize', () => {
  camera.aspect = container.clientWidth / container.clientHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(container.clientWidth, container.clientHeight);
});

// --- RENDER LOOP ---
function animate() {
  requestAnimationFrame(animate);
  controls.update();
  if (skeletonHelper && skeletonHelper.visible) {
    skeletonHelper.update();
  }
  renderer.render(scene, camera);
}
animate();
