import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

const $ = (id) => document.getElementById(id);
const colors = ['#78d8c2', '#dd997c', '#92a9e0', '#d8b875'];
const viewport = $('viewport');
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(40, 1, 0.05, 300);
const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.setClearColor(0x101a27, 0);
viewport.appendChild(renderer.domElement);
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.09;
controls.minDistance = 2;
controls.maxDistance = 80;
scene.add(new THREE.HemisphereLight(0xc4e9f7, 0x293c49, 2.4));
const light = new THREE.DirectionalLight(0xffe3ce, 3.0);
light.position.set(4, 7, 6);
scene.add(light);
const rim = new THREE.DirectionalLight(0x8adbc6, 1.7);
rim.position.set(-4, 2, -5);
scene.add(rim);
const group = new THREE.Group();
scene.add(group);
const grid = new THREE.GridHelper(18, 36, 0x345466, 0x233a4a);
grid.material.transparent = true;
grid.material.opacity = 0.35;
scene.add(grid);
const sphere = new THREE.SphereGeometry(1, 32, 20);
const cylinder = new THREE.CylinderGeometry(1, 1, 1, 12);
const atomMaterial = new THREE.MeshStandardMaterial({ roughness: 0.28, metalness: 0.06 });
const bondMaterial = new THREE.MeshStandardMaterial({ roughness: 0.45, metalness: 0.0 });
const selection = new THREE.Mesh(new THREE.SphereGeometry(0.24, 24, 16),
  new THREE.MeshBasicMaterial({ color: 0x70dfc1, wireframe: true, transparent: true, opacity: 0.18 }));
group.add(selection);
const trailGeometry = new THREE.BufferGeometry();
const trail = new THREE.Line(trailGeometry,
  new THREE.LineBasicMaterial({ color: 0x70dfc1, transparent: true, opacity: 0.6 }));
group.add(trail);
const matrix = new THREE.Matrix4();
const quat = new THREE.Quaternion();
const yAxis = new THREE.Vector3(0, 1, 0);
const scale = new THREE.Vector3();
const a = new THREE.Vector3();
const b = new THREE.Vector3();
const mid = new THREE.Vector3();
const color = new THREE.Color();
const raycaster = new THREE.Raycaster();
const pointer = new THREE.Vector2();
let manifest, data, active, atoms, bonds;
let selected = 1, cursor = 0, playing = false, lastClock = 0, extent = 4, inspectedAtom = null;
let requestNumber = 0;

function resize() {
  const { width, height } = viewport.getBoundingClientRect();
  renderer.setSize(width, height, false);
  camera.aspect = width / height;
  camera.updateProjectionMatrix();
  if (data) drawCharts();
}
new ResizeObserver(resize).observe(viewport);

function resetCamera() {
  const distance = extent * 2.4;
  camera.position.set(distance * 0.45, distance * 0.32, distance * 0.85);
  controls.target.set(0, 0, 0);
  controls.update();
}

function setPlaying(value) {
  playing = value;
  $('play').innerHTML = value ? 'Ⅱ <span>暂停</span>' : '▶ <span>播放</span>';
  $('play').setAttribute('aria-label', value ? '暂停轨迹' : '播放轨迹');
  lastClock = performance.now();
}

async function loadRun(id) {
  const token = ++requestNumber;
  const run = manifest.runs.find((r) => r.id === id);
  setPlaying(false);
  const response = await fetch(`./data/${run.data_file}`);
  if (!response.ok) throw new Error(`无法读取轨迹：${response.status}`);
  const loaded = await response.json();
  if (token !== requestNumber) return;
  active = run;
  data = loaded;
  trailGeometry.setAttribute('position', new THREE.BufferAttribute(
    new Float32Array((data.time_fs.length + 1) * 3), 3).setUsage(THREE.DynamicDrawUsage));
  trail.frustumCulled = false;
  cursor = 0;
  $('scene-title').textContent = run.title;
  $('scenario-note').textContent = run.description;
  $('atom-count').textContent = data.numbers.length;
  $('duration').innerHTML = `${data.time_fs.at(-1).toFixed(1)} <small>fs</small>`;
  $('timeline').max = data.time_fs.length - 1;
  $('timeline').value = 0;
  $('download-data').href = `./data/${run.raw_file}`;
  $('run-scope').textContent = run.scope;
  $('evidence-note').textContent = run.finding;
  const hydrogen = data.numbers.flatMap((z, i) => z === 1 ? [i] : []);
  selected = run.tracked_hydrogen_id;
  $('proton').replaceChildren(...hydrogen.map((h) => new Option(`H ${h} · 持续原子身份`, String(h))));
  $('proton').value = selected;
  if (atoms) { group.remove(atoms); atoms.dispose(); }
  if (bonds) { group.remove(bonds); bonds.dispose(); }
  atoms = new THREE.InstancedMesh(sphere, atomMaterial, data.numbers.length);
  atoms.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
  group.add(atoms);
  const oxygen = data.numbers.flatMap((z, i) => z === 8 ? [i] : []);
  bonds = new THREE.InstancedMesh(cylinder, bondMaterial, Math.max(1, oxygen.length * hydrogen.length));
  bonds.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
  group.add(bonds);
  const first = data.positions_angstrom[0];
  const center = first.reduce((s, r) => s.map((v, k) => v + r[k] / first.length), [0, 0, 0]);
  group.position.set(...center.map((x) => -x));
  extent = Math.max(3.0, ...first.map((r) => Math.hypot(...r.map((v, k) => v - center[k])))) * 1.15;
  grid.position.y = Math.min(...first.map((r) => r[1] - center[1])) - 0.7;
  $('atom-tooltip').hidden = true;
  inspectedAtom = null;
  document.querySelectorAll('.chart-key').forEach((key) => {
    key.replaceChildren(...oxygen.flatMap((o, i) => {
      const dot = document.createElement('i');
      dot.style.background = colors[i % colors.length];
      dot.style.marginLeft = i ? '7px' : '0';
      const label = document.createElement('span');
      label.textContent = `氧 ${o}`;
      return [dot, label];
    }));
  });
  resetCamera();
  renderFrame();
}

function renderFrame() {
  if (!data) return;
  const index = Math.min(data.time_fs.length - 1, Math.floor(cursor));
  const next = Math.min(index + 1, data.time_fs.length - 1);
  const alpha = cursor - index;
  const frame = data.frames[index];
  const xyz = data.positions_angstrom[index].map((r, i) => r.map((v, k) =>
    v + alpha * (data.positions_angstrom[next][i][k] - v)));
  const hydrogen = frame.hydrogen_ids;
  const oxygen = frame.oxygen_ids;
  for (let i = 0; i < xyz.length; i++) {
    a.set(...xyz[i]);
    const radius = data.numbers[i] === 8 ? 0.34 : 0.17;
    matrix.compose(a, new THREE.Quaternion(), scale.setScalar(radius));
    atoms.setMatrixAt(i, matrix);
    if ($('charge-color').checked) {
      const charge = frame.atomic_mulliken_charges[i];
      color.set('#d4e1e9').lerp(new THREE.Color(charge > 0 ? '#ef887b' : '#68c6da'),
        Math.min(1, Math.abs(charge) / 0.65));
    } else color.set(data.numbers[i] === 8 ? '#ed826e' : '#d8e4f0');
    if (i === selected) color.set('#70dfc1');
    atoms.setColorAt(i, color);
  }
  atoms.instanceMatrix.needsUpdate = true;
  atoms.instanceColor.needsUpdate = true;
  atoms.computeBoundingSphere();
  let bondCount = 0;
  for (let h = 0; h < hydrogen.length; h++) {
    for (let o = 0; o < oxygen.length; o++) {
      const order = frame.oh_mayer_indices[h][o];
      if (order <= 0.12 || frame.oh_distances_angstrom[h][o] > 2.2) continue;
      a.set(...xyz[hydrogen[h]]);
      b.set(...xyz[oxygen[o]]);
      const distance = a.distanceTo(b);
      mid.copy(a).add(b).multiplyScalar(0.5);
      quat.setFromUnitVectors(yAxis, b.sub(a).normalize());
      const radius = 0.022 + 0.03 * Math.sqrt(Math.min(1.5, order));
      matrix.compose(mid, quat, scale.set(radius, distance, radius));
      bonds.setMatrixAt(bondCount, matrix);
      color.set(hydrogen[h] === selected ? '#5fb29f' : '#7795a5');
      bonds.setColorAt(bondCount++, color);
    }
  }
  bonds.count = bondCount;
  bonds.visible = $('show-bonds').checked;
  bonds.instanceMatrix.needsUpdate = true;
  if (bonds.instanceColor) bonds.instanceColor.needsUpdate = true;
  bonds.computeBoundingSphere();
  selection.position.set(...xyz[selected]);
  const trailPositions = trailGeometry.attributes.position;
  for (let i = 0; i <= index; i++) trailPositions.setXYZ(i, ...data.positions_angstrom[i][selected]);
  trailPositions.setXYZ(index + 1, ...xyz[selected]);
  trailPositions.needsUpdate = true;
  trailGeometry.setDrawRange(0, index + 2);
  trail.visible = $('show-trail').checked;
  const time = data.time_fs[index] + alpha * (data.time_fs[next] - data.time_fs[index]);
  $('current-time').innerHTML = `${time.toFixed(2)} <small>fs</small>`;
  $('timeline').value = index;
  $('frame-number').textContent = `${index + 1} / ${data.time_fs.length}`;
  $('energy-error').textContent = `${data.energy_balance_ev[index].toExponential(2)} eV`;
  $('electron-error').textContent = frame.electron_error.toExponential(1);
  $('scf-iterations').textContent = frame.scf_iterations;
  $('coordination').textContent = frame.coordination.join(' / ');
  $('fragment-charge').textContent = frame.fragment_mulliken_charges.map((q) =>
    `${q >= 0 ? '+' : ''}${q.toFixed(2)}`).join(' / ');
  if (inspectedAtom !== null) $('atom-tooltip').textContent = `${data.numbers[inspectedAtom] === 8 ? '氧' : '氢'} · ID ${inspectedAtom}\nMulliken 电荷 ${frame.atomic_mulliken_charges[inspectedAtom].toFixed(3)} e`;
  const hi = hydrogen.indexOf(selected);
  $('distance-current').textContent = frame.oh_distances_angstrom[hi].map((v) => v.toFixed(2)).join(' / ');
  $('bond-current').textContent = frame.oh_mayer_indices[hi].map((v) => v.toFixed(2)).join(' / ');
  if (data.charge === 1) $('reaction-state').textContent = frame.ambiguous[hi] ? '共享质子 · 几何归属不确定' : '质子化体系 · 观察归属切换';
  else if (frame.bond_supported_ionic_pair) $('reaction-state').textContent = '电子支持的离子候选';
  else if (frame.geometric_ionic_pair) $('reaction-state').textContent = '几何离子候选 · 需电子核验';
  else if (frame.compact_neutral_coordination) $('reaction-state').textContent = '中性水配位 · 未检出离子对';
  else $('reaction-state').textContent = '键长或配位变化 · 尚未确认反应';
  drawCharts();
}

function drawChart(id, values) {
  const canvas = $(id);
  const rect = canvas.getBoundingClientRect();
  const ratio = Math.min(window.devicePixelRatio, 2);
  canvas.width = Math.round(rect.width * ratio);
  canvas.height = Math.round(rect.height * ratio);
  const ctx = canvas.getContext('2d');
  ctx.scale(ratio, ratio);
  const width = rect.width, height = rect.height;
  const pad = { left: 28, right: 5, top: 5, bottom: 18 };
  const all = values.flat();
  let low = Math.max(0, Math.min(...all) - 0.08), high = Math.max(...all) + 0.08;
  if (high - low < 0.3) { const mid = (low + high) / 2; low = Math.max(0, mid - 0.15); high = mid + 0.15; }
  const span = high - low;
  const tmax = Math.max(0.1, data.time_fs.at(-1));
  const px = (t) => pad.left + t / tmax * (width - pad.left - pad.right);
  const py = (v) => pad.top + (high - v) / span * (height - pad.top - pad.bottom);
  ctx.font = '8px monospace';
  for (let i = 0; i <= 2; i++) {
    const value = low + span * i / 2;
    ctx.strokeStyle = '#283846';
    ctx.beginPath(); ctx.moveTo(pad.left, py(value)); ctx.lineTo(width, py(value)); ctx.stroke();
    ctx.fillStyle = '#678298';
    ctx.fillText(value.toFixed(1), 0, py(value) + 3);
  }
  for (let series = 0; series < values[0].length; series++) {
    ctx.strokeStyle = colors[series % colors.length];
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    for (let i = 0; i < values.length; i++) {
      const x = px(data.time_fs[i]), y = py(values[i][series]);
      i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    }
    ctx.stroke();
  }
  const frame = Math.min(values.length - 1, Math.floor(cursor));
  ctx.strokeStyle = '#97b3c68c';
  ctx.setLineDash([3, 3]);
  ctx.beginPath();ctx.moveTo(px(data.time_fs[frame]), 0);ctx.lineTo(px(data.time_fs[frame]), height - pad.bottom);ctx.stroke();
  ctx.setLineDash([]);
  ctx.fillStyle = '#678298';
  ctx.fillText('0', pad.left, height - 2);
  ctx.fillText(tmax.toFixed(1), width - 27, height - 2);
}

function drawCharts() {
  const h = data.frames[0].hydrogen_ids.indexOf(selected);
  drawChart('distance-chart', data.frames.map((f) => f.oh_distances_angstrom[h]));
  drawChart('bond-chart', data.frames.map((f) => f.oh_mayer_indices[h]));
}

function seek(index) {
  if (!data) return;
  setPlaying(false);
  cursor = Math.max(0, Math.min(data.time_fs.length - 1, index));
  renderFrame();
}

$('experiment').addEventListener('change', () => loadRun($('experiment').value).catch(showError));
$('proton').addEventListener('change', () => { selected = Number($('proton').value); renderFrame(); });
$('play').addEventListener('click', () => { if (data) setPlaying(!playing); });
$('timeline').addEventListener('input', () => seek(Number($('timeline').value)));
$('previous-frame').addEventListener('click', () => seek(Math.floor(cursor) - 1));
$('next-frame').addEventListener('click', () => seek(Math.floor(cursor) + 1));
$('reset-camera').addEventListener('click', resetCamera);
for (const id of ['show-bonds', 'charge-color', 'show-trail']) $(id).addEventListener('change', renderFrame);
let down;
renderer.domElement.addEventListener('pointerdown', (event) => { down = [event.clientX, event.clientY]; });
renderer.domElement.addEventListener('pointerup', (event) => {
  if (!atoms || !down || Math.hypot(event.clientX - down[0], event.clientY - down[1]) > 5) return;
  const rect = renderer.domElement.getBoundingClientRect();
  pointer.set(2 * (event.clientX - rect.left) / rect.width - 1, 1 - 2 * (event.clientY - rect.top) / rect.height);
  raycaster.setFromCamera(pointer, camera);
  const hit = raycaster.intersectObject(atoms)[0];
  if (!hit) { inspectedAtom = null; $('atom-tooltip').hidden = true; return; }
  const id = hit.instanceId;
  inspectedAtom = id;
  const frame = data.frames[Math.floor(cursor)];
  $('atom-tooltip').textContent = `${data.numbers[id] === 8 ? '氧' : '氢'} · ID ${id}\nMulliken 电荷 ${frame.atomic_mulliken_charges[id].toFixed(3)} e`;
  $('atom-tooltip').hidden = false;
  if (data.numbers[id] === 1) { selected = id; $('proton').value = id; renderFrame(); }
});

function animate(now) {
  requestAnimationFrame(animate);
  if (playing && data) {
    const delta = Math.min(0.1, Math.max(0, (now - lastClock) / 1000));
    cursor += delta * Number($('speed').value) / data.dt_fs;
    if (cursor >= data.time_fs.length - 1) { cursor = data.time_fs.length - 1; setPlaying(false); }
    renderFrame();
  }
  lastClock = now;
  controls.update();
  renderer.render(scene, camera);
}

function showError(error) {
  $('load-error').textContent = `轨迹载入失败：${error.message}`;
  $('load-error').hidden = false;
  console.error(error);
}

async function boot() {
  const response = await fetch('./data/manifest.json');
  if (!response.ok) throw new Error('实验数据尚未导出。请先运行计算和网页数据导出脚本。');
  manifest = await response.json();
  $('performance-environment').textContent = `${manifest.environment.cpu_model} / WSL2 / CPU float64。此处仅测 CPU 数值线程，尚无多 GPU 或分布式元胞调度。`;
  $('experiment').replaceChildren(...manifest.runs.map((r) => new Option(r.title, r.id)));
  for (const row of manifest.performance) {
    const tr = document.createElement('tr');
    const values = [row.waters, row.one_thread_seconds?.toFixed(2) ?? '预算超限',
      row.four_thread_seconds?.toFixed(2) ?? '—', row.speedup ? `${row.speedup.toFixed(2)}×` : '—',
      row.density_fill_fraction === null ? '—' : `${(100 * row.density_fill_fraction).toFixed(0)}%`];
    tr.replaceChildren(...values.map((value) => { const td = document.createElement('td');td.textContent = value;return td; }));
    $('performance-body').appendChild(tr);
  }
  const preferred = manifest.runs.find((r) => r.scenario === 'shared_proton') ?? manifest.runs[0];
  if (!preferred) throw new Error('没有已计算的轨迹。');
  $('experiment').value = preferred.id;
  await loadRun(preferred.id);
}
resize();
resetCamera();
requestAnimationFrame(animate);
boot().catch(showError);
