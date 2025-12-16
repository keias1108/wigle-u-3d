import {
  GRID_SIZE_PRESETS,
  DEFAULT_GRID_SIZE,
  SPEED_OPTIONS,
  INITIAL_YAW,
  INITIAL_PITCH,
  INITIAL_DISTANCE,
} from './config/constants.js';
import { DEFAULT_PARAMS, PARAM_SPECS } from './config/defaults.js';
import { WebGPUSimulation3D } from './core/WebGPUSimulation3D.js';

const canvas = document.getElementById('canvas');
const paramsContainer = document.getElementById('params');
const gridSizeSelect = document.getElementById('gridSize');
const seedBtn = document.getElementById('seedBtn');
const paletteSelect = document.getElementById('palette');
const rayStepsSelect = document.getElementById('raySteps');
const neighborModeSelect = document.getElementById('neighborMode');
const yawInput = document.getElementById('yaw');
const yawLabel = document.getElementById('yawLabel');
const pitchInput = document.getElementById('pitch');
const pitchLabel = document.getElementById('pitchLabel');
const saveBtn = document.getElementById('saveBtn');
const loadInput = document.getElementById('loadInput');
const fpsLabel = document.getElementById('fps');
const speedDisplay = document.getElementById('speedDisplay');
const filterLowCheckbox = document.getElementById('filterLow');
const filterMidLowCheckbox = document.getElementById('filterMidLow');
const filterMidHighCheckbox = document.getElementById('filterMidHigh');
const filterHighCheckbox = document.getElementById('filterHigh');

let lastNonZeroSpeed = 1;
let captureDirHandle = null;

// localStorage: Collapsible 그룹 상태 관리
const STORAGE_KEY = 'wigle-u-3d-collapsed-groups';

function loadCollapsedState(groupName) {
  try {
    const saved = localStorage.getItem(STORAGE_KEY);
    if (!saved) return false;
    const state = JSON.parse(saved);
    return state[groupName] || false;
  } catch {
    return false;
  }
}

function saveCollapsedState(groupName, isCollapsed) {
  try {
    const saved = localStorage.getItem(STORAGE_KEY) || '{}';
    const state = JSON.parse(saved);
    state[groupName] = isCollapsed;
    localStorage.setItem(STORAGE_KEY, JSON.stringify(state));
  } catch (e) {
    console.error('Failed to save collapsed state:', e);
  }
}

const sim = new WebGPUSimulation3D({
  canvas,
  initialParams: DEFAULT_PARAMS,
  gridSize: DEFAULT_GRID_SIZE,
  yaw: INITIAL_YAW,
  pitch: INITIAL_PITCH,
  distance: INITIAL_DISTANCE,
});

initControls();
await sim.init();
updateUIFromParams(sim.params);

function initControls() {
  GRID_SIZE_PRESETS.forEach((p) => {
    const option = document.createElement('option');
    option.value = p.value;
    option.textContent = p.label;
    if (p.value === DEFAULT_GRID_SIZE) option.selected = true;
    gridSizeSelect.appendChild(option);
  });

  const makeControl = (spec) => {
    const wrapper = document.createElement('div');
    wrapper.className = 'param';
    const label = document.createElement('label');
    label.textContent = spec.label;
    const valueSpan = document.createElement('span');
    valueSpan.className = 'value';
    valueSpan.id = `value-${spec.key}`;
    label.appendChild(valueSpan);

    const row = document.createElement('div');
    row.className = 'row';

    const slider = document.createElement('input');
    slider.type = 'range';
    slider.min = spec.min;
    slider.max = spec.max;
    slider.step = spec.step;
    slider.value = DEFAULT_PARAMS[spec.key];
    slider.dataset.key = spec.key;

    const number = document.createElement('input');
    number.type = 'number';
    number.min = spec.min;
    number.max = spec.max;
    number.step = spec.step;
    number.value = DEFAULT_PARAMS[spec.key];
    number.dataset.key = spec.key;

    const onChange = (val, key) => {
      sim.updateParam(key, val);
      slider.value = val;
      number.value = val;
      updateValueLabel(key, val);
    };

    slider.addEventListener(
      'wheel',
      (e) => {
        e.preventDefault();
        const delta = e.deltaY > 0 ? -1 : 1;
        const baseStep = parseFloat(spec.step);
        const step =
          spec.key === 'growthWidth' && baseStep < 0.001 ? 0.001 : baseStep;
        const next = parseFloat(slider.value) + delta * step;
        onChange(next, spec.key);
      },
      { passive: false },
    );
    slider.addEventListener('input', (e) => {
      const key = e.target.dataset.key;
      const value = parseFloat(e.target.value);
      onChange(value, key);
    });

    number.addEventListener('input', (e) => {
      const key = e.target.dataset.key;
      const value = parseFloat(e.target.value);
      if (Number.isNaN(value)) return;
      onChange(value, key);
    });

    row.appendChild(slider);
    row.appendChild(number);
    wrapper.appendChild(label);
    wrapper.appendChild(row);
    return wrapper;
  };

  // Group controls like wigle-u
  const tension = PARAM_SPECS.slice(0, 4);
  const economy = PARAM_SPECS.slice(4, 7);
  const growth = PARAM_SPECS.slice(7, 10);
  const globalMod = PARAM_SPECS.slice(10);

  const addGroup = (title, specs) => {
    // 그룹 컨테이너
    const group = document.createElement('div');
    group.className = 'param-group';

    // 헤더 (클릭 가능)
    const header = document.createElement('h3');
    header.className = 'group-header';
    header.innerHTML = `<span class="toggle">▼</span>${title}`;

    // 콘텐츠 컨테이너
    const content = document.createElement('div');
    content.className = 'group-content';

    // localStorage에서 상태 복원
    const collapsed = loadCollapsedState(title);
    if (collapsed) {
      content.style.display = 'none';
      header.querySelector('.toggle').textContent = '▶';
    }

    // 클릭 이벤트: 접었다 펼치기
    header.addEventListener('click', () => {
      const isCollapsed = content.style.display === 'none';
      content.style.display = isCollapsed ? 'block' : 'none';
      header.querySelector('.toggle').textContent = isCollapsed ? '▼' : '▶';
      saveCollapsedState(title, !isCollapsed);
    });

    // 그룹 조립
    group.appendChild(header);
    group.appendChild(content);
    paramsContainer.appendChild(group);

    // 파라미터들을 content에 추가
    specs.forEach((spec) => {
      const control = makeControl(spec);
      content.appendChild(control);
    });
  };

  addGroup('Dynamic Tension', tension);
  addGroup('Energy Economy', economy);
  addGroup('Growth Function', growth);
  addGroup('Global', globalMod);

  // Settings, Data 그룹 collapsible 설정
  const allSections = document.querySelectorAll('.section');
  // allSections[0] = #params, [1] = Settings, [2] = Data
  const settingsSection = allSections[1];
  const dataSection = allSections[2];

  const settingsHeader = settingsSection.querySelector('.group-header');
  const settingsContent = settingsSection.querySelector('.group-content');
  const dataHeader = dataSection.querySelector('.group-header');
  const dataContent = dataSection.querySelector('.group-content');

  const setupCollapsible = (header, content, groupName) => {
    const collapsed = loadCollapsedState(groupName);
    if (collapsed) {
      content.style.display = 'none';
      header.querySelector('.toggle').textContent = '▶';
    } else {
      content.style.display = 'block';
      header.querySelector('.toggle').textContent = '▼';
    }

    header.addEventListener('click', () => {
      const isCollapsed = content.style.display === 'none';
      content.style.display = isCollapsed ? 'block' : 'none';
      header.querySelector('.toggle').textContent = isCollapsed ? '▼' : '▶';
      saveCollapsedState(groupName, !isCollapsed);
    });
  };

  setupCollapsible(settingsHeader, settingsContent, 'Settings');
  setupCollapsible(dataHeader, dataContent, 'Data');

  // Speed display update helper
  const updateSpeedDisplay = (speed) => {
    speedDisplay.textContent = speed === 0 ? '⏸' : `${speed}x`;
  };

  gridSizeSelect.addEventListener('change', async (e) => {
    await sim.resizeGrid(Number(e.target.value));
  });

  seedBtn.addEventListener('click', () => {
    sim.reseed();
  });

  paletteSelect.addEventListener('change', (e) => {
    const mode = Number(e.target.value);
    sim.updateParam('paletteMode', mode);
    updateValueLabel('paletteMode', mode);
  });

  rayStepsSelect.addEventListener('change', (e) => {
    const steps = Number(e.target.value);
    sim.updateParam('raySteps', steps);
  });

  neighborModeSelect.addEventListener('change', (e) => {
    const mode = Number(e.target.value);
    sim.updateParam('neighborMode', mode);
  });

  // Energy range filter handlers
  const FILTER_STORAGE_KEY = 'wigle-u-3d-energy-filters';

  const updateFilters = () => {
    let filterBits = 0;
    if (filterLowCheckbox.checked) filterBits |= 0b0001;
    if (filterMidLowCheckbox.checked) filterBits |= 0b0010;
    if (filterMidHighCheckbox.checked) filterBits |= 0b0100;
    if (filterHighCheckbox.checked) filterBits |= 0b1000;

    sim.updateParam('energyRangeFilters', filterBits);

    // Save to localStorage
    try {
      localStorage.setItem(FILTER_STORAGE_KEY, filterBits.toString());
    } catch (e) {
      console.warn('Failed to save filter state:', e);
    }
  };

  filterLowCheckbox.addEventListener('change', updateFilters);
  filterMidLowCheckbox.addEventListener('change', updateFilters);
  filterMidHighCheckbox.addEventListener('change', updateFilters);
  filterHighCheckbox.addEventListener('change', updateFilters);

  // Load filter state from localStorage on startup
  try {
    const saved = localStorage.getItem(FILTER_STORAGE_KEY);
    if (saved !== null) {
      const filterBits = parseInt(saved, 10);
      sim.updateParam('energyRangeFilters', filterBits);
      filterLowCheckbox.checked = (filterBits & 0b0001) !== 0;
      filterMidLowCheckbox.checked = (filterBits & 0b0010) !== 0;
      filterMidHighCheckbox.checked = (filterBits & 0b0100) !== 0;
      filterHighCheckbox.checked = (filterBits & 0b1000) !== 0;
    }
  } catch (e) {
    console.warn('Failed to load filter state:', e);
  }

  const updateYaw = (deg) => {
    yawLabel.textContent = `${deg}°`;
    sim.setRotation(deg, sim.pitch);
  };
  const updatePitch = (deg) => {
    pitchLabel.textContent = `${deg}°`;
    sim.setRotation(sim.yaw, deg);
  };
  yawInput.addEventListener('input', (e) => updateYaw(parseFloat(e.target.value)));
  pitchInput.addEventListener('input', (e) => updatePitch(parseFloat(e.target.value)));
  updateYaw(INITIAL_YAW);
  updatePitch(INITIAL_PITCH);

  // Canvas interactions: wheel zoom, drag rotate
  let dragging = false;
  let lastX = 0;
  let lastY = 0;
  canvas.addEventListener(
    'wheel',
    (e) => {
      e.preventDefault();
      // 반대 방향 줌
      sim.adjustDistance(e.deltaY * 0.001);
    },
    { passive: false },
  );
  canvas.addEventListener('contextmenu', (e) => e.preventDefault());
  canvas.addEventListener('mousedown', (e) => {
    if (e.button !== 0) return; // 좌클릭 회전
    dragging = true;
    lastX = e.clientX;
    lastY = e.clientY;
  });
  window.addEventListener('mouseup', () => {
    dragging = false;
  });
  window.addEventListener('mousemove', (e) => {
    if (!dragging) return;
    const dx = e.clientX - lastX;
    const dy = e.clientY - lastY;
    lastX = e.clientX;
    lastY = e.clientY;
    sim.adjustRotation(dx, dy); // 지구본 스타일: 드래그 방향대로 회전
    yawInput.value = sim.yaw.toFixed(0);
    pitchInput.value = sim.pitch.toFixed(0);
    updateYaw(sim.yaw);
    updatePitch(sim.pitch);
  });

  // WASD pan
  const keyMap = { KeyW: 'w', KeyA: 'a', KeyS: 's', KeyD: 'd' };
  window.addEventListener('keydown', (e) => {
    if (e.altKey && e.code === 'KeyC') {
      e.preventDefault();
      captureSnapshot();
      return;
    }
    const k = keyMap[e.code];
    if (k) sim.setKeyState(k, true);
    if (e.code === 'Space') {
      e.preventDefault();
      if (sim.speed > 0) {
        sim.setSpeed(0);
      } else {
        sim.setSpeed(lastNonZeroSpeed || 1);
      }
      updateSpeedDisplay(sim.speed);
    }
    if (!['INPUT', 'TEXTAREA'].includes(e.target.tagName)) {
      const speedMap = { Digit0: 0, Digit1: 1, Digit2: 2, Digit5: 5 };
      if (e.code in speedMap) {
        const s = speedMap[e.code];
        sim.setSpeed(s);
        if (s > 0) lastNonZeroSpeed = s;
        updateSpeedDisplay(s);
      }
    }
  });
  window.addEventListener('keyup', (e) => {
    const k = keyMap[e.code];
    if (k) sim.setKeyState(k, false);
  });

  saveBtn.addEventListener('click', () => {
    const data = { params: sim.params, gridSize: sim.gridSize };
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'wigle-u-3d.json';
    a.click();
    URL.revokeObjectURL(url);
  });

  loadInput.addEventListener('change', async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    const text = await file.text();
    const parsed = JSON.parse(text);
    if (parsed.params) {
      Object.entries(parsed.params).forEach(([k, v]) => {
        const num = Number(v);
        const value = Number.isNaN(num) ? v : num;
        sim.updateParam(k, value);
        paramsContainer.querySelectorAll(`input[data-key="${k}"]`).forEach((input) => {
          input.value = value;
        });
        updateValueLabel(k, value);
      });
    }
    if (parsed.gridSize) {
      gridSizeSelect.value = parsed.gridSize;
      await sim.resizeGrid(parsed.gridSize);
    }
    if (typeof parsed.paletteMode === 'number') {
      sim.updateParam('paletteMode', parsed.paletteMode);
      paletteSelect.value = parsed.paletteMode;
      updateValueLabel('paletteMode', parsed.paletteMode);
    }
    updateUIFromParams(sim.params);
  });

  sim.onFps = (fps) => {
    fpsLabel.textContent = fps.toFixed(0);
  };
}

function updateValueLabel(key, value) {
  const el = document.getElementById(`value-${key}`);
  if (el) el.textContent = value.toFixed(4);
}

function updateUIFromParams(params) {
  Object.entries(params).forEach(([k, v]) => updateValueLabel(k, v));
  if (paletteSelect) {
    paletteSelect.value = params.paletteMode ?? 0;
  }
  if (rayStepsSelect) {
    rayStepsSelect.value = params.raySteps ?? 96;
  }
  if (neighborModeSelect) {
    neighborModeSelect.value = params.neighborMode ?? 6;
  }

  // Update energy filter checkboxes
  const filterBits = params.energyRangeFilters ?? 0b1111;
  if (filterLowCheckbox) filterLowCheckbox.checked = (filterBits & 0b0001) !== 0;
  if (filterMidLowCheckbox) filterMidLowCheckbox.checked = (filterBits & 0b0010) !== 0;
  if (filterMidHighCheckbox) filterMidHighCheckbox.checked = (filterBits & 0b0100) !== 0;
  if (filterHighCheckbox) filterHighCheckbox.checked = (filterBits & 0b1000) !== 0;
}

async function ensurePermission(handle) {
  if (!handle) return false;
  if (handle.requestPermission) {
    const perm = await handle.requestPermission({ mode: 'readwrite' });
    return perm === 'granted';
  }
  if (handle.queryPermission) {
    const perm = await handle.queryPermission({ mode: 'readwrite' });
    return perm === 'granted';
  }
  return false;
}

async function ensureCaptureDir() {
  if (!('showDirectoryPicker' in window)) {
    alert('File System Access API is not supported in this browser.');
    throw new Error('DirectoryPicker unsupported');
  }
  if (!captureDirHandle) {
    captureDirHandle = await window.showDirectoryPicker({
      id: 'wigle-u-3d-capture',
      mode: 'readwrite',
    });
  }
  const ok = await ensurePermission(captureDirHandle);
  if (!ok) {
    alert('Permission denied for the selected folder.');
    captureDirHandle = null;
    throw new Error('Permission denied');
  }
  return captureDirHandle;
}

async function writeFile(dirHandle, name, data) {
  const fileHandle = await dirHandle.getFileHandle(name, { create: true });
  const writable = await fileHandle.createWritable();
  await writable.write(data);
  await writable.close();
}

function timestampFolder() {
  const ts = new Date();
  const pad = (n) => String(n).padStart(2, '0');
  return `capture_${ts.getFullYear()}${pad(ts.getMonth() + 1)}${pad(ts.getDate())}_${pad(ts.getHours())}${pad(ts.getMinutes())}${pad(ts.getSeconds())}`;
}

async function captureSnapshot() {
  try {
    const root = await ensureCaptureDir();
    const folderName = timestampFolder();
    const subDir = await root.getDirectoryHandle(folderName, { create: true });

    const pngBlob = await new Promise((resolve) => {
      canvas.toBlob((b) => resolve(b), 'image/png');
    });
    if (!pngBlob) throw new Error('Canvas capture failed');
    await writeFile(subDir, 'snapshot.png', pngBlob);

    const payload = { params: sim.params, gridSize: sim.gridSize, paletteMode: sim.params.paletteMode };
    const json = new Blob([JSON.stringify(payload, null, 2)], { type: 'application/json' });
    await writeFile(subDir, 'params.json', json);

    console.log(`Captured to ${folderName}/snapshot.png`);
  } catch (err) {
    console.error('Capture failed:', err);
  }
}
