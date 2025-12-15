import { DEFAULT_PARAMS } from '../config/defaults.js';
import {
  DEFAULT_GRID_SIZE,
  GLOBAL_AVG_INTERVAL,
  KERNEL_SIZE,
  SEED_ENERGY_MAX,
  CAMERA_BOUNDS,
  INITIAL_DISTANCE,
  PAN_SPEED,
  ROTATE_SENSITIVITY,
} from '../config/constants.js';

export class WebGPUSimulation3D {
  constructor({
    canvas,
    initialParams = DEFAULT_PARAMS,
    gridSize = DEFAULT_GRID_SIZE,
    yaw = 0,
    pitch = 0,
    distance = INITIAL_DISTANCE,
  }) {
    this.canvas = canvas;
    this.params = { ...initialParams };
    this.gridSize = gridSize;
    this.yaw = yaw;
    this.pitch = pitch;
    this.distance = distance;
    this.offsetX = 0;
    this.offsetY = 0;
    this.target = { x: 0.5, y: 0.5, z: 0.5 };

    this.adapter = null;
    this.device = null;
    this.context = null;
    this.format = null;

    this.fieldTextures = [];
    this.currentIndex = 0;
    this.sampler = null;

    this.paramBuffer = null;

    this.computePipeline = null;
    this.renderPipeline = null;
    this.reducePipeline = null;

    this.speed = 1;
    this.frameCounter = 0;

    this.frameId = null;
    this.lastFpsUpdate = performance.now();
    this.frameCount = 0;
    this.onFps = null;

    this.reduceBuffer = null;
    this.reduceBufferSize = 0;
    this.isReducing = false;
    this.readbackBuffer = null;
    this.reduceTextures = [];
    this.reduceParamBuffer = null;

    this.keyState = { w: false, a: false, s: false, d: false };
    this.lastStepTime = performance.now();
  }

  async init() {
    if (!navigator.gpu) {
      throw new Error('WebGPU not supported in this browser.');
    }
    this.adapter = await navigator.gpu.requestAdapter();
    this.device = await this.adapter.requestDevice();

    this.context = this.canvas.getContext('webgpu');
    this.format = navigator.gpu.getPreferredCanvasFormat();
    this.context.configure({
      device: this.device,
      format: this.format,
      alphaMode: 'premultiplied',
    });

    this.sampler = this.device.createSampler({
      magFilter: 'linear',
      minFilter: 'linear',
      addressModeU: 'repeat',
      addressModeV: 'repeat',
      addressModeW: 'repeat',
    });

    this.paramBuffer = this.device.createBuffer({
      size: 128, // up to 8 vec4 blocks * 16 bytes
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    this.#createPipelines();
    this.#createFieldTextures();
    this.#createReduceResources();
    this.reseed();

    this.#resizeCanvas();
    window.addEventListener('resize', () => this.#resizeCanvas());
    this.#writeParamsBuffer();
    this.#start();
  }

  async resizeGrid(size) {
    this.gridSize = size;
    this.#createFieldTextures();
    this.#createReduceResources();
    this.reseed();
    this.#writeParamsBuffer();
  }

  updateParam(key, value) {
    if (key in this.params) {
      this.params[key] = value;
      this.#writeParamsBuffer();
    }
  }

  setRotation(yawDeg, pitchDeg) {
    if (!this.device) {
      // init not ready yet
      this.yaw = yawDeg;
      this.pitch = pitchDeg;
      return;
    }
    this.yaw = yawDeg;
    this.pitch = pitchDeg;
    this.#writeParamsBuffer();
  }

  adjustRotation(dx, dy) {
    this.yaw += dx * ROTATE_SENSITIVITY * (180 / Math.PI);
    this.pitch += dy * ROTATE_SENSITIVITY * (180 / Math.PI);
    const limit = 179.0;
    this.pitch = Math.max(-limit, Math.min(limit, this.pitch));
    this.#writeParamsBuffer();
  }

  adjustDistance(delta) {
    this.distance = Math.min(CAMERA_BOUNDS.max, Math.max(CAMERA_BOUNDS.min, this.distance * (1 + delta)));
    this.#writeParamsBuffer();
  }

  setKeyState(key, value) {
    if (key in this.keyState) {
      this.keyState[key] = value;
    }
  }

  setSpeed(speed) {
    this.speed = speed;
  }

  reseed() {
    const size = this.gridSize;
    const total = size * size * size * 4;
    const data = new Float32Array(total);
    for (let i = 0; i < total; i += 4) {
      data[i] = Math.random() * SEED_ENERGY_MAX;
    }
    for (const tex of this.fieldTextures) {
      this.device.queue.writeTexture(
        { texture: tex },
        data,
        { bytesPerRow: size * 4 * 4, rowsPerImage: size },
        { width: size, height: size, depthOrArrayLayers: size },
      );
    }
  }

  #createFieldTextures() {
    const usage =
      GPUTextureUsage.TEXTURE_BINDING |
      GPUTextureUsage.STORAGE_BINDING |
      GPUTextureUsage.COPY_DST |
      GPUTextureUsage.RENDER_ATTACHMENT;
    const descriptor = {
      dimension: '3d',
      size: { width: this.gridSize, height: this.gridSize, depthOrArrayLayers: this.gridSize },
      format: 'rgba16float',
      usage,
    };
    this.fieldTextures = [0, 1].map(() => this.device.createTexture(descriptor));
    this.currentIndex = 0;
  }

  #createReduceResources() {
    // build downsample chain of textures down to 1x1x1
    this.reduceChain = [];
    this.reduceTextures = [];
    let size = this.gridSize;
    while (size > 1) {
      const next = Math.max(1, Math.floor(size / 2));
      this.reduceChain.push({ from: size, to: next });
      const tex = this.device.createTexture({
        dimension: '3d',
        size: { width: next, height: next, depthOrArrayLayers: next },
        format: 'r32float',
        usage:
          GPUTextureUsage.STORAGE_BINDING |
          GPUTextureUsage.TEXTURE_BINDING |
          GPUTextureUsage.COPY_SRC,
      });
      this.reduceTextures.push(tex);
      size = next;
    }

    this.reduceParamBuffer = this.device.createBuffer({
      size: 16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    this.readbackBuffer = this.device.createBuffer({
      size: 256,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });
  }

  #createPipelines() {
    const computeModule = this.device.createShaderModule({
      code: this.#computeShader(),
    });

    this.computePipeline = this.device.createComputePipeline({
      layout: 'auto',
      compute: {
        module: computeModule,
        entryPoint: 'main',
      },
    });

    const renderModule = this.device.createShaderModule({
      code: this.#renderShader(),
    });

    this.renderPipeline = this.device.createRenderPipeline({
      layout: 'auto',
      vertex: {
        module: renderModule,
        entryPoint: 'vs',
      },
      fragment: {
        module: renderModule,
        entryPoint: 'fs',
        targets: [{ format: this.format }],
      },
      primitive: { topology: 'triangle-list' },
    });

    const reduceModule = this.device.createShaderModule({
      code: this.#reduceShader(),
    });
    this.reducePipeline = this.device.createComputePipeline({
      layout: 'auto',
      compute: { module: reduceModule, entryPoint: 'main' },
    });
  }

  #createComputeBindGroup(readTexture, writeTexture) {
    return this.device.createBindGroup({
      layout: this.computePipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.paramBuffer } },
        { binding: 1, resource: readTexture.createView({ dimension: '3d' }) },
        { binding: 2, resource: writeTexture.createView({ dimension: '3d' }) },
      ],
    });
  }

  #createRenderBindGroup(texture) {
    return this.device.createBindGroup({
      layout: this.renderPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: this.sampler },
        { binding: 1, resource: texture.createView({ dimension: '3d' }) },
        { binding: 2, resource: { buffer: this.paramBuffer } },
      ],
    });
  }

  #createReduceBindGroup(input, output, toSize) {
    const outSize = new Uint32Array([toSize, toSize, toSize, 0]);
    this.device.queue.writeBuffer(this.reduceParamBuffer, 0, outSize.buffer);
    return this.device.createBindGroup({
      layout: this.reducePipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.reduceParamBuffer } },
        { binding: 1, resource: input.createView({ dimension: '3d' }) },
        { binding: 2, resource: output.createView({ dimension: '3d' }) },
      ],
    });
  }

  #writeParamsBuffer() {
    const size = this.gridSize;
    const inv = 1 / size;
    const buffer = new ArrayBuffer(96);
    const u32 = new Uint32Array(buffer);
    const f32 = new Float32Array(buffer);

    // dims vec4<u32>
    u32[0] = size;
    u32[1] = size;
    u32[2] = size;
    u32[3] = 0;

    // inner vec4<f32>
    f32[4] = this.params.innerRadius;
    f32[5] = this.params.innerStrength;
    f32[6] = this.params.outerRadius;
    f32[7] = this.params.outerStrength;

    // growthA vec4<f32>
    f32[8] = this.params.growthCenter;
    f32[9] = this.params.growthWidth;
    f32[10] = this.params.growthRate;
    f32[11] = this.params.suppressionFactor;

    // economy vec4<f32>
    f32[12] = this.params.globalAverage || 0.0;
    f32[13] = this.params.decayRate;
    f32[14] = this.params.diffusionRate;
    f32[15] = this.params.fissionThreshold;

    // instab vec4<f32>
    f32[16] = this.params.instabilityFactor;
    f32[17] = inv;
    f32[18] = inv;
    f32[19] = inv;

    // misc vec4<f32> (yaw, pitch, distance, seed)
    f32[20] = (this.yaw * Math.PI) / 180;
    f32[21] = (this.pitch * Math.PI) / 180;
    f32[22] = this.distance;
    f32[23] = Math.random();

    // camera vec4<f32> (offsetX, offsetY, time, unused)
    f32[24] = this.offsetX;
    f32[25] = this.offsetY;
    f32[26] = performance.now() * 0.001;
    f32[27] = 0.0;

    this.device.queue.writeBuffer(this.paramBuffer, 0, buffer);
  }

  #start() {
    const loop = () => {
      this.#step();
      this.frameId = requestAnimationFrame(loop);
    };
    loop();
  }

  #step() {
    const now = performance.now();
    const dt = Math.min(0.05, (now - this.lastStepTime) * 0.001);
    this.#updateCamera(dt);
    this.lastStepTime = now;

    if (this.speed > 0) {
      for (let i = 0; i < this.speed; i++) {
        this.#computePass();
      }
    }
    this.frameCounter++;
    if (this.frameCounter % GLOBAL_AVG_INTERVAL === 0) {
      this.#reducePassAndRead();
    }
    this.#renderPass();
    this.#updateFps();
  }

  #updateCamera(dt) {
    const yawRad = (this.yaw * Math.PI) / 180;
    const forward = { x: Math.sin(yawRad), z: Math.cos(yawRad) };
    const right = { x: forward.z, z: -forward.x };
    const speed = PAN_SPEED * dt;
    if (this.keyState.w) {
      this.offsetX += forward.x * speed;
      this.offsetY += forward.z * speed;
    }
    if (this.keyState.s) {
      this.offsetX -= forward.x * speed;
      this.offsetY -= forward.z * speed;
    }
    if (this.keyState.a) {
      this.offsetX -= right.x * speed;
      this.offsetY -= right.z * speed;
    }
    if (this.keyState.d) {
      this.offsetX += right.x * speed;
      this.offsetY += right.z * speed;
    }
    const maxOffset = 0.5;
    this.offsetX = Math.max(-maxOffset, Math.min(maxOffset, this.offsetX));
    this.offsetY = Math.max(-maxOffset, Math.min(maxOffset, this.offsetY));
  }

  #computePass() {
    const readTex = this.fieldTextures[this.currentIndex];
    const writeTex = this.fieldTextures[1 - this.currentIndex];
    const bindGroup = this.#createComputeBindGroup(readTex, writeTex);

    const commandEncoder = this.device.createCommandEncoder();
    const pass = commandEncoder.beginComputePass();
    pass.setPipeline(this.computePipeline);
    pass.setBindGroup(0, bindGroup);
    const wg = 4;
    const gx = Math.ceil(this.gridSize / wg);
    const gy = Math.ceil(this.gridSize / wg);
    const gz = Math.ceil(this.gridSize / wg);
    pass.dispatchWorkgroups(gx, gy, gz);
    pass.end();
    this.device.queue.submit([commandEncoder.finish()]);
    this.currentIndex = 1 - this.currentIndex;
  }

  async #reducePassAndRead() {
    if (this.isReducing) return;
    this.isReducing = true;
    const commandEncoder = this.device.createCommandEncoder();
    let currentInput = this.fieldTextures[this.currentIndex];

    // run reduction chain
    for (let i = 0; i < this.reduceChain.length; i++) {
      const { to } = this.reduceChain[i];
      const outputTex = this.reduceTextures[i];
      const bindGroup = this.#createReduceBindGroup(currentInput, outputTex, to);
      const pass = commandEncoder.beginComputePass();
      pass.setPipeline(this.reducePipeline);
      pass.setBindGroup(0, bindGroup);
      const wg = 4;
      const gx = Math.ceil(to / wg);
      const gy = Math.ceil(to / wg);
      const gz = Math.ceil(to / wg);
      pass.dispatchWorkgroups(gx, gy, gz);
      pass.end();
      currentInput = outputTex;
    }

    // copy final 1x1x1 to buffer
    commandEncoder.copyTextureToBuffer(
      { texture: currentInput },
      { buffer: this.readbackBuffer, bytesPerRow: 256, rowsPerImage: 1 },
      { width: 1, height: 1, depthOrArrayLayers: 1 },
    );
    this.device.queue.submit([commandEncoder.finish()]);

    try {
      await this.readbackBuffer.mapAsync(GPUMapMode.READ);
      const slice = this.readbackBuffer.getMappedRange();
      const values = new Float32Array(slice.slice(0, 4));
      const avg = values[0];
      this.readbackBuffer.unmap();
      this.params.globalAverage = avg;
      this.#writeParamsBuffer();
    } finally {
      this.isReducing = false;
    }
  }

  #renderPass() {
    const texture = this.fieldTextures[this.currentIndex];
    const bindGroup = this.#createRenderBindGroup(texture);
    const encoder = this.device.createCommandEncoder();
    const view = this.context.getCurrentTexture().createView();
    const pass = encoder.beginRenderPass({
      colorAttachments: [
        {
          view,
          loadOp: 'clear',
          storeOp: 'store',
          clearValue: { r: 0, g: 0, b: 0, a: 1 },
        },
      ],
    });
    pass.setPipeline(this.renderPipeline);
    pass.setBindGroup(0, bindGroup);
    pass.draw(6, 1, 0, 0);
    pass.end();
    this.device.queue.submit([encoder.finish()]);
  }

  #updateFps() {
    const now = performance.now();
    this.frameCount += 1;
    if (now - this.lastFpsUpdate >= 500) {
      const fps = (this.frameCount * 1000) / (now - this.lastFpsUpdate);
      if (this.onFps) this.onFps(fps);
      this.frameCount = 0;
      this.lastFpsUpdate = now;
    }
  }

  #resizeCanvas() {
    const parent = this.canvas.parentElement;
    const dpr = Math.min(window.devicePixelRatio || 1, 1.5);
    const width = Math.floor(parent.clientWidth * dpr);
    const height = Math.floor(window.innerHeight * dpr);
    this.canvas.width = width;
    this.canvas.height = height;
  }

  #computeShader() {
    return /* wgsl */ `
struct SimParams {
  dims : vec4<u32>,
  inner : vec4<f32>,
  growthA : vec4<f32>,
  economy : vec4<f32>,
  instab : vec4<f32>,
  misc : vec4<f32>,
  camera : vec4<f32>,
};

@group(0) @binding(0) var<uniform> params : SimParams;
@group(0) @binding(1) var inputTex : texture_3d<f32>;
@group(0) @binding(2) var outputTex : texture_storage_3d<rgba16float, write>;

const KERNEL : i32 = ${KERNEL_SIZE};

fn wrapCoord(coord : i32, dim : i32) -> i32 {
  var v = coord % dim;
  if (v < 0) { v += dim; }
  return v;
}

fn loadEnergy(coord : vec3<i32>, dims : vec3<i32>) -> f32 {
  let wrapped = vec3<i32>(
    wrapCoord(coord.x, dims.x),
    wrapCoord(coord.y, dims.y),
    wrapCoord(coord.z, dims.z)
  );
  return textureLoad(inputTex, wrapped, 0).x;
}

fn kernelWeight(dist : f32, innerRadius : f32, innerStrength : f32, outerRadius : f32, outerStrength : f32) -> f32 {
  var weight = 0.0;
  if (dist < innerRadius) {
    let t = 1.0 - (dist / innerRadius);
    weight = weight + innerStrength * t * t;
  }
  let ringStart = innerRadius + 1.0;
  let ringEnd = outerRadius;
  if (dist > ringStart && dist < ringEnd) {
    let t = (dist - ringStart) / (ringEnd - ringStart);
    weight = weight + outerStrength * exp(-2.0 * t * t);
  }
  return weight;
}

fn growthFunction(potential : f32, currentEnergy : f32, center : f32, width : f32, threshold : f32, instability : f32) -> f32 {
  let x = (potential - center) / width;
  var bell = exp(-x * x * 0.5);
  if (currentEnergy > threshold) {
    let excess = (currentEnergy - threshold) / (1.0 - threshold);
    bell = bell - excess * instability;
  }
  return bell;
}

fn laplacian(coord : vec3<i32>, dims : vec3<i32>, current : f32) -> f32 {
  let xp = loadEnergy(coord + vec3<i32>(1, 0, 0), dims);
  let xm = loadEnergy(coord + vec3<i32>(-1, 0, 0), dims);
  let yp = loadEnergy(coord + vec3<i32>(0, 1, 0), dims);
  let ym = loadEnergy(coord + vec3<i32>(0, -1, 0), dims);
  let zp = loadEnergy(coord + vec3<i32>(0, 0, 1), dims);
  let zm = loadEnergy(coord + vec3<i32>(0, 0, -1), dims);
  return (xp + xm + yp + ym + zp + zm) - 6.0 * current;
}

fn hash31(p : vec3<u32>) -> f32 {
  var h = p.x * 0x1e35a7bdu + p.y * 0x94d049bbu + p.z * 0x5bd1e995u;
  h = (h ^ (h >> 15u)) * 0x2c1b3c6du;
  h = h ^ (h >> 12u);
  return f32(h & 0x007fffffu) / f32(0x00800000u);
}

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let dims = vec3<i32>(i32(params.dims.x), i32(params.dims.y), i32(params.dims.z));
  if (any(gid >= params.dims.xyz)) {
    return;
  }
  let coord = vec3<i32>(gid);
  let innerRadius = params.inner.x;
  let innerStrength = params.inner.y;
  let outerRadius = params.inner.z;
  let outerStrength = params.inner.w;
  let growthCenter = params.growthA.x;
  let growthWidth = params.growthA.y;
  let growthRate = params.growthA.z;
  let suppressionFactor = params.growthA.w;
  let globalAverage = params.economy.x;
  let decayRate = params.economy.y;
  let diffusionRate = params.economy.z;
  let fissionThreshold = params.economy.w;
  let instability = params.instab.x;
  let time = params.camera.z;
  let seed = params.misc.w;

  let currentEnergy = textureLoad(inputTex, coord, 0).x;

  var potential = 0.0;
  var totalWeight = 0.0;

  for (var dz : i32 = -KERNEL; dz <= KERNEL; dz = dz + 1) {
    for (var dy : i32 = -KERNEL; dy <= KERNEL; dy = dy + 1) {
      for (var dx : i32 = -KERNEL; dx <= KERNEL; dx = dx + 1) {
        let offset = vec3<i32>(dx, dy, dz);
        let dist = length(vec3<f32>(offset));
        if (dist <= outerRadius) {
          let neighbor = loadEnergy(coord + offset, dims);
          let w = kernelWeight(dist, innerRadius, innerStrength, outerRadius, outerStrength);
          potential = potential + neighbor * w;
          totalWeight = totalWeight + abs(w);
        }
      }
    }
  }

  if (totalWeight > 0.0) {
    potential = potential / totalWeight;
  }

  var growth = growthFunction(potential, currentEnergy, growthCenter, growthWidth, fissionThreshold, instability) - 0.5;
  growth = growth - globalAverage * suppressionFactor;

  let metabolism = currentEnergy * currentEnergy * decayRate;
  let diffusion = laplacian(coord, dims, currentEnergy) * diffusionRate;

  var fissionNoise = 0.0;
  if (currentEnergy > fissionThreshold) {
    let excess = (currentEnergy - fissionThreshold) / (1.0 - fissionThreshold);
    let chaos = sin((f32(coord.x + coord.y + coord.z) + time) * 0.5);
    fissionNoise = chaos * excess * 0.1;
  }

  let noise = (hash31(gid + vec3<u32>(u32(seed * 100000.0))) - 0.5) * 0.001;

  let deltaEnergy = growthRate * growth - metabolism + diffusion + fissionNoise + noise;
  var newEnergy = currentEnergy + deltaEnergy;
  newEnergy = clamp(newEnergy, 0.0, 1.0);

  textureStore(outputTex, coord, vec4<f32>(newEnergy, 0.0, 0.0, 1.0));
}
`;
  }

  #reduceShader() {
    return /* wgsl */ `
struct ReduceParams {
  outSize : vec4<u32>,
};

@group(0) @binding(0) var<uniform> reduce : ReduceParams;
@group(0) @binding(1) var inputTex : texture_3d<f32>;
@group(0) @binding(2) var outputTex : texture_storage_3d<r32float, write>;

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let outSize = reduce.outSize.xyz;
  if (any(gid >= outSize)) {
    return;
  }
  let inSize = textureDimensions(inputTex);
  let base = vec3<i32>(gid) * 2;
  var sum = 0.0;
  var count = 0.0;
  for (var z : i32 = 0; z < 2; z = z + 1) {
    for (var y : i32 = 0; y < 2; y = y + 1) {
      for (var x : i32 = 0; x < 2; x = x + 1) {
        let coord = base + vec3<i32>(x, y, z);
        if (all(coord < vec3<i32>(inSize))) {
          sum = sum + textureLoad(inputTex, coord, 0).x;
          count = count + 1.0;
        }
      }
    }
  }
  let avg = sum / max(count, 1.0);
  textureStore(outputTex, vec3<i32>(gid), vec4<f32>(avg, 0.0, 0.0, 1.0));
}
`;
  }

  #renderShader() {
    return /* wgsl */ `
struct SimParams {
  dims : vec4<u32>,
  inner : vec4<f32>,
  growthA : vec4<f32>,
  economy : vec4<f32>,
  instab : vec4<f32>,
  misc : vec4<f32>,
  camera : vec4<f32>,
};

@group(0) @binding(0) var samp : sampler;
@group(0) @binding(1) var fieldTex : texture_3d<f32>;
@group(0) @binding(2) var<uniform> params : SimParams;

struct VertexOut {
  @builtin(position) pos : vec4<f32>,
  @location(0) uv : vec2<f32>,
};

@vertex
fn vs(@builtin(vertex_index) idx : u32) -> VertexOut {
  var positions = array<vec2<f32>, 6>(
    vec2<f32>(-1.0, -1.0),
    vec2<f32>(1.0, -1.0),
    vec2<f32>(-1.0, 1.0),
    vec2<f32>(-1.0, 1.0),
    vec2<f32>(1.0, -1.0),
    vec2<f32>(1.0, 1.0)
  );
  var uv = array<vec2<f32>, 6>(
    vec2<f32>(0.0, 1.0),
    vec2<f32>(1.0, 1.0),
    vec2<f32>(0.0, 0.0),
    vec2<f32>(0.0, 0.0),
    vec2<f32>(1.0, 1.0),
    vec2<f32>(1.0, 0.0)
  );
  var out: VertexOut;
  out.pos = vec4<f32>(positions[idx], 0.0, 1.0);
  out.uv = uv[idx];
  return out;
}

fn energyGradient(energy : f32) -> vec3<f32> {
  var color: vec3<f32>;
  if (energy < 0.1) {
    color = mix(vec3<f32>(0.0, 0.0, 0.0), vec3<f32>(0.0, 0.0, 0.2), energy * 10.0);
  } else if (energy < 0.3) {
    color = mix(vec3<f32>(0.0, 0.0, 0.2), vec3<f32>(0.0, 0.3, 0.8), (energy - 0.1) * 5.0);
  } else if (energy < 0.5) {
    color = mix(vec3<f32>(0.0, 0.3, 0.8), vec3<f32>(0.0, 0.8, 1.0), (energy - 0.3) * 5.0);
  } else if (energy < 0.7) {
    color = mix(vec3<f32>(0.0, 0.8, 1.0), vec3<f32>(0.2, 1.0, 0.3), (energy - 0.5) * 5.0);
  } else if (energy < 0.85) {
    color = mix(vec3<f32>(0.2, 1.0, 0.3), vec3<f32>(1.0, 1.0, 0.0), (energy - 0.7) * 6.67);
  } else {
    color = mix(vec3<f32>(1.0, 1.0, 0.0), vec3<f32>(1.0, 1.0, 1.0), (energy - 0.85) * 6.67);
    color = color + vec3<f32>(0.2) * sin(energy * 50.0);
  }
  color = color + energy * 0.15;
  return color;
}

fn rotateDir(dir : vec3<f32>, yaw : f32, pitch : f32) -> vec3<f32> {
  let cy = cos(yaw);
  let sy = sin(yaw);
  let cp = cos(pitch);
  let sp = sin(pitch);
  let x = dir.x * cy + dir.z * sy;
  let z = -dir.x * sy + dir.z * cy;
  let y = dir.y;
  let x2 = x;
  let y2 = y * cp - z * sp;
  let z2 = y * sp + z * cp;
  return vec3<f32>(x2, y2, z2);
}

fn intersectAabb(ro : vec3<f32>, rd : vec3<f32>, minB : vec3<f32>, maxB : vec3<f32>) -> vec2<f32> {
  let invD = 1.0 / rd;
  let t0s = (minB - ro) * invD;
  let t1s = (maxB - ro) * invD;
  let tsmaller = min(t0s, t1s);
  let tbigger = max(t0s, t1s);
  let tMin = max(max(tsmaller.x, tsmaller.y), tsmaller.z);
  let tMax = min(min(tbigger.x, tbigger.y), tbigger.z);
  return vec2<f32>(tMin, tMax);
}

@fragment
fn fs(in : VertexOut) -> @location(0) vec4<f32> {
  let uv = in.uv * 2.0 - vec2<f32>(1.0, 1.0);
  let yaw = params.misc.x;
  let pitch = params.misc.y;
  let distance = params.misc.z;
  let offset = vec2<f32>(params.camera.x, params.camera.y);
  let center = vec3<f32>(0.5 + offset.x, 0.5 + offset.y, 0.5);
  let dirLocal = normalize(vec3<f32>(uv.x, uv.y, 1.0));
  let dir = rotateDir(dirLocal, yaw, pitch);
  let camDir = rotateDir(vec3<f32>(0.0, 0.0, 1.0), yaw, pitch);
  let ro = center - camDir * distance; // orbit camera outside box
  let boundsMin = vec3<f32>(0.0, 0.0, 0.0);
  let boundsMax = vec3<f32>(1.0, 1.0, 1.0);
  let hit = intersectAabb(ro, dir, boundsMin, boundsMax);
  if (hit.y < max(hit.x, 0.0)) {
    return vec4<f32>(0.0, 0.0, 0.0, 1.0);
  }
  let tStart = max(hit.x, 0.0);
  let tEnd = hit.y;
  let steps: i32 = 64;
  let dt = (tEnd - tStart) / f32(steps);
  var t = tStart;
  var maxE = 0.0;
  for (var i: i32 = 0; i < steps; i = i + 1) {
    let pos = fract(ro + dir * t);
    let e = textureSampleLevel(fieldTex, samp, pos, 0.0).x;
    if (e > maxE) { maxE = e; }
    t = t + dt;
  }
  let color = energyGradient(maxE);
  return vec4<f32>(color, 1.0);
}
`;
  }
}
