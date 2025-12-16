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
  WORKGROUP_SIZE_X,
  WORKGROUP_SIZE_Y,
  WORKGROUP_SIZE_Z,
} from '../config/constants.js';
import {
  getComputeShader,
  getReduceShader,
  getRenderShader,
} from '../shaders/shaderLoader.js';
import { packSimParams } from '../utils/bufferUtils.js';
import { createFieldTexture, seedTexture, createReduceTexture } from '../utils/textureUtils.js';

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

    // Create kernel weight lookup texture for optimization
    this.kernelWeightTexture = this.#createKernelWeightLUT();

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
      if (key === 'paletteMode') {
        // paletteMode lives in the camera vec4 slot; update immediately
        this.#writeParamsBuffer();
        return;
      }
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
    for (const tex of this.fieldTextures) {
      seedTexture(this.device, tex, this.gridSize, SEED_ENERGY_MAX);
    }
  }

  #createFieldTextures() {
    this.fieldTextures = [0, 1].map(() => createFieldTexture(this.device, this.gridSize));
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
      code: getComputeShader(KERNEL_SIZE, WORKGROUP_SIZE_X, WORKGROUP_SIZE_Y, WORKGROUP_SIZE_Z),
    });

    this.computePipeline = this.device.createComputePipeline({
      layout: 'auto',
      compute: {
        module: computeModule,
        entryPoint: 'main',
      },
    });

    const renderModule = this.device.createShaderModule({
      code: getRenderShader(),
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
      code: getReduceShader(WORKGROUP_SIZE_X, WORKGROUP_SIZE_Y, WORKGROUP_SIZE_Z),
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
        { binding: 3, resource: this.kernelWeightTexture.createView({ dimension: '1d' }) },
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
    const camera = {
      yaw: this.yaw,
      pitch: this.pitch,
      distance: this.distance,
      offsetX: this.offsetX,
      offsetY: this.offsetY,
    };
    const buffer = packSimParams(this.params, this.gridSize, camera);
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
    const gx = Math.ceil(this.gridSize / WORKGROUP_SIZE_X);
    const gy = Math.ceil(this.gridSize / WORKGROUP_SIZE_Y);
    const gz = Math.ceil(this.gridSize / WORKGROUP_SIZE_Z);
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
      const gx = Math.ceil(to / WORKGROUP_SIZE_X);
      const gy = Math.ceil(to / WORKGROUP_SIZE_Y);
      const gz = Math.ceil(to / WORKGROUP_SIZE_Z);
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

  /**
   * Create 1D lookup texture for kernel weight function
   * Pre-computes exp(-2*t*t) for 256 samples to avoid expensive math in shader
   *
   * @returns {GPUTexture} 1D texture with pre-computed weights
   * @private
   */
  #createKernelWeightLUT() {
    const size = 256;
    const data = new Float32Array(size * 4); // rgba32float = 4 floats per pixel

    // Pre-compute weight values: exp(-2 * t * t) where t ∈ [0, 1]
    for (let i = 0; i < size; i++) {
      const t = i / (size - 1); // Normalize to [0, 1]
      const weight = Math.exp(-2.0 * t * t);

      // Store in red channel
      data[i * 4] = weight;
      data[i * 4 + 1] = 0.0;
      data[i * 4 + 2] = 0.0;
      data[i * 4 + 3] = 1.0;
    }

    const texture = this.device.createTexture({
      dimension: '1d',
      size: { width: size, height: 1 },
      format: 'rgba32float',
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
    });

    this.device.queue.writeTexture(
      { texture },
      data,
      { bytesPerRow: size * 4 * 4 },
      { width: size, height: 1 }
    );

    return texture;
  }
}
