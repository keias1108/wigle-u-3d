/**
 * Hierarchical Reduction Shader
 * Reference WGSL file: reduce.wgsl (documentation)
 */
export default /* wgsl */ `// Hierarchical Reduction Shader for Global Average Computation
//
// This shader implements a multi-pass reduction pipeline:
// Input: 32³ texture → Output: 16³ texture (average of 2×2×2 blocks)
// Repeat: 16³ → 8³ → 4³ → 2³ → 1³ (single value)
//
// Used to compute global average energy efficiently on GPU

struct ReduceParams {
  outSize : vec4<u32>,
};

@group(0) @binding(0) var<uniform> reduce : ReduceParams;
@group(0) @binding(1) var inputTex : texture_3d<f32>;
@group(0) @binding(2) var outputTex : texture_storage_3d<r32float, write>;

// Each thread processes one output cell by averaging 2×2×2 input cells
@compute @workgroup_size({{WORKGROUP_X}}, {{WORKGROUP_Y}}, {{WORKGROUP_Z}})
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
}`;
