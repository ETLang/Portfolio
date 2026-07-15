// Denoiser stub - see this project's denoiser plan. The actual size-argument/guided-blur
// algorithm is explicitly out of scope for this pass: this just passes mip0 of the combined
// irradiance straight through unblurred, so the pipeline keeps rendering a correct (if
// undenoised) image while that algorithm gets designed separately on top. Once real logic exists
// here, it'll also consume the combined-irradiance mip chain (mip1+), the G-Buffer's own mips,
// and filteredVariance - none of that is threaded in yet since nothing here uses it.
//
// combineAlbedoDensity is a compile-time switch (see DenoiseOperation.updateSwitches): this is
// where albedo/density get folded into the final image now - deliberately moved here (post-
// denoise) from convert_photon_irradiance_to_hdr.wgsl, since combining must happen after variance
// computation and after denoising. Kept toggleable for debugging (raw irradiance vs. final lit
// image), same as the switch on the earlier HDR-conversion stage.
#include "LitboxCommon.wgsl"

@group(1) @binding(0) var combinedIrradianceMip0: texture_2d<f32>;
@group(1) @binding(1) var albedo: texture_2d<f32>;
@group(1) @binding(2) var density: texture_2d<f32>;

@group(2) @binding(0) var output: texture_storage_2d<rgba16float, write>;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let size = textureDimensions(output);
    if (id.x >= size.x || id.y >= size.y) {
        return;
    }

    let coords = vec2<i32>(id.xy);
    let irradiance = textureLoad(combinedIrradianceMip0, coords, 0).rgb;

#ifdef COMBINE_ALBEDO_DENSITY
    let albedoSample = textureLoad(albedo, coords, 0).rgb;
    let densitySample = textureLoad(density, coords, 0).r / DENSITY_SCALE;
    textureStore(output, coords, vec4<f32>(irradiance * albedoSample * densitySample, 1.0));
#else
    textureStore(output, coords, vec4<f32>(irradiance, 1.0));
#endif
}
