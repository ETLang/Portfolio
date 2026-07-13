// Ported from Random.cginc (Unity Litbox project) - a combined Tausworthe/LCG PRNG.
// https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-37-efficient-random-number-generation-and-application
//
// WGSL structs can't have member functions, so the HLSL `Random` struct's methods become free
// functions here taking a `ptr<function, Random>` to the state instead - randomNext(&rand)
// where the HLSL source would write rand.Next(). The HLSL struct also cached its last-generated
// value in a `value` field; nothing outside the struct ever read it (Next() always overwrites it
// before returning), so it's dropped here rather than carried along as dead state.

struct Random {
    state: vec4<u32>,
}

fn randomInit(seed: vec4<u32>) -> Random {
    var rand: Random;
    rand.state = seed;
    return rand;
}

fn randomTausStep(z: u32, s1: u32, s2: u32, s3: u32, m: u32) -> u32 {
    let b = ((z << s1) ^ z) >> s2;
    return ((z & m) << s3) ^ b;
}

fn randomLcgStep(z: u32, a: u32, c: u32) -> u32 {
    return a * z + c;
}

fn randomNext(rand: ptr<function, Random>) -> f32 {
    (*rand).state.x = randomTausStep((*rand).state.x, 13u, 19u, 12u, 4294967294u);
    (*rand).state.y = randomTausStep((*rand).state.y, 2u, 25u, 4u, 4294967288u);
    (*rand).state.z = randomTausStep((*rand).state.z, 3u, 11u, 17u, 4294967280u);
    (*rand).state.w = randomLcgStep((*rand).state.w, 1664525u, 1013904223u);
    let combined = (*rand).state.x ^ (*rand).state.y ^ (*rand).state.z ^ (*rand).state.w;
    return 2.3283064365387e-10 * f32(combined);
}

fn randomNextRange(rand: ptr<function, Random>, lo: f32, hi: f32) -> f32 {
    return lo + randomNext(rand) * (hi - lo);
}

fn randomNext2(rand: ptr<function, Random>) -> vec2<f32> {
    return vec2<f32>(randomNext(rand), randomNext(rand));
}

fn randomNext3(rand: ptr<function, Random>) -> vec3<f32> {
    return vec3<f32>(randomNext(rand), randomNext(rand), randomNext(rand));
}

fn randomNext4(rand: ptr<function, Random>) -> vec4<f32> {
    return vec4<f32>(randomNext(rand), randomNext(rand), randomNext(rand), randomNext(rand));
}

fn randomNextDirection(rand: ptr<function, Random>) -> vec2<f32> {
    let theta = randomNext(rand) * 2.0 * 3.141592654;
    return vec2<f32>(cos(theta), sin(theta));
}

fn randomNextCircle(rand: ptr<function, Random>) -> vec2<f32> {
    return randomNextDirection(rand) * sqrt(randomNext(rand));
}
