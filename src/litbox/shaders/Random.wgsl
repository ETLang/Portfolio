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

// ---------------------------------------------------------------------------------------------
// Ported from David Hoskin's integer-hash family. Created by David Hoskins, May 2018.
// https://www.shadertoy.com/view/XdGfRR
// Licensed under Creative Commons Attribution-ShareAlike 4.0 International
// (https://creativecommons.org/licenses/by-sa/4.0/).
//
// Naming: hash(out)(in), e.g. hash23 takes 2 inputs and produces a 3-component output. The
// original GLSL overloads each name for both a uint/uvec seed and a float/vec seed, but WGSL has
// no function overloading, so the uint/uvec overload here is suffixed "U" (e.g. hash12U) and the
// float/vec overload keeps the bare name (e.g. hash12) - otherwise this is a direct, unmodified
// port, including hash12's odd `/ 0xffffffffu` divisor (every other function normalizes with the
// `2.328306437080797e-10` constant instead) and hash44(vec4)'s vec3 return that silently drops
// the w channel - both quirks are present in the original source, kept here for fidelity.
//---------------------------------------------------------------------------------------------------------------
fn hash11U(qIn: u32) -> f32 {
    let n = qIn * vec2<u32>(1597334673u, 3812015801u);
    let q = (n.x ^ n.y) * 1597334673u;
    return f32(q) * 2.328306437080797e-10;
}

fn hash11(p: f32) -> f32 {
    let n = u32(i32(p)) * vec2<u32>(1597334673u, 3812015801u);
    let q = (n.x ^ n.y) * 1597334673u;
    return f32(q) * 2.328306437080797e-10;
}

//---------------------------------------------------------------------------------------------------------------
fn hash12U(qIn: vec2<u32>) -> f32 {
    let q = qIn * vec2<u32>(1597334673u, 3812015801u);
    let n = (q.x ^ q.y) * 1597334673u;
    return f32(n) / f32(0xffffffffu);
}

fn hash12(p: vec2<f32>) -> f32 {
    let q = vec2<u32>(vec2<i32>(p)) * vec2<u32>(1597334673u, 3812015801u);
    let n = (q.x ^ q.y) * 1597334673u;
    return f32(n) * 2.328306437080797e-10;
}

//---------------------------------------------------------------------------------------------------------------
fn hash13U(qIn: vec3<u32>) -> f32 {
    let q = qIn * vec3<u32>(1597334673u, 3812015801u, 2798796415u);
    let n = (q.x ^ q.y ^ q.z) * 1597334673u;
    return f32(n) * 2.328306437080797e-10;
}

fn hash13(p: vec3<f32>) -> f32 {
    let q = vec3<u32>(vec3<i32>(p)) * vec3<u32>(1597334673u, 3812015801u, 2798796415u);
    let n = (q.x ^ q.y ^ q.z) * 1597334673u;
    return f32(n) * 2.328306437080797e-10;
}

//---------------------------------------------------------------------------------------------------------------
fn hash14U(qIn: vec4<u32>) -> f32 {
    let q = qIn * vec4<u32>(1597334673u, 3812015801u, 2798796415u, 1979697957u);
    let n = (q.x ^ q.y ^ q.z ^ q.w) * 1597334673u;
    return f32(n) * 2.328306437080797e-10;
}

fn hash14(p: vec4<f32>) -> f32 {
    let q = vec4<u32>(vec4<i32>(p)) * vec4<u32>(1597334673u, 3812015801u, 2798796415u, 1979697957u);
    let n = (q.x ^ q.y ^ q.z ^ q.w) * 1597334673u;
    return f32(n) * 2.328306437080797e-10;
}

//---------------------------------------------------------------------------------------------------------------
fn hash21U(qIn: u32) -> vec2<f32> {
    var n = qIn * vec2<u32>(1597334673u, 3812015801u);
    n = (n.x ^ n.y) * vec2<u32>(1597334673u, 3812015801u);
    return vec2<f32>(n) * 2.328306437080797e-10;
}

fn hash21(p: f32) -> vec2<f32> {
    var n = u32(i32(p)) * vec2<u32>(1597334673u, 3812015801u);
    n = (n.x ^ n.y) * vec2<u32>(1597334673u, 3812015801u);
    return vec2<f32>(n) * 2.328306437080797e-10;
}

//---------------------------------------------------------------------------------------------------------------
fn hash22U(qIn: vec2<u32>) -> vec2<f32> {
    var q = qIn * vec2<u32>(1597334673u, 3812015801u);
    q = (q.x ^ q.y) * vec2<u32>(1597334673u, 3812015801u);
    return vec2<f32>(q) * 2.328306437080797e-10;
}

fn hash22(p: vec2<f32>) -> vec2<f32> {
    var q = vec2<u32>(vec2<i32>(p)) * vec2<u32>(1597334673u, 3812015801u);
    q = (q.x ^ q.y) * vec2<u32>(1597334673u, 3812015801u);
    return vec2<f32>(q) * 2.328306437080797e-10;
}

//---------------------------------------------------------------------------------------------------------------
fn hash23U(qIn: vec3<u32>) -> vec2<f32> {
    let q = qIn * vec3<u32>(1597334673u, 3812015801u, 2798796415u);
    let n = (q.x ^ q.y ^ q.z) * vec2<u32>(1597334673u, 3812015801u);
    return vec2<f32>(n) * 2.328306437080797e-10;
}

fn hash23(p: vec3<f32>) -> vec2<f32> {
    let q = vec3<u32>(vec3<i32>(p)) * vec3<u32>(1597334673u, 3812015801u, 2798796415u);
    let n = (q.x ^ q.y ^ q.z) * vec2<u32>(1597334673u, 3812015801u);
    return vec2<f32>(n) * 2.328306437080797e-10;
}

//---------------------------------------------------------------------------------------------------------------
fn hash31U(qIn: u32) -> vec3<f32> {
    var n = qIn * vec3<u32>(1597334673u, 3812015801u, 2798796415u);
    n = (n.x ^ n.y ^ n.z) * vec3<u32>(1597334673u, 3812015801u, 2798796415u);
    return vec3<f32>(n) * 2.328306437080797e-10;
}

fn hash31(p: f32) -> vec3<f32> {
    var n = u32(i32(p)) * vec3<u32>(1597334673u, 3812015801u, 2798796415u);
    n = (n.x ^ n.y ^ n.z) * vec3<u32>(1597334673u, 3812015801u, 2798796415u);
    return vec3<f32>(n) * 2.328306437080797e-10;
}

//---------------------------------------------------------------------------------------------------------------
fn hash32U(qIn: vec2<u32>) -> vec3<f32> {
    var n = qIn.xyx * vec3<u32>(1597334673u, 3812015801u, 2798796415u);
    n = (n.x ^ n.y ^ n.z) * vec3<u32>(1597334673u, 3812015801u, 2798796415u);
    return vec3<f32>(n) * 2.328306437080797e-10;
}

fn hash32(q: vec2<f32>) -> vec3<f32> {
    var n = vec3<u32>(vec3<i32>(q.xyx)) * vec3<u32>(1597334673u, 3812015801u, 2798796415u);
    n = (n.x ^ n.y ^ n.z) * vec3<u32>(1597334673u, 3812015801u, 2798796415u);
    return vec3<f32>(n) * 2.328306437080797e-10;
}

//---------------------------------------------------------------------------------------------------------------
fn hash33U(qIn: vec3<u32>) -> vec3<f32> {
    var q = qIn * vec3<u32>(1597334673u, 3812015801u, 2798796415u);
    q = (q.x ^ q.y ^ q.z) * vec3<u32>(1597334673u, 3812015801u, 2798796415u);
    return vec3<f32>(q) * 2.328306437080797e-10;
}

fn hash33(p: vec3<f32>) -> vec3<f32> {
    var q = vec3<u32>(vec3<i32>(p)) * vec3<u32>(1597334673u, 3812015801u, 2798796415u);
    q = (q.x ^ q.y ^ q.z) * vec3<u32>(1597334673u, 3812015801u, 2798796415u);
    return vec3<f32>(q) * 2.328306437080797e-10;
}

//---------------------------------------------------------------------------------------------------------------
fn hash44U(qIn: vec4<u32>) -> vec4<f32> {
    var q = qIn * vec4<u32>(1597334673u, 3812015801u, 2798796415u, 1979697957u);
    q = (q.x ^ q.y ^ q.z ^ q.w) * vec4<u32>(1597334673u, 3812015801u, 2798796415u, 1979697957u);
    return vec4<f32>(q) * 2.328306437080797e-10;
}

// NOTE: returns vec3, not vec4 - matches the original source exactly, which drops the w channel
// here (the uint4 overload above returns the full vec4).
fn hash44(p: vec4<f32>) -> vec3<f32> {
    var q = vec4<u32>(vec4<i32>(p)) * vec4<u32>(1597334673u, 3812015801u, 2798796415u, 1979697957u);
    q = (q.x ^ q.y ^ q.z ^ q.w) * vec4<u32>(1597334673u, 3812015801u, 2798796415u, 1979697957u);
    return vec3<f32>(q.xyz) * 2.328306437080797e-10;
}
 