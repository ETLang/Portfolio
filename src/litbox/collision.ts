// General-purpose segment-vs-shape collision tests, in the same unit local space every raytraced
// primitive shares (see primitive_mesh.ts's QUAD_REGION_VERTICES/RECT_REGION_VERTICES/
// ELLIPSE_REGION_VERTICES, and litbox_scene.ts's own UNIT_QUAD_LOCAL_CORNERS - all confined to
// [-0.5, 0.5] on both axes regardless of an object's own scale/rotation). Deliberately returns the
// impact point and surface normal, not just a hit/miss boolean, so a caller wiring up hit effects
// (sparks, an explosion) later doesn't need to redo any of this geometry math.
//
// Segment (not point) tests are used throughout rather than a point-in-shape test at just the
// segment's endpoint, so a fast-moving object that fully crosses a thin shape between two frames
// still registers a hit instead of tunneling through it.
//
// Normals are transformed to world space via the object's own world transform's linear part
// (re-normalized afterward, not a true inverse-transpose normal matrix) - acceptable for a
// stylized 2D game effect, not physically-rigorous lighting.

import { mat4, vec4 } from 'gl-matrix';
import type { Vector2 } from './scene.ts';

export type PrimitiveShape = 'rect' | 'ellipse';

/** A hit in the shape's own local [-0.5, 0.5] space. `t` is the segment parameter (0 = localP0, 1 = localP1) at which the hit occurred. */
export interface LocalHit {
    t: number;
    point: Vector2;
    normal: Vector2;
}

/** A hit already converted to world space by testSegmentAgainstShape - see its doc comment. */
export interface WorldHit {
    point: Vector2;
    normal: Vector2;
}

function normalizeOrFallback(x: number, y: number): Vector2 {
    const length = Math.hypot(x, y);
    return length > 0 ? { x: x / length, y: y / length } : { x: 0, y: 1 };
}

/**
 * Segment (localP0 -> localP1) vs. the unit circle (radius 0.5, centered at the local origin) -
 * the local footprint every raytraced ellipse shares. Standard quadratic segment/circle solve.
 * If localP0 already lies inside (or on) the circle, reports an immediate hit at t=0 rather than
 * solving for an entry crossing that already happened before this segment started.
 */
export function segmentIntersectsUnitEllipse(localP0: Vector2, localP1: Vector2): LocalHit | null {
    const RADIUS = 0.5;
    const c = localP0.x * localP0.x + localP0.y * localP0.y - RADIUS * RADIUS;
    if (c <= 0) {
        return { t: 0, point: { ...localP0 }, normal: normalizeOrFallback(localP0.x, localP0.y) };
    }

    const dx = localP1.x - localP0.x;
    const dy = localP1.y - localP0.y;
    const a = dx * dx + dy * dy;
    if (a === 0) {
        return null; // zero-length segment starting outside the circle
    }
    const b = 2 * (localP0.x * dx + localP0.y * dy);
    const discriminant = b * b - 4 * a * c;
    if (discriminant < 0) {
        return null;
    }
    // c > 0 here (p0 outside), so the smaller root is always the entry crossing.
    const t = (-b - Math.sqrt(discriminant)) / (2 * a);
    if (t < 0 || t > 1) {
        return null;
    }
    const point = { x: localP0.x + dx * t, y: localP0.y + dy * t };
    return { t, point, normal: normalizeOrFallback(point.x, point.y) };
}

/** Outward normal of whichever unit-rect face (of the 4) is nearest to a point already known to be inside it - see segmentIntersectsUnitRect's own "already inside" case. */
function nearestRectFaceNormal(x: number, y: number): Vector2 {
    const distanceToVerticalEdge = 0.5 - Math.abs(x);
    const distanceToHorizontalEdge = 0.5 - Math.abs(y);
    if (distanceToVerticalEdge <= distanceToHorizontalEdge) {
        return { x: x >= 0 ? 1 : -1, y: 0 };
    }
    return { x: 0, y: y >= 0 ? 1 : -1 };
}

/**
 * Segment (localP0 -> localP1) vs. the unit square [-0.5, 0.5]^2 - the local footprint every
 * raytraced rect shares. Standard slab method. If localP0 already lies inside (or on) the square,
 * reports an immediate hit at t=0, same reasoning as segmentIntersectsUnitEllipse.
 */
export function segmentIntersectsUnitRect(localP0: Vector2, localP1: Vector2): LocalHit | null {
    const HALF = 0.5;
    if (Math.abs(localP0.x) <= HALF && Math.abs(localP0.y) <= HALF) {
        return { t: 0, point: { ...localP0 }, normal: nearestRectFaceNormal(localP0.x, localP0.y) };
    }

    const dx = localP1.x - localP0.x;
    const dy = localP1.y - localP0.y;
    let tMin = 0;
    let tMax = 1;
    let normal: Vector2 = { x: 0, y: 0 };

    for (const axis of ['x', 'y'] as const) {
        const origin = axis === 'x' ? localP0.x : localP0.y;
        const delta = axis === 'x' ? dx : dy;
        if (delta === 0) {
            if (origin < -HALF || origin > HALF) {
                return null; // parallel to, and outside, this axis' slab - can never cross it
            }
            continue;
        }
        let tNear = (-HALF - origin) / delta;
        let tFar = (HALF - origin) / delta;
        let nearSign = -1;
        if (tNear > tFar) {
            [tNear, tFar] = [tFar, tNear];
            nearSign = 1;
        }
        if (tNear > tMin) {
            tMin = tNear;
            normal = axis === 'x' ? { x: nearSign, y: 0 } : { x: 0, y: nearSign };
        }
        tMax = Math.min(tMax, tFar);
        if (tMin > tMax) {
            return null;
        }
    }

    // Invariant maintained above: reaching here means 0 <= tMin <= tMax <= 1.
    const point = { x: localP0.x + dx * tMin, y: localP0.y + dy * tMin };
    return { t: tMin, point, normal };
}

function transformPoint(p: Vector2, transform: mat4): Vector2 {
    const out = vec4.transformMat4(vec4.create(), vec4.fromValues(p.x, p.y, 0, 1), transform);
    return { x: out[0], y: out[1] };
}

function transformNormal(n: Vector2, transform: mat4): Vector2 {
    const out = vec4.transformMat4(vec4.create(), vec4.fromValues(n.x, n.y, 0, 0), transform);
    return normalizeOrFallback(out[0], out[1]);
}

/**
 * World-space wrapper around segmentIntersectsUnitEllipse/segmentIntersectsUnitRect: transforms
 * both segment endpoints into the shape's local space via `inverseWorldTransform`, dispatches on
 * `shape`, then maps a hit's point/normal back to world space via `worldTransform`. Takes both
 * transforms rather than inverting internally so a caller testing many segments against the same
 * shape in one frame (e.g. several tracer bullets against one UFO) can invert its world transform
 * once and reuse it, instead of re-inverting per test.
 */
export function testSegmentAgainstShape(
    worldP0: Vector2,
    worldP1: Vector2,
    worldTransform: mat4,
    inverseWorldTransform: mat4,
    shape: PrimitiveShape,
): WorldHit | null {
    const localP0 = transformPoint(worldP0, inverseWorldTransform);
    const localP1 = transformPoint(worldP1, inverseWorldTransform);
    const hit =
        shape === 'ellipse'
            ? segmentIntersectsUnitEllipse(localP0, localP1)
            : segmentIntersectsUnitRect(localP0, localP1);
    if (!hit) {
        return null;
    }
    return {
        point: transformPoint(hit.point, worldTransform),
        normal: transformNormal(hit.normal, worldTransform),
    };
}
