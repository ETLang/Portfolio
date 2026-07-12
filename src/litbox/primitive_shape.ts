// erasableSyntaxOnly forbids `enum` - matches the shapeId encoding shared by sprite.wgsl and raytraced_gbuffer.wgsl.
export const PRIMITIVE_SHAPE_ID: Record<string, number> = { '': 0, rect: 1, ellipse: 2 };

/** Resolves a scene-authored primitiveShape string to its shader-side id, warning and falling back to unspecified (0) on an unrecognized value. */
export function resolvePrimitiveShapeId(shape: string, ownerId: number): number {
    const shapeId = PRIMITIVE_SHAPE_ID[shape];
    if (shapeId === undefined) {
        console.warn(`Litbox: unrecognized primitiveShape "${shape}" on owner ${ownerId}; treating as unspecified.`);
        return PRIMITIVE_SHAPE_ID[''];
    }
    return shapeId;
}
