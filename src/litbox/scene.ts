// TypeScript mirror of the JSON schema produced by LitboxDemoSceneExporter.cs

export interface Vector2 {
    x: number;
    y: number;
}

export interface Color {
    r: number;
    g: number;
    b: number;
    a: number;
}

export interface SceneSimulation {
    ownerId: number;
    width: number;
    height: number;
    raysPerFrame: number;
    integrationInterval: number;
    photonBounces: number;
}

export interface SceneObject {
    active: boolean;
    id: number;
    name: string;
    parentId: number;
    position: Vector2;
    depth: number;
    rotation: number;
    scale: Vector2;
}

export interface SceneCamera {
    ownerId: number;
    verticalSize: number;
    exposure: number;
}

export interface RaytracedObject {
    ownerId: number;
    logDensity: number;
    roughness: number;
    heightScale: number;
    albedo: Color;
    albedoMap: string;
    logDensityMap: string;
    sdfNormalMap: string;
    primitiveShape: string;
}

export interface SceneSprite {
    ownerId: number;
    layer: number; // negative = renders before additive simulation. Positive = renders after
    opacity: number;
    image: string;
    colorMod: Color; // modulates the image color to produce the final albedo
    ambient: Color;
    emissive: Color;
    simContribution: Color;
    simBlur: number;
    primitiveShape: string;
}

export interface PointLight {
    ownerId: number;
    color: Color;
    intensity: number;
    bounces: number;
}

export interface Spotlight {
    ownerId: number;
    color: Color;
    intensity: number;
    pinch: number;
    bounces: number;
}

export interface LaserLight {
    ownerId: number;
    color: Color;
    intensity: number;
    bounces: number;
}

export interface DirectionalLight {
    ownerId: number;
    color: Color;
    intensity: number;
    bounces: number;
}

export interface AmbientLight {
    ownerId: number;
    color: Color;
    intensity: number;
    bounces: number;
}

export type AnyLight = PointLight | Spotlight | LaserLight | DirectionalLight | AmbientLight;

/**
 * A 3x2 affine transform applied to the UV coordinates used to sample a texture that has
 * been packed into an atlas: `u' = a*u + b*v + c`, `v' = d*u + e*v + f`.
 */
export interface UvTransform {
    a: number;
    b: number;
    c: number;
    d: number;
    e: number;
    f: number;
}

/**
 * Maps one packed texture's name to the atlas it lives in and the UV transform needed to
 * sample it: `sample(atlasName, <u,v> * uvTransform)`.
 */
export interface TextureAtlasKey {
    textureName: string;
    atlasName: string;
    uvTransform: UvTransform;
}

interface RawTextureAtlasKey {
    textureName: string;
    atlasName: string;
    uvTransform: string;
}

/** Parses the `"[[a, b, c], [d, e, f]]"` string form of a UvTransform. */
export function parseUvTransform(raw: string): UvTransform {
    const rows = JSON.parse(raw) as number[][];
    if (rows.length !== 2 || rows[0].length !== 3 || rows[1].length !== 3) {
        throw new Error(`Litbox: malformed uvTransform "${raw}" - expected a 2x3 array.`);
    }
    const [[a, b, c], [d, e, f]] = rows;
    return { a, b, c, d, e, f };
}

export interface Scene {
    simulations: SceneSimulation[];
    objects: SceneObject[];
    cameras: SceneCamera[];
    raytraced: RaytracedObject[];
    sprites: SceneSprite[];
    pointLights: PointLight[];
    spotlights: Spotlight[];
    laserLights: LaserLight[];
    directionalLights: DirectionalLight[];
    ambientLights: AmbientLight[];
    textureAtlasKeys: TextureAtlasKey[];
}

/**
 * Parses a scene from a raw JSON string, such as one produced by
 * LitboxDemoSceneExporter.cs. Missing array fields default to empty arrays.
 */
export function parseScene(json: string): Scene {
    const data = JSON.parse(json) as Partial<{
        simulations: SceneSimulation[];
        objects: SceneObject[];
        cameras: SceneCamera[];
        raytraced: RaytracedObject[];
        sprites: SceneSprite[];
        pointLights: PointLight[];
        spotlights: Spotlight[];
        laserLights: LaserLight[];
        directionalLights: DirectionalLight[];
        ambientLights: AmbientLight[];
        textureAtlasKeys: RawTextureAtlasKey[];
    }>;

    return {
        simulations: data.simulations ?? [],
        objects: data.objects ?? [],
        cameras: data.cameras ?? [],
        raytraced: data.raytraced ?? [],
        sprites: data.sprites ?? [],
        pointLights: data.pointLights ?? [],
        spotlights: data.spotlights ?? [],
        laserLights: data.laserLights ?? [],
        directionalLights: data.directionalLights ?? [],
        ambientLights: data.ambientLights ?? [],
        textureAtlasKeys: (data.textureAtlasKeys ?? []).map(key => ({
            textureName: key.textureName,
            atlasName: key.atlasName,
            uvTransform: parseUvTransform(key.uvTransform),
        })),
    };
}

/**
 * Loads and parses a scene from a File (e.g. from a file input or drag-and-drop).
 */
export async function loadSceneFromFile(file: File): Promise<Scene> {
    return parseScene(await file.text());
}
