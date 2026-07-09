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
}

/**
 * Parses a scene from a raw JSON string, such as one produced by
 * LitboxDemoSceneExporter.cs. Missing array fields default to empty arrays.
 */
export function parseScene(json: string): Scene {
    const data = JSON.parse(json) as Partial<Scene>;

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
    };
}

/**
 * Loads and parses a scene from a File (e.g. from a file input or drag-and-drop).
 */
export async function loadSceneFromFile(file: File): Promise<Scene> {
    return parseScene(await file.text());
}
