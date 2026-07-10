import { mat4 } from 'gl-matrix';
import type { RaytracedObject, Scene } from './scene.ts';
import type { SceneGraph } from './scene_graph.ts';
import type { TextureCache } from './texture_cache.ts';

export interface ResolvedRaytracedEntry {
    entry: RaytracedObject;
    worldTransform: mat4;
    albedoMap: GPUTexture;
    logDensityMap: GPUTexture;
    sdfNormalMap: GPUTexture;
}

/**
 * Raytraced objects are simulation *inputs* only - they are never drawn to
 * the frame buffer. This manager stays CPU-side (no GPU buffer/pipeline)
 * until the actual simulation pass is implemented, to avoid allocating
 * GPU resources nothing consumes yet.
 */
export class RaytracedResources {
    private entries: ResolvedRaytracedEntry[] = [];

    public getEntries(): readonly ResolvedRaytracedEntry[] {
        return this.entries;
    }

    public async updateFromScene(scene: Scene, sceneGraph: SceneGraph, textureCache: TextureCache): Promise<void> {
        this.entries = await Promise.all(scene.raytraced.map(async entry => ({
            entry,
            worldTransform: sceneGraph.getWorldTransform(entry.ownerId),
            albedoMap: await textureCache.resolve(entry.albedoMap, 'white'),
            logDensityMap: await textureCache.resolve(entry.logDensityMap, 'black'),
            sdfNormalMap: await textureCache.resolve(entry.sdfNormalMap, 'black'),
        })));
    }

    /**
     * Targeted CPU-side world-transform refresh for the entry owned by `ownerId`
     * (no-op if none exists). TODO: once a GPU storage buffer + simulation pass exist
     * for raytraced objects, split transform from properties the same way
     * LightResources/SpriteResources do, and also queue.writeBuffer the updated
     * transform here - there's currently nothing to upload to.
     */
    public refreshEntry(ownerId: number, sceneGraph: SceneGraph): void {
        const resolved = this.entries.find(e => e.entry.ownerId === ownerId);
        if (!resolved) {
            return;
        }
        resolved.worldTransform = sceneGraph.getWorldTransform(ownerId);
    }
}
