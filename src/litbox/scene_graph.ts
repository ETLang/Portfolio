import { mat4 } from 'gl-matrix';
import type { Scene, SceneObject } from './scene.ts';

const ROOT_PARENT_ID = -1;

/**
 * Resolves the flat, id/parentId-linked SceneObject list into cached world
 * transforms and hierarchical active/inactive state.
 */
export class SceneGraph {
    private objectsById = new Map<number, SceneObject>();
    private childrenByParentId = new Map<number, number[]>();
    private worldTransformCache = new Map<number, mat4>();
    private activeInHierarchyCache = new Map<number, boolean>();

    constructor(scene: Scene) {
        for (const obj of scene.objects) {
            this.objectsById.set(obj.id, obj);
        }
        for (const obj of scene.objects) {
            if (obj.parentId === ROOT_PARENT_ID) {
                continue;
            }
            const siblings = this.childrenByParentId.get(obj.parentId);
            if (siblings) {
                siblings.push(obj.id);
            } else {
                this.childrenByParentId.set(obj.parentId, [obj.id]);
            }
        }
    }

    public getObject(id: number): SceneObject | undefined {
        return this.objectsById.get(id);
    }

    /** Registers a newly-created object, indexing it under its (already-set) parentId. */
    public addObject(obj: SceneObject): void {
        this.objectsById.set(obj.id, obj);
        if (obj.parentId === ROOT_PARENT_ID) {
            return;
        }
        const siblings = this.childrenByParentId.get(obj.parentId);
        if (siblings) {
            siblings.push(obj.id);
        } else {
            this.childrenByParentId.set(obj.parentId, [obj.id]);
        }
    }

    /**
     * Removes `id` and its whole subtree from the graph and both caches. Returns the removed ids
     * (`id` first, then descendants depth-first) so callers can cascade-clean anything owned by
     * them. No-op returning `[]` if `id` isn't found.
     */
    public removeObject(id: number): number[] {
        const obj = this.objectsById.get(id);
        if (!obj) {
            return [];
        }
        const cascade = [id, ...this.getDescendantIds(id)];

        if (obj.parentId !== ROOT_PARENT_ID) {
            const siblings = this.childrenByParentId.get(obj.parentId);
            if (siblings) {
                const index = siblings.indexOf(id);
                if (index !== -1) {
                    siblings.splice(index, 1);
                }
            }
        }

        for (const removedId of cascade) {
            this.objectsById.delete(removedId);
            this.childrenByParentId.delete(removedId);
            this.worldTransformCache.delete(removedId);
            this.activeInHierarchyCache.delete(removedId);
        }

        return cascade;
    }

    /**
     * Moves `id` (and its whole subtree) to a new parent, invalidating its cached world transform
     * and active-in-hierarchy state. Throws if `id`/`newParentId` can't be resolved, if
     * `newParentId === id`, or if `newParentId` is a descendant of `id` (which would create a cycle).
     */
    public setParent(id: number, newParentId: number): void {
        const obj = this.objectsById.get(id);
        if (!obj) {
            throw new Error(`Litbox scene graph: cannot reparent unknown object id ${id}.`);
        }
        if (newParentId !== ROOT_PARENT_ID && !this.objectsById.has(newParentId)) {
            throw new Error(`Litbox scene graph: cannot reparent object id ${id} to unknown parent id ${newParentId}.`);
        }
        if (newParentId === id) {
            throw new Error(`Litbox scene graph: cannot reparent object id ${id} to itself.`);
        }
        if (newParentId !== ROOT_PARENT_ID && this.getDescendantIds(id).includes(newParentId)) {
            throw new Error(`Litbox scene graph: cannot reparent object id ${id} to its own descendant id ${newParentId}.`);
        }

        if (obj.parentId !== ROOT_PARENT_ID) {
            const siblings = this.childrenByParentId.get(obj.parentId);
            if (siblings) {
                const index = siblings.indexOf(id);
                if (index !== -1) {
                    siblings.splice(index, 1);
                }
            }
        }

        obj.parentId = newParentId;

        if (newParentId !== ROOT_PARENT_ID) {
            const newSiblings = this.childrenByParentId.get(newParentId);
            if (newSiblings) {
                newSiblings.push(id);
            } else {
                this.childrenByParentId.set(newParentId, [id]);
            }
        }

        this.invalidateSubtree(id);
    }

    /** Strict descendants of `id` (excludes `id` itself), depth-first, cycle-guarded. */
    public getDescendantIds(id: number): number[] {
        const result: number[] = [];
        this.collectDescendants(id, result, new Set());
        return result;
    }

    /** Clears cached world transform + active-in-hierarchy state for `id` and every descendant. */
    public invalidateSubtree(id: number): void {
        this.worldTransformCache.delete(id);
        this.activeInHierarchyCache.delete(id);
        for (const descendantId of this.getDescendantIds(id)) {
            this.worldTransformCache.delete(descendantId);
            this.activeInHierarchyCache.delete(descendantId);
        }
    }

    public getWorldTransform(id: number): mat4 {
        const cached = this.worldTransformCache.get(id);
        if (cached) {
            return cached;
        }
        return this.resolveWorldTransform(id, new Set());
    }

    public isActiveInHierarchy(id: number): boolean {
        const cached = this.activeInHierarchyCache.get(id);
        if (cached !== undefined) {
            return cached;
        }
        return this.resolveActiveInHierarchy(id, new Set());
    }

    private resolveWorldTransform(id: number, visiting: Set<number>): mat4 {
        const obj = this.objectsById.get(id);
        if (!obj) {
            console.warn(`Litbox scene graph: object id ${id} not found; using identity transform.`);
            return mat4.create();
        }

        if (visiting.has(id)) {
            console.warn(`Litbox scene graph: cycle detected involving object id ${id}; using identity transform.`);
            return mat4.create();
        }
        visiting.add(id);

        const local = mat4.create();
        mat4.translate(local, local, [obj.position.x, obj.position.y, obj.depth]);
        mat4.rotateZ(local, local, (obj.rotation * Math.PI) / 180);
        mat4.scale(local, local, [obj.scale.x, obj.scale.y, 1]);

        let world: mat4;
        if (obj.parentId === ROOT_PARENT_ID) {
            world = local;
        } else {
            const parentWorld = this.worldTransformCache.get(obj.parentId)
                ?? this.resolveWorldTransform(obj.parentId, visiting);
            world = mat4.create();
            mat4.multiply(world, parentWorld, local);
        }

        visiting.delete(id);
        this.worldTransformCache.set(id, world);
        return world;
    }

    private resolveActiveInHierarchy(id: number, visiting: Set<number>): boolean {
        const obj = this.objectsById.get(id);
        if (!obj) {
            return false;
        }
        if (visiting.has(id)) {
            console.warn(`Litbox scene graph: cycle detected involving object id ${id}; treating as inactive.`);
            return false;
        }
        visiting.add(id);

        let active = obj.active;
        if (active && obj.parentId !== ROOT_PARENT_ID) {
            active = this.activeInHierarchyCache.get(obj.parentId)
                ?? this.resolveActiveInHierarchy(obj.parentId, visiting);
        }

        visiting.delete(id);
        this.activeInHierarchyCache.set(id, active);
        return active;
    }

    private collectDescendants(id: number, result: number[], visiting: Set<number>): void {
        if (visiting.has(id)) {
            console.warn(`Litbox scene graph: cycle detected involving object id ${id} while collecting descendants.`);
            return;
        }
        visiting.add(id);
        for (const childId of this.childrenByParentId.get(id) ?? []) {
            result.push(childId);
            this.collectDescendants(childId, result, visiting);
        }
        visiting.delete(id);
    }
}
