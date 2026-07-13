/**
 * Shared base for compute-shader operations (see CLAUDE.md's "Compute-shader operation
 * architecture"). Bind groups are fixed by convention: group 0 = uniforms, group 1 = inputs,
 * group 2 = outputs, each independently dirty-tracked and lazily rebuilt on the next execute() -
 * mirrors RaytracedResources' sharedBindGroupDirty/rebuildSharedBindGroup pattern
 * (raytraced_resources.ts). Subclasses expose bespoke, named updateUniforms/updateInputs/
 * updateOutputs methods that translate typed parameters into GPUBindGroupEntry[] and hand them to
 * setUniforms/setInputs/setOutputs below; dispatch extent is derived by the subclass (from
 * whatever it was last given via updateOutputs) and reported via setDispatchExtent, not passed to
 * execute().
 *
 * Pipeline layout is always 'auto' - each subclass's own WGSL @group/@binding declarations are the
 * single source of truth for its bind group layouts, so nothing here (or in a subclass) declares
 * a GPUBindGroupLayout by hand.
 *
 * Compile-time switches (#define/#ifdef, see shader_preprocessor.ts) are expected to change
 * extremely rarely, so a switch change is handled as a full pipeline recompile rather than a
 * keyed cache of pipelines per switch combination - see setShaderCode below. A subclass's
 * updateSwitches(...) re-runs preprocessShader(rawShaderSource, defines) itself (translating its
 * typed switch parameters into a ShaderDefines object) and hands the resulting WGSL text to
 * setShaderCode.
 */
export abstract class ComputeOperation {
    protected device: GPUDevice;
    private shaderCode: string;
    private entryPoint: string;

    private pipeline: GPUComputePipeline | null = null;
    private workgroupSize: [number, number, number] = [1, 1, 1];

    private uniformEntries: GPUBindGroupEntry[] = [];
    private inputEntries: GPUBindGroupEntry[] = [];
    private outputEntries: GPUBindGroupEntry[] = [];
    private uniformBindGroup: GPUBindGroup | null = null;
    private inputBindGroup: GPUBindGroup | null = null;
    private outputBindGroup: GPUBindGroup | null = null;
    private uniformGroupDirty = true;
    private inputGroupDirty = true;
    private outputGroupDirty = true;

    private dispatchWidth = 0;
    private dispatchHeight = 0;
    private dispatchDepth = 1;

    protected constructor(device: GPUDevice, shaderCode: string, entryPoint = 'main') {
        this.device = device;
        this.shaderCode = shaderCode;
        this.entryPoint = entryPoint;
    }

    /** Replaces group 0's bind group entries; a no-op if `entries` describes the same resources already bound. */
    protected setUniforms(entries: GPUBindGroupEntry[]): void {
        if (entriesEqual(this.uniformEntries, entries)) {
            return;
        }
        this.uniformEntries = entries;
        this.uniformGroupDirty = true;
    }

    /** Replaces group 1's bind group entries; a no-op if `entries` describes the same resources already bound. */
    protected setInputs(entries: GPUBindGroupEntry[]): void {
        if (entriesEqual(this.inputEntries, entries)) {
            return;
        }
        this.inputEntries = entries;
        this.inputGroupDirty = true;
    }

    /** Replaces group 2's bind group entries; a no-op if `entries` describes the same resources already bound. */
    protected setOutputs(entries: GPUBindGroupEntry[]): void {
        if (entriesEqual(this.outputEntries, entries)) {
            return;
        }
        this.outputEntries = entries;
        this.outputGroupDirty = true;
    }

    /** Subclasses call this from their own updateOutputs, deriving the extent from whatever output they were just given. */
    protected setDispatchExtent(width: number, height: number, depth = 1): void {
        this.dispatchWidth = width;
        this.dispatchHeight = height;
        this.dispatchDepth = depth;
    }

    /**
     * Subclasses' updateSwitches(...) call this with the result of re-running
     * preprocessShader(rawShaderSource, defines) against the subclass's own raw shader text - see
     * the class doc comment. A no-op if `code` is identical to what's already compiled (the common
     * case: updateSwitches called again with the same switch combination). Otherwise this is a full
     * recompile - no keyed pipeline cache - since switches change extremely rarely. Also forces all
     * three bind groups to rebuild next execute(): a new pipeline's 'auto' bind group layouts are
     * distinct GPUBindGroupLayout objects even when the bound resources haven't changed, so the old
     * bind groups are no longer valid against it.
     */
    protected setShaderCode(code: string): void {
        if (code === this.shaderCode) {
            return;
        }
        this.shaderCode = code;
        this.pipeline = null;
        this.uniformGroupDirty = true;
        this.inputGroupDirty = true;
        this.outputGroupDirty = true;
    }

    private ensurePipeline(): GPUComputePipeline {
        if (!this.pipeline) {
            this.workgroupSize = parseWorkgroupSize(this.shaderCode);
            const module = this.device.createShaderModule({ code: this.shaderCode });
            this.pipeline = this.device.createComputePipeline({
                layout: 'auto',
                compute: { module, entryPoint: this.entryPoint },
            });
        }
        return this.pipeline;
    }

    /** Rebuilds any dirty bind group and dispatches. A no-op until updateOutputs has established a non-zero dispatch extent. */
    public execute(encoder: GPUCommandEncoder): void {
        const pipeline = this.ensurePipeline();

        if (this.uniformGroupDirty) {
            this.uniformBindGroup = this.uniformEntries.length > 0
                ? this.device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries: this.uniformEntries })
                : null;
            this.uniformGroupDirty = false;
        }
        if (this.inputGroupDirty) {
            this.inputBindGroup = this.inputEntries.length > 0
                ? this.device.createBindGroup({ layout: pipeline.getBindGroupLayout(1), entries: this.inputEntries })
                : null;
            this.inputGroupDirty = false;
        }
        if (this.outputGroupDirty) {
            this.outputBindGroup = this.outputEntries.length > 0
                ? this.device.createBindGroup({ layout: pipeline.getBindGroupLayout(2), entries: this.outputEntries })
                : null;
            this.outputGroupDirty = false;
        }

        if (this.dispatchWidth <= 0 || this.dispatchHeight <= 0 || this.dispatchDepth <= 0) {
            return;
        }

        const pass = encoder.beginComputePass();
        pass.setPipeline(pipeline);
        if (this.uniformBindGroup) {
            pass.setBindGroup(0, this.uniformBindGroup);
        }
        if (this.inputBindGroup) {
            pass.setBindGroup(1, this.inputBindGroup);
        }
        if (this.outputBindGroup) {
            pass.setBindGroup(2, this.outputBindGroup);
        }
        const [workgroupX, workgroupY, workgroupZ] = this.workgroupSize;
        pass.dispatchWorkgroups(
            Math.ceil(this.dispatchWidth / workgroupX),
            Math.ceil(this.dispatchHeight / workgroupY),
            Math.ceil(this.dispatchDepth / workgroupZ),
        );
        pass.end();
    }
}

/** Shallow identity comparison used to skip a bind-group rebuild when updateInputs/updateOutputs/updateUniforms is called again with the same underlying resources (the common per-frame case). */
function entriesEqual(a: GPUBindGroupEntry[], b: GPUBindGroupEntry[]): boolean {
    if (a.length !== b.length) {
        return false;
    }
    for (let i = 0; i < a.length; i++) {
        if (a[i].binding !== b[i].binding || resourceIdentity(a[i].resource) !== resourceIdentity(b[i].resource)) {
            return false;
        }
    }
    return true;
}

/** GPUBindGroupEntry.resource is either a buffer-binding object ({buffer, ...}) or the resource itself (a view/sampler) - normalize to whichever underlying object identifies the binding. */
function resourceIdentity(resource: GPUBindingResource): unknown {
    return typeof resource === 'object' && resource !== null && 'buffer' in resource ? resource.buffer : resource;
}

const WORKGROUP_SIZE_RE = /@workgroup_size\(\s*(\d+)\s*(?:,\s*(\d+)\s*(?:,\s*(\d+)\s*)?)?\)/;

/** Parses a shader's own @workgroup_size(...) attribute rather than duplicating it as a separate JS constant that could silently drift - see CLAUDE.md. */
function parseWorkgroupSize(shaderCode: string): [number, number, number] {
    const match = WORKGROUP_SIZE_RE.exec(shaderCode);
    if (!match) {
        throw new Error('ComputeOperation: no @workgroup_size(...) found in shader source.');
    }
    const x = parseInt(match[1], 10);
    const y = match[2] !== undefined ? parseInt(match[2], 10) : 1;
    const z = match[3] !== undefined ? parseInt(match[3], 10) : 1;
    return [x, y, z];
}
