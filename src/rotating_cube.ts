// Based on https://austin-eng.com/webgpu-samples/samples/rotatingCube
import { mat4, vec3 } from 'gl-matrix';

export class RotatingCube {
    private canvas: HTMLCanvasElement;
    private adapter!: GPUAdapter;
    private device!: GPUDevice;
    private context!: GPUCanvasContext;
    private presentationFormat!: GPUTextureFormat;
    private presentationSize!: [number, number];

    private renderPipeline!: GPURenderPipeline;
    private vertexBuffer!: GPUBuffer;
    private indexBuffer!: GPUBuffer;
    private uniformBuffer!: GPUBuffer;
    private uniformBindGroup!: GPUBindGroup;
    private depthTexture!: GPUTexture;

    private readonly cubeVertexSize = 4 * 8; // 4 bytes * (4 pos + 4 color)
    private readonly cubePositionOffset = 0;
    private readonly cubeColorOffset = 4 * 4; // 4 bytes * 4 pos
    private readonly cubeIndexCount = 36;

    constructor(canvas?: HTMLCanvasElement) {
        this.canvas = canvas || document.createElement('canvas');
        if (!canvas) {
            document.body.appendChild(this.canvas);
        }
    }

    public async start(): Promise<void> {
        if (!await this.initWebGPU()) {
            return;
        }

        this.configureCanvas();
        this.createPipeline();
        this.createBuffers();

        this.render = this.render.bind(this);
        requestAnimationFrame(this.render);
    }

    private async initWebGPU(): Promise<boolean> {
        try {
            if (!navigator.gpu) {
                console.error("WebGPU not supported on this browser.");
                return false;
            }

            const adapter = await navigator.gpu.requestAdapter();
            if (!adapter) {
                console.error("No appropriate GPUAdapter found.");
                return false;
            }
            this.adapter = adapter;
            this.device = await this.adapter.requestDevice();
        } catch (error) {
            console.error("Error initializing WebGPU:", error);
            return false;
        }
        return true;
    }

    private configureCanvas(): void {
        const context = this.canvas.getContext('webgpu');
        if (!context) {
            throw new Error("Could not get WebGPU context from canvas.");
        }
        this.context = context;
        this.presentationFormat = navigator.gpu.getPreferredCanvasFormat();
        this.presentationSize = [this.canvas.width, this.canvas.height];
        this.context.configure({
            device: this.device,
            format: this.presentationFormat,
            alphaMode: 'premultiplied',
        });
    }

    private createPipeline(): void {
        const shaderModule = this.device.createShaderModule({
            code: `
                struct Uniforms {
                    modelViewProjectionMatrix : mat4x4<f32>,
                }
                @binding(0) @group(0) var<uniform> uniforms : Uniforms;

                struct VertexOutput {
                    @builtin(position) position : vec4<f32>,
                    @location(0) fragColor : vec4<f32>,
                }

                @vertex
                fn vertex_main(
                    @location(0) position : vec4<f32>,
                    @location(1) color : vec4<f32>
                ) -> VertexOutput {
                    var output : VertexOutput;
                    output.position = uniforms.modelViewProjectionMatrix * position;
                    output.fragColor = color;
                    return output;
                }

                @fragment
                fn fragment_main(
                    @location(0) fragColor : vec4<f32>
                ) -> @location(0) vec4<f32> {
                    return fragColor;
                }
            `,
        });

        const pipelineLayout = this.device.createPipelineLayout({
            bindGroupLayouts: [
                this.device.createBindGroupLayout({
                    entries: [
                        {
                            binding: 0,
                            visibility: GPUShaderStage.VERTEX,
                            buffer: { type: 'uniform' },
                        },
                    ],
                }),
            ],
        });

        this.renderPipeline = this.device.createRenderPipeline({
            layout: pipelineLayout,
            vertex: {
                module: shaderModule,
                entryPoint: 'vertex_main',
                buffers: [
                    {
                        arrayStride: this.cubeVertexSize,
                        attributes: [
                            {
                                // position
                                shaderLocation: 0,
                                offset: this.cubePositionOffset,
                                format: 'float32x4',
                            },
                            {
                                // color
                                shaderLocation: 1,
                                offset: this.cubeColorOffset,
                                format: 'float32x4',
                            },
                        ],
                    },
                ],
            },
            fragment: {
                module: shaderModule,
                entryPoint: 'fragment_main',
                targets: [{ format: this.presentationFormat }],
            },
            primitive: {
                topology: 'triangle-list',
                cullMode: 'back',
            },
            depthStencil: {
                depthWriteEnabled: true,
                depthCompare: 'less',
                format: 'depth24plus',
            },
        });
    }

    private createBuffers(): void {
        const cubeVertexArray = new Float32Array([
            // float4 position, float4 color
            //g c b w
            1, -1, 1, 1,    1, 0, 1, 1, //
           -1, -1, 1, 1,    0, 0, 1, 1, //
           -1, 1, 1, 1,     0, 1, 1, 1, //
            1, 1, 1, 1,     1, 1, 1, 1, //
            1, -1, -1, 1,   1, 0, 0, 1, //
           -1, -1, -1, 1,   0, 0, 0, 1, //
           -1, 1, -1, 1,    0, 1, 0, 1, //
            1, 1, -1, 1,    1, 1, 0, 1, //

        ]);
        this.vertexBuffer = this.device.createBuffer({
            size: cubeVertexArray.byteLength,
            usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
            mappedAtCreation: true,
        });
        new Float32Array(this.vertexBuffer.getMappedRange()).set(cubeVertexArray);
        this.vertexBuffer.unmap();
        
        const indexArray = new Uint16Array([
            // front
            0, 2, 1,
            0, 3, 2,
            // back
            4, 5, 6,
            4, 6, 7,
            // top
            2, 3, 7,
            2, 7, 6,
            // bottom
            0, 1, 5,
            0, 5, 4,
            // right
            3, 0, 4,
            3, 4, 7,
            // left
            2, 5, 1,
            2, 6, 5,
        ]);

        this.indexBuffer = this.device.createBuffer({
            size: indexArray.byteLength,
            usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST,
            mappedAtCreation: true,
        });
        new Uint16Array(this.indexBuffer.getMappedRange()).set(indexArray);
        this.indexBuffer.unmap();


        this.depthTexture = this.device.createTexture({
            size: this.presentationSize,
            format: 'depth24plus',
            usage: GPUTextureUsage.RENDER_ATTACHMENT,
        });

        this.uniformBuffer = this.device.createBuffer({
            size: 4 * 16, // 4x4 matrix
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        this.uniformBindGroup = this.device.createBindGroup({
            layout: this.renderPipeline.getBindGroupLayout(0),
            entries: [
                {
                    binding: 0,
                    resource: { buffer: this.uniformBuffer },
                },
            ],
        });
    }

    private getTransformationMatrix(): Float32Array {
      const viewMatrix = mat4.create();
      mat4.translate(viewMatrix, viewMatrix, vec3.fromValues(0, 0, -4));
      const now = Date.now() / 1000;
      mat4.rotate(
        viewMatrix,
        viewMatrix,
        1,
        vec3.fromValues(Math.sin(now), Math.cos(now), 0)
      );
  
      const projectionMatrix = mat4.create();
      const aspect = this.canvas.height > 0 ? this.canvas.width / this.canvas.height : 1;
      mat4.perspective(
        projectionMatrix,
        (2 * Math.PI) / 5,
        aspect,
        1,
        100.0
      );
  
      const modelViewProjectionMatrix = mat4.create();
      mat4.multiply(modelViewProjectionMatrix, projectionMatrix, viewMatrix);
  
      return modelViewProjectionMatrix as Float32Array;
    }

    public render(): void {
        // If the device is not ready, we can't render.
        // This can happen if a resize event comes in before initialization.
        if (!this.device) return;

        if (this.canvas.width !== this.presentationSize[0] || this.canvas.height !== this.presentationSize[1]) {
            this.presentationSize = [this.canvas.width, this.canvas.height];
            this.depthTexture.destroy();
            this.depthTexture = this.device.createTexture({
                size: this.presentationSize,
                format: 'depth24plus',
                usage: GPUTextureUsage.RENDER_ATTACHMENT,
            });
        }

        const transformationMatrix = this.getTransformationMatrix();
        this.device.queue.writeBuffer(
            this.uniformBuffer,
            0,
            transformationMatrix.buffer,
            transformationMatrix.byteOffset,
            transformationMatrix.byteLength
        );

        const commandEncoder = this.device.createCommandEncoder();
        const textureView = this.context.getCurrentTexture().createView();

        const renderPassDescriptor: GPURenderPassDescriptor = {
            colorAttachments: [
                {
                    view: textureView,
                    clearValue: { r: 0.0, g: 0.0, b: 0.0, a: 0.0 },
                    loadOp: 'clear',
                    storeOp: 'store',
                },
            ],
            depthStencilAttachment: {
                view: this.depthTexture.createView(),
                depthClearValue: 1.0,
                depthLoadOp: 'clear',
                depthStoreOp: 'store',
            },
        };

        const passEncoder = commandEncoder.beginRenderPass(renderPassDescriptor);
        passEncoder.setPipeline(this.renderPipeline);
        passEncoder.setBindGroup(0, this.uniformBindGroup);
        passEncoder.setVertexBuffer(0, this.vertexBuffer);
        passEncoder.setIndexBuffer(this.indexBuffer, 'uint16');
        passEncoder.drawIndexed(this.cubeIndexCount);
        passEncoder.end();

        this.device.queue.submit([commandEncoder.finish()]);

        requestAnimationFrame(this.render);
    }
}
