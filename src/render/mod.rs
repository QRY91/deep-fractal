use crate::app::FractalState;
use crate::gpu::GpuContext;
use crate::perturbation::IterationResult;

pub struct Renderer {
    pipeline: wgpu::RenderPipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    uniform_buffer: wgpu::Buffer,
    palette_texture: wgpu::Texture,
    palette_view: wgpu::TextureView,
    palette_sampler: wgpu::Sampler,
    iter_texture: wgpu::Texture,
    iter_view: wgpu::TextureView,
    bind_group: wgpu::BindGroup,
    iter_width: u32,
    iter_height: u32,
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct Uniforms {
    resolution: [f32; 2],
    center_hi: [f32; 2],
    center_lo: [f32; 2],
    zoom: f32,
    max_iter: u32,
    color_shift: f32,
    mode: u32, // 0 = direct GPU, 1 = perturbation (read iter texture)
    _pad: [f32; 2],
}

impl Renderer {
    pub fn new(gpu: &GpuContext) -> Self {
        let shader = gpu
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("fractal shader"),
                source: wgpu::ShaderSource::Wgsl(include_str!("../../shaders/fractal.wgsl").into()),
            });

        let bind_group_layout =
            gpu.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("fractal bgl"),
                    entries: &[
                        // Uniform buffer
                        wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Uniform,
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        // Palette texture (1D)
                        wgpu::BindGroupLayoutEntry {
                            binding: 1,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Texture {
                                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                                view_dimension: wgpu::TextureViewDimension::D1,
                                multisampled: false,
                            },
                            count: None,
                        },
                        // Palette sampler
                        wgpu::BindGroupLayoutEntry {
                            binding: 2,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                            count: None,
                        },
                        // Iteration texture (2D, Rg32Float)
                        wgpu::BindGroupLayoutEntry {
                            binding: 3,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Texture {
                                sample_type: wgpu::TextureSampleType::Float { filterable: false },
                                view_dimension: wgpu::TextureViewDimension::D2,
                                multisampled: false,
                            },
                            count: None,
                        },
                    ],
                });

        let pipeline_layout =
            gpu.device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("fractal pipeline layout"),
                    bind_group_layouts: &[&bind_group_layout],
                    push_constant_ranges: &[],
                });

        let pipeline = gpu
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("fractal pipeline"),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs_main"),
                    buffers: &[],
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fs_main"),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: gpu.surface_format,
                        blend: Some(wgpu::BlendState::REPLACE),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: Default::default(),
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleList,
                    ..Default::default()
                },
                depth_stencil: None,
                multisample: Default::default(),
                multiview: None,
                cache: None,
            });

        let uniform_buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("uniforms"),
            size: std::mem::size_of::<Uniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Palette texture (256x1 RGBA)
        let palette_data = Self::bake_inferno_palette();
        let palette_texture = gpu.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("palette"),
            size: wgpu::Extent3d {
                width: 256,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D1,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        gpu.queue.write_texture(
            palette_texture.as_image_copy(),
            &palette_data,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(256 * 4),
                rows_per_image: None,
            },
            wgpu::Extent3d {
                width: 256,
                height: 1,
                depth_or_array_layers: 1,
            },
        );

        let palette_view = palette_texture.create_view(&Default::default());
        let palette_sampler = gpu.device.create_sampler(&wgpu::SamplerDescriptor {
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            address_mode_u: wgpu::AddressMode::Repeat,
            ..Default::default()
        });

        // Iteration texture (window-sized, Rg32Float: r=iter, g=mag2)
        let (iter_texture, iter_view) = Self::create_iter_texture(gpu, gpu.width, gpu.height);

        let bind_group = Self::create_bind_group(
            &gpu.device,
            &bind_group_layout,
            &uniform_buffer,
            &palette_view,
            &palette_sampler,
            &iter_view,
        );

        Self {
            pipeline,
            bind_group_layout,
            uniform_buffer,
            palette_texture,
            palette_view,
            palette_sampler,
            iter_texture,
            iter_view,
            bind_group,
            iter_width: gpu.width,
            iter_height: gpu.height,
        }
    }

    fn create_iter_texture(gpu: &GpuContext, width: u32, height: u32) -> (wgpu::Texture, wgpu::TextureView) {
        let w = width.max(1);
        let h = height.max(1);
        let texture = gpu.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("iteration data"),
            size: wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rg32Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let view = texture.create_view(&Default::default());
        (texture, view)
    }

    fn create_bind_group(
        device: &wgpu::Device,
        layout: &wgpu::BindGroupLayout,
        uniform_buffer: &wgpu::Buffer,
        palette_view: &wgpu::TextureView,
        palette_sampler: &wgpu::Sampler,
        iter_view: &wgpu::TextureView,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("fractal bg"),
            layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(palette_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(palette_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(iter_view),
                },
            ],
        })
    }

    /// Resize the iteration texture when the window changes size.
    pub fn resize(&mut self, gpu: &GpuContext, width: u32, height: u32) {
        if width == 0 || height == 0 {
            return;
        }
        if width == self.iter_width && height == self.iter_height {
            return;
        }
        let (tex, view) = Self::create_iter_texture(gpu, width, height);
        self.iter_texture = tex;
        self.iter_view = view;
        self.iter_width = width;
        self.iter_height = height;
        self.bind_group = Self::create_bind_group(
            &gpu.device,
            &self.bind_group_layout,
            &self.uniform_buffer,
            &self.palette_view,
            &self.palette_sampler,
            &self.iter_view,
        );
    }

    /// Upload CPU-computed iteration data to the GPU texture.
    pub fn upload_iterations(&self, gpu: &GpuContext, result: &IterationResult) {
        gpu.queue.write_texture(
            self.iter_texture.as_image_copy(),
            bytemuck::cast_slice(&result.data),
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(result.width * 2 * 4), // 2 channels × 4 bytes
                rows_per_image: None,
            },
            wgpu::Extent3d {
                width: result.width,
                height: result.height,
                depth_or_array_layers: 1,
            },
        );
    }

    pub fn render(
        &self,
        gpu: &GpuContext,
        encoder: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        state: &FractalState,
    ) {
        let cx = state.center_re.to_f64();
        let cy = state.center_im.to_f64();
        let cx_hi = cx as f32;
        let cx_lo = (cx - cx_hi as f64) as f32;
        let cy_hi = cy as f32;
        let cy_lo = (cy - cy_hi as f64) as f32;

        let uniforms = Uniforms {
            resolution: [gpu.width as f32, gpu.height as f32],
            center_hi: [cx_hi, cy_hi],
            center_lo: [cx_lo, cy_lo],
            zoom: state.zoom as f32,
            max_iter: state.effective_max_iter(),
            color_shift: state.color_shift,
            mode: 1,
            _pad: [0.0; 2],
        };

        gpu.queue
            .write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(&uniforms));

        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("fractal"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
            })],
            ..Default::default()
        });

        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.bind_group, &[]);
        pass.draw(0..3, 0..1);
    }

    fn bake_inferno_palette() -> Vec<u8> {
        let stops: &[(f32, u8, u8, u8)] = &[
            (0.0, 0, 0, 4),
            (0.13, 20, 5, 50),
            (0.25, 60, 10, 100),
            (0.38, 120, 20, 100),
            (0.5, 180, 50, 50),
            (0.63, 220, 90, 10),
            (0.75, 250, 160, 20),
            (0.88, 255, 220, 80),
            (1.0, 252, 255, 164),
        ];

        let mut data = vec![0u8; 256 * 4];
        for i in 0..256 {
            let t = i as f32 / 255.0;
            let (mut lo, mut hi) = (stops[0], stops[stops.len() - 1]);
            for s in 0..stops.len() - 1 {
                if t >= stops[s].0 && t <= stops[s + 1].0 {
                    lo = stops[s];
                    hi = stops[s + 1];
                    break;
                }
            }
            let range = hi.0 - lo.0;
            let frac = if range > 0.0 {
                (t - lo.0) / range
            } else {
                0.0
            };
            data[i * 4] = (lo.1 as f32 + (hi.1 as f32 - lo.1 as f32) * frac) as u8;
            data[i * 4 + 1] = (lo.2 as f32 + (hi.2 as f32 - lo.2 as f32) * frac) as u8;
            data[i * 4 + 2] = (lo.3 as f32 + (hi.3 as f32 - lo.3 as f32) * frac) as u8;
            data[i * 4 + 3] = 255;
        }
        data
    }
}
