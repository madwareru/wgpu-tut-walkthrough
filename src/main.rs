use winit::{
    event::*,
    event_loop::{EventLoop, ControlFlow},
    window::{WindowBuilder},
    dpi::{LogicalSize}
};
use wgpu::{BufferAsyncMapping, BufferMapAsyncResult};
use image::GenericImageView;

#[repr(C)]
#[derive(Copy, Clone, Debug)]
struct VertexData {
    position: [f32; 3],
    uv: [f32; 2]
}

impl VertexData {
    fn desc<'a>() -> wgpu::VertexBufferDescriptor<'a> {
        use std::mem;
        let stride = mem::size_of::<VertexData>() as wgpu::BufferAddress;
        wgpu::VertexBufferDescriptor{
            stride,
            step_mode: wgpu::InputStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttributeDescriptor {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float3
                },
                wgpu::VertexAttributeDescriptor {
                    offset: mem::size_of::<[f32; 3]>() as wgpu::BufferAddress,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float2
                }
            ]
        }
    }
}

const TRIANGLE_VERTICES: &[VertexData] = &[
    VertexData{ position: [ 0.0,  0.5, 0.0], uv: [0.5, 0.0]},
    VertexData{ position: [ 0.5, -0.5, 0.0], uv: [1.0, 1.0]},
    VertexData{ position: [-0.5, -0.5, 0.0], uv: [0.0, 1.0]},
];

const TRIANGLE_INDICES: &[u16] = &[
    0, 1, 2,
];

const QUAD_VERTICES: &[VertexData] = &[
    VertexData{ position: [-0.5,  0.5, 0.0], uv: [0.0, 0.0]},
    VertexData{ position: [-0.5, -0.5, 0.0], uv: [0.0, 1.0]},
    VertexData{ position: [ 0.5, -0.5, 0.0], uv: [1.0, 1.0]},
    VertexData{ position: [ 0.5,  0.5, 0.0], uv: [1.0, 0.0]},
];

const QUAD_INDICES: &[u16] = &[
    0, 2, 1,
    2, 0, 3,
];

#[cfg_attr(rustfmt, rustfmt_skip)]
const OPENGL_TO_WGPU_MATRIX: cgmath::Matrix4<f32> = cgmath::Matrix4::new(
    1.0,  0.0, 0.0, 0.0,
    0.0, -1.0, 0.0, 0.0,
    0.0,  0.0, 0.5, 0.0,
    0.0,  0.0, 0.5, 1.0
);

struct Camera{
    eye: cgmath::Point3<f32>,
    target: cgmath::Point3<f32>,
    up: cgmath::Vector3<f32>,
    aspect: f32,
    fovy: f32,
    znear: f32,
    zfar: f32,
}

impl Camera {
    fn to_view_proj(&self) -> cgmath::Matrix4<f32> {
        let view = cgmath::Matrix4::look_at(self.eye, self.target, self.up);
        let proj = cgmath::perspective(cgmath::Deg(self.fovy), self.aspect, self.znear, self.zfar);
        OPENGL_TO_WGPU_MATRIX * proj * view
    }
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
struct Uniforms {
    view_proj: cgmath::Matrix4<f32>
}

impl Uniforms {
    fn new() -> Self {
        use cgmath::SquareMatrix;
        Self {
            view_proj: cgmath::Matrix4::identity()
        }
    }
    fn update_view_proj(&mut self, camera: &Camera) {
        self.view_proj = camera.to_view_proj();
    }
}

struct Mesh {
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    num_indices: u32
}

struct TextureHandle {
    texture: wgpu::Texture,
    view: wgpu::TextureView,
    sampler: wgpu::Sampler
}

struct Texture {
    handle: TextureHandle,
    bind_group: wgpu::BindGroup
}

struct UserData {
    camera: Camera,
    uniforms: Uniforms,
    uniform_buffer: wgpu::Buffer,
    uniform_bind_group: wgpu::BindGroup,
    clear_color: wgpu::Color,
    triangle_mesh: Mesh,
    quad_mesh: Mesh,
    render_pipeline: wgpu::RenderPipeline,
    mode_switched: bool,
    necromancer_texture: Texture,
    mage_texture: Texture,
}

struct MainState {
    surface: wgpu::Surface,
    adapter: wgpu::Adapter,
    device: wgpu::Device,
    queue: wgpu::Queue,
    swap_chain_desc: wgpu::SwapChainDescriptor,
    swap_chain: wgpu::SwapChain,

    size: winit::dpi::PhysicalSize<u32>,
    user_data: UserData
}

fn create_render_pipeline_from_shaders(
    device: &wgpu::Device,
    vert_src: &'static str,
    frag_src: &'static str,
    texture_bind_group_layout: &wgpu::BindGroupLayout,
    uniforms_bind_group_layout: &wgpu::BindGroupLayout
) -> wgpu::RenderPipeline {
    let vs_spirv = glsl_to_spirv::compile(vert_src, glsl_to_spirv::ShaderType::Vertex).unwrap();
    let fs_spirv = glsl_to_spirv::compile(frag_src, glsl_to_spirv::ShaderType::Fragment).unwrap();

    let vs_data = wgpu::read_spirv(vs_spirv).unwrap();
    let fs_data = wgpu::read_spirv(fs_spirv).unwrap();

    let vs_module = device.create_shader_module(&vs_data);
    let fs_module = device.create_shader_module(&fs_data);

    let render_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor{
        bind_group_layouts: &[
            &texture_bind_group_layout,
            &uniforms_bind_group_layout
        ]
    });

    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor{
        layout: &render_pipeline_layout,
        vertex_stage: wgpu::ProgrammableStageDescriptor{
            module: &vs_module,
            entry_point: "main"
        },
        fragment_stage: Some(wgpu::ProgrammableStageDescriptor{
            module: &fs_module,
            entry_point: "main"
        }),
        rasterization_state: Some(
            wgpu::RasterizationStateDescriptor {
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: wgpu::CullMode::Back,
                depth_bias: 0,
                depth_bias_slope_scale: 0.0,
                depth_bias_clamp: 0.0
            }
        ),
        color_states: &[
            wgpu::ColorStateDescriptor {
                format: wgpu::TextureFormat::Bgra8UnormSrgb,
                color_blend: wgpu::BlendDescriptor{
                    src_factor: wgpu::BlendFactor::SrcAlpha,
                    dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                    operation: wgpu::BlendOperation::Add
                },
                alpha_blend: wgpu::BlendDescriptor::REPLACE,
                write_mask: wgpu::ColorWrite::ALL
            }
        ],
        primitive_topology: wgpu::PrimitiveTopology::TriangleList,
        depth_stencil_state: None,
        index_format: wgpu::IndexFormat::Uint16,
        vertex_buffers: &[
            VertexData::desc()
        ],
        sample_count: 1,
        sample_mask: !0,
        alpha_to_coverage_enabled: false
    })
}

fn create_mesh(device: &wgpu::Device, vertices: &[VertexData], indices: &[u16]) -> Mesh {
    let vertex_buffer = device
        .create_buffer_mapped(vertices.len(), wgpu::BufferUsage::VERTEX)
        .fill_from_slice(vertices);
    let index_buffer = device
        .create_buffer_mapped(indices.len(), wgpu::BufferUsage::INDEX)
        .fill_from_slice(indices);

    Mesh {
        vertex_buffer,
        index_buffer,
        num_indices: indices.len() as u32
    }
}

fn create_texture_bind_group(
    device: &wgpu::Device,
    texture_view: &wgpu::TextureView,
    sampler: &wgpu::Sampler,
    texture_bind_group_layout: &wgpu::BindGroupLayout
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor{
        layout: &texture_bind_group_layout,
        bindings: &[
            wgpu::Binding {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&texture_view)
            },
            wgpu::Binding {
                binding: 1,
                resource: wgpu::BindingResource::Sampler(&sampler)
            }
        ]
    })
}

fn create_texture(device: &wgpu::Device, bytes: &[u8]) -> (TextureHandle, wgpu::CommandBuffer) {
    let diffuse_image = image::load_from_memory(bytes).unwrap();
    let diffuse_rgba = diffuse_image.as_rgba8().unwrap();

    use image::GenericImageView;
    let (width, height) = diffuse_image.dimensions();

    let size = wgpu::Extent3d{
        width,
        height,
        depth: 1
    };

    let texture = device.create_texture(&wgpu::TextureDescriptor{
        size,
        array_layer_count: 1,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8UnormSrgb,
        usage: wgpu::TextureUsage::SAMPLED | wgpu::TextureUsage::COPY_DST
    });

    let diffuse_buffer = device
        .create_buffer_mapped(diffuse_rgba.len(), wgpu::BufferUsage::COPY_SRC)
        .fill_from_slice(&diffuse_rgba);

    let mut encoder = device.create_command_encoder(&Default::default());

    encoder.copy_buffer_to_texture(
        wgpu::BufferCopyView {
            buffer: &diffuse_buffer,
            offset: 0,
            row_pitch: 4 * width,
            image_height: height
        },
        wgpu::TextureCopyView {
            texture: &texture,
            mip_level: 0,
            array_layer: 0,
            origin: wgpu::Origin3d::ZERO
        },
        size
    );

    let command_buffer = encoder.finish();

    let view = texture.create_default_view();
    let sampler = device.create_sampler(&wgpu::SamplerDescriptor{
        address_mode_u: wgpu::AddressMode::ClampToEdge,
        address_mode_v: wgpu::AddressMode::ClampToEdge,
        address_mode_w: wgpu::AddressMode::ClampToEdge,
        mag_filter: wgpu::FilterMode::Nearest,
        min_filter: wgpu::FilterMode::Nearest,
        mipmap_filter: wgpu::FilterMode::Nearest,
        lod_min_clamp: -100.0,
        lod_max_clamp: 100.0,
        compare_function: wgpu::CompareFunction::Always
    });

    (
        TextureHandle { texture, view, sampler },
        command_buffer
    )
}

impl MainState {
    fn new(window: &winit::window::Window) -> Self {
        let size = window.inner_size();
        let surface = wgpu::Surface::create(window);
        let adapter = wgpu::Adapter::request(&wgpu::RequestAdapterOptions{
            ..Default::default()
        }).unwrap();
        let (device, mut queue) = adapter.request_device(&wgpu::DeviceDescriptor{
            extensions: wgpu::Extensions {
                anisotropic_filtering: false
            },
            limits: Default::default()
        });
        let swap_chain_desc = wgpu::SwapChainDescriptor {
            usage: wgpu::TextureUsage::OUTPUT_ATTACHMENT,
            format: wgpu::TextureFormat::Bgra8UnormSrgb,
            width: size.width,
            height: size.height,
            present_mode: wgpu::PresentMode::Vsync
        };
        let swap_chain = device.create_swap_chain(&surface, &swap_chain_desc);

        let vs_src = include_str!("shader_vert.glsl");
        let fs_src = include_str!("shader_frag.glsl");

        let necromancer_bytes = include_bytes!("necromancer.png");
        let mage_bytes = include_bytes!("mage.png");
        let (necromancer_texture_handle, necromancer_c_buffer) = create_texture(&device, necromancer_bytes);
        let (mage_texture_handle, mage_c_buffer) = create_texture(&device, mage_bytes);

        queue.submit(&[necromancer_c_buffer, mage_c_buffer]);

        let texture_bind_group_layout = device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor{
                bindings: &[
                    wgpu::BindGroupLayoutBinding {
                        binding: 0,
                        visibility: wgpu::ShaderStage::FRAGMENT,
                        ty: wgpu::BindingType::SampledTexture {
                            multisampled: false,
                            dimension: wgpu::TextureViewDimension::D2
                        }
                    },
                    wgpu::BindGroupLayoutBinding {
                        binding: 1,
                        visibility: wgpu::ShaderStage::FRAGMENT,
                        ty: wgpu::BindingType::Sampler
                    }
                ]
            });

        let necromancer_bind_group = create_texture_bind_group(
            &device,
            &necromancer_texture_handle.view,
            &necromancer_texture_handle.sampler,
            &texture_bind_group_layout
        );

        let mage_bind_group = create_texture_bind_group(
            &device,
            &mage_texture_handle.view,
            &mage_texture_handle.sampler,
            &texture_bind_group_layout
        );

        let necromancer_texture = Texture {
            handle: necromancer_texture_handle,
            bind_group: necromancer_bind_group
        };

        let mage_texture = Texture {
            handle: mage_texture_handle,
            bind_group: mage_bind_group
        };

        let camera = Camera {
            eye: (0.0, 0.0, -2.0).into(),
            target: (0.0, 0.0, 0.0).into(),
            up: cgmath::Vector3::unit_y(),
            aspect: swap_chain_desc.width as f32 / swap_chain_desc.height as f32,
            fovy: 45.0,
            znear: 0.1,
            zfar: 100.0
        };

        let mut uniforms = Uniforms::new();
        uniforms.update_view_proj(&camera);

        let uniform_buffer = device
            .create_buffer_mapped(1, wgpu::BufferUsage::UNIFORM | wgpu::BufferUsage::COPY_DST)
            .fill_from_slice(&[uniforms]);

        let uniform_bind_group_layout = device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                bindings: &[
                    wgpu::BindGroupLayoutBinding {
                        binding: 0,
                        visibility: wgpu::ShaderStage::VERTEX,
                        ty: wgpu::BindingType::UniformBuffer {
                            dynamic: false
                        }
                    }
                ]
            });

        let uniform_bind_group = device
            .create_bind_group(&wgpu::BindGroupDescriptor{
                layout: &uniform_bind_group_layout,
                bindings: &[
                    wgpu::Binding {
                        binding: 0,
                        resource: wgpu::BindingResource::Buffer {
                            buffer: &uniform_buffer,
                            range: 0..std::mem::size_of_val(&uniforms) as wgpu::BufferAddress
                        }
                    }
                ]
            });

        let render_pipeline = create_render_pipeline_from_shaders(
            &device,
            vs_src,
            fs_src,
            &texture_bind_group_layout,
            &uniform_bind_group_layout
        );

        let triangle_mesh = create_mesh(&device, TRIANGLE_VERTICES, TRIANGLE_INDICES);
        let quad_mesh = create_mesh(&device, QUAD_VERTICES, QUAD_INDICES);

        let user_data = UserData {
            camera,
            uniforms,
            uniform_buffer,
            uniform_bind_group,
            clear_color: wgpu::Color{r:0.1, g: 0.2, b: 0.3, a: 1.0},
            render_pipeline,
            mode_switched: false,
            triangle_mesh,
            quad_mesh,
            necromancer_texture,
            mage_texture
        };

        Self {
            surface,
            adapter,
            device,
            queue,
            swap_chain_desc,
            swap_chain,
            size,
            user_data
        }
    }

    fn input(&mut self, event: &WindowEvent) -> bool {
        if let WindowEvent::CursorMoved{
            position, ..
        } = event {
            self.user_data.clear_color = wgpu::Color{
                r: position.x as f64 / self.size.width as f64,
                g: position.y as f64 / self.size.height as f64,
                ..self.user_data.clear_color
            };
            self.user_data.camera.eye.x = self.user_data.clear_color.r as f32 - 0.5;
            self.user_data.camera.eye.y = self.user_data.clear_color.g as f32 - 0.5;
            return true;
        } else if let WindowEvent::KeyboardInput {
            input: KeyboardInput{
                state: ElementState::Released,
                virtual_keycode: Some(VirtualKeyCode::Space),
                ..
            },
            ..
        } = event {
            self.user_data.mode_switched = !self.user_data.mode_switched;
            return true;
        }
        false
    }

    fn update(&mut self) {
        self.user_data.uniforms.update_view_proj(&self.user_data.camera);

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor{
            todo: 0
        });

        let staging_buffer =self.device
            .create_buffer_mapped(1, wgpu::BufferUsage::COPY_SRC)
            .fill_from_slice(&[self.user_data.uniforms]);

        encoder.copy_buffer_to_buffer(
            &staging_buffer, 0,
            &self.user_data.uniform_buffer, 0,
            std::mem::size_of::<Uniforms>() as wgpu::BufferAddress
        );

        self.queue.submit(&[encoder.finish()]);
    }

    fn render(&mut self) {
        let frame = self.swap_chain.get_next_texture();
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            todo: 0
        });

        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                color_attachments: &[
                    wgpu::RenderPassColorAttachmentDescriptor {
                        attachment: &frame.view,
                        resolve_target: None,
                        load_op: wgpu::LoadOp::Clear,
                        store_op: wgpu::StoreOp::Store,
                        clear_color: self.user_data.clear_color
                    }
                ],
                depth_stencil_attachment: None
            });

            render_pass.set_pipeline(&self.user_data.render_pipeline);

            let mesh = if self.user_data.mode_switched {
                &self.user_data.quad_mesh
            } else {
                &self.user_data.triangle_mesh
            };

            let texture = if self.user_data.mode_switched {
                &self.user_data.mage_texture
            } else {
                &self.user_data.necromancer_texture
            };

            render_pass.set_vertex_buffers(0, &[(&mesh.vertex_buffer, 0)]);
            render_pass.set_index_buffer(&mesh.index_buffer, 0);
            render_pass.set_bind_group(0, &texture.bind_group, &[]);
            render_pass.set_bind_group(1, &self.user_data.uniform_bind_group, &[]);
            render_pass.draw_indexed(0..mesh.num_indices, 0, 0..1);
        }

        self.queue.submit(&[
            encoder.finish()
        ]);
    }
}

fn main() {
    let event_loop = EventLoop::new();
    let fixed_size = winit::dpi::Size::Logical(LogicalSize{width: 1280.0, height: 800.0});
    let window = WindowBuilder::new()
        .with_min_inner_size(fixed_size)
        .with_max_inner_size(fixed_size)
        .with_resizable(false)
        .with_title("wgpu-tut-walkthrough")
        .build(&event_loop)
        .unwrap();

    let mut main_state = MainState::new(&window);

    event_loop.run(move |event, _, control_flow| {
        *control_flow = {
            if let Event::WindowEvent { ref event, window_id} = event {
                if window_id == window.id() {
                    if main_state.input(event) {
                        ControlFlow::Wait
                    } else {
                        match event {
                            WindowEvent::CloseRequested |
                            WindowEvent::KeyboardInput {
                                input: KeyboardInput {
                                    state: ElementState::Pressed,
                                    virtual_keycode: Some(VirtualKeyCode::Escape),
                                    ..
                                },
                                ..
                            } => ControlFlow::Exit,
                            _ => ControlFlow::Wait
                        }
                    }
                } else {
                    ControlFlow::Wait
                }
            } else if let Event::MainEventsCleared = event {
                main_state.update();
                main_state.render();
                ControlFlow::Wait
            } else {
                ControlFlow::Wait
            }
        }
    });
}
