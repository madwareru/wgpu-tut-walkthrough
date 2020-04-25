use winit::{
    event::*,
    event_loop::{EventLoop, ControlFlow},
    window::{WindowBuilder},
    dpi::{LogicalSize}
};

#[repr(C)]
#[derive(Copy, Clone, Debug)]
struct VertexData {
    position: [f32; 3],
    color: [f32; 3]
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
                    format: wgpu::VertexFormat::Float3
                }
            ]
        }
    }
}

const TRIANGLE_VERTICES: &[VertexData] = &[
    VertexData{ position: [ 0.0, -0.5, 0.0], color: [1.0, 0.0, 0.0]},
    VertexData{ position: [-0.5,  0.5, 0.0], color: [0.0, 1.0, 0.0]},
    VertexData{ position: [ 0.5,  0.5, 0.0], color: [0.0, 0.0, 1.0]},
];

const TRIANGLE_INDICES: &[u16] = &[
    0, 1, 2,
];

const QUAD_VERTICES: &[VertexData] = &[
    VertexData{ position: [-0.5, -0.5, 0.0], color: [1.0, 0.0, 0.0]},
    VertexData{ position: [-0.5,  0.5, 0.0], color: [0.0, 1.0, 0.0]},
    VertexData{ position: [ 0.5,  0.5, 0.0], color: [0.0, 0.0, 1.0]},
    VertexData{ position: [ 0.5, -0.5, 0.0], color: [1.0, 1.0, 0.0]},
];

const QUAD_INDICES: &[u16] = &[
    0, 1, 2,
    2, 3, 0,
];

struct Mesh {
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    num_indices: u32
}

struct UserData {
    clear_color: wgpu::Color,
    triangle_mesh: Mesh,
    quad_mesh: Mesh,
    triangle_render_pipeline: wgpu::RenderPipeline,
    colored_render_pipeline: wgpu::RenderPipeline,
    pipeline_switched: bool
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
    frag_src: &'static str
) -> wgpu::RenderPipeline {
    let vs_spirv = glsl_to_spirv::compile(vert_src, glsl_to_spirv::ShaderType::Vertex).unwrap();
    let fs_spirv = glsl_to_spirv::compile(frag_src, glsl_to_spirv::ShaderType::Fragment).unwrap();

    let vs_data = wgpu::read_spirv(vs_spirv).unwrap();
    let fs_data = wgpu::read_spirv(fs_spirv).unwrap();

    let vs_module = device.create_shader_module(&vs_data);
    let fs_module = device.create_shader_module(&fs_data);

    let render_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor{
        bind_group_layouts: &[]
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
                color_blend: wgpu::BlendDescriptor::REPLACE,
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

impl MainState {
    fn new(window: &winit::window::Window) -> Self {
        let size = window.inner_size();
        let surface = wgpu::Surface::create(window);
        let adapter = wgpu::Adapter::request(&wgpu::RequestAdapterOptions{
            ..Default::default()
        }).unwrap();
        let (device, queue) = adapter.request_device(&wgpu::DeviceDescriptor{
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
        let vs_src2 = include_str!("shader_vert2.glsl");
        let fs_src2 = include_str!("shader_frag2.glsl");

        let render_pipeline = create_render_pipeline_from_shaders(&device, vs_src, fs_src);
        let colored_pipeline = create_render_pipeline_from_shaders(&device, vs_src2, fs_src2);

        let triangle_mesh = create_mesh(&device, TRIANGLE_VERTICES, TRIANGLE_INDICES);
        let quad_mesh = create_mesh(&device, QUAD_VERTICES, QUAD_INDICES);

        let user_data = UserData {
            clear_color: wgpu::Color{r:0.1, g: 0.2, b: 0.3, a: 1.0},
            triangle_render_pipeline: render_pipeline,
            colored_render_pipeline: colored_pipeline,
            pipeline_switched: false,
            triangle_mesh,
            quad_mesh
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
            return true;
        } else if let WindowEvent::KeyboardInput {
            input: KeyboardInput{
                state: ElementState::Released,
                virtual_keycode: Some(VirtualKeyCode::Space),
                ..
            },
            ..
        } = event {
            self.user_data.pipeline_switched = !self.user_data.pipeline_switched;
            return true;
        }
        false
    }

    fn update(&mut self) {

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

            render_pass.set_pipeline(if self.user_data.pipeline_switched {
                &self.user_data.colored_render_pipeline
            } else {
                &self.user_data.triangle_render_pipeline
            });

            let mesh = if self.user_data.pipeline_switched {
                &self.user_data.quad_mesh
            } else {
                &self.user_data.triangle_mesh
            };

            render_pass.set_vertex_buffers(0, &[(&mesh.vertex_buffer, 0)]);
            render_pass.set_index_buffer(&mesh.index_buffer, 0);
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
