use winit::{
    event::*,
    event_loop::{EventLoop, ControlFlow},
    window::{WindowBuilder},
    dpi::{LogicalSize}
};

struct UserData {
    clear_color: wgpu::Color
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
        let user_data = UserData {clear_color: wgpu::Color{r:0.1, g: 0.2, b: 0.3, a: 1.0}};
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
            device_id, position, ..
        } = event {
            self.user_data.clear_color = wgpu::Color{
                r: position.x as f64 / self.size.width as f64,
                g: position.y as f64 / self.size.height as f64,
                ..self.user_data.clear_color
            };
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
            let _render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
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
