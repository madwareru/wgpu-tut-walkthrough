use winit::{
    event::*,
    event_loop::{EventLoop, ControlFlow},
    window::{WindowBuilder}
};
use winit::dpi::{PhysicalSize, LogicalSize};

fn main() {
    let event_loop = EventLoop::new();
    let fixed_size = winit::dpi::Size::Logical(LogicalSize{width: 1280.0, height: 800.0});
    let window = WindowBuilder::new()
        .with_min_inner_size(fixed_size)
        .with_max_inner_size(fixed_size)
        .build(&event_loop)
        .unwrap();

    event_loop.run(move |event, _, control_flow| {
        *control_flow = match event {
            Event::WindowEvent {
                ref event,
                window_id
            } if window_id == window.id() => {
                match event {
                    WindowEvent::CloseRequested => ControlFlow::Exit,
                    WindowEvent::KeyboardInput {
                        input,
                        ..
                    } => match input {
                        KeyboardInput {
                            state: ElementState::Pressed,
                            virtual_keycode: Some(VirtualKeyCode::Escape),
                            ..
                        } => ControlFlow::Exit,
                        _ => ControlFlow::Wait,
                    },
                    _ => ControlFlow::Wait
                }
            },
            _ => ControlFlow::Wait
        }
    });
}
