use std::sync::Arc;
use std::time::Instant;
use rug::{Assign, Float};
use winit::application::ApplicationHandler;
use winit::dpi::PhysicalSize;
use winit::event::{ElementState, MouseButton, MouseScrollDelta, WindowEvent};
use winit::event_loop::ActiveEventLoop;
use winit::keyboard::{Key, NamedKey};
use winit::window::{Window, WindowId};

use crate::gpu::GpuContext;
use crate::perturbation;
use crate::reference::ReferenceOrbit;
use crate::render::Renderer;

const DEFAULT_PRECISION: u32 = 128;

pub struct FractalState {
    pub center_re: Float,
    pub center_im: Float,
    pub zoom: f64,
    pub max_iter: u32,
    pub color_shift: f32,
    pub color_cycling: bool,
    pub color_speed: f32,
    pub needs_render: bool,
    pub needs_recompute: bool,
    // Auto-zoom with boundary seeking
    pub auto_zoom: bool,
    pub zoom_rate: f64,
    pub frame_time_ms: f64,
    pub boundary_lost: u32, // consecutive frames without boundary
}

impl FractalState {
    fn new() -> Self {
        Self {
            center_re: Float::with_val(DEFAULT_PRECISION, -0.5),
            center_im: Float::with_val(DEFAULT_PRECISION, 0.0),
            zoom: 0.5,
            max_iter: 300,
            color_shift: 0.0,
            color_cycling: false,
            color_speed: 0.3,
            needs_render: true,
            needs_recompute: true,
            auto_zoom: false,
            zoom_rate: 1.02,
            frame_time_ms: 0.0,
            boundary_lost: 0,
        }
    }

    fn update_precision(&mut self) {
        let digits = (self.zoom.max(1.0).log10() + 20.0) as u32;
        let prec = ((digits as f64 * 3.322) as u32).max(DEFAULT_PRECISION).min(16000);
        if self.center_re.prec() < prec {
            self.center_re.set_prec(prec);
            self.center_im.set_prec(prec);
        }
    }

    /// Auto-scale iterations with zoom depth
    pub fn effective_max_iter(&self) -> u32 {
        let base = 200u32;
        let zoom_bonus = (self.zoom.max(1.0).log10() * 80.0) as u32;
        let auto = base + zoom_bonus;
        // Use max of auto-computed and user setting, capped at 50000
        auto.max(self.max_iter).min(50000)
    }

}

struct Initialized {
    window: Arc<Window>,
    gpu: GpuContext,
    renderer: Renderer,
    egui_ctx: egui::Context,
    egui_state: egui_winit::State,
    egui_renderer: egui_wgpu::Renderer,
}

pub struct App {
    state: FractalState,
    init: Option<Initialized>,
    drag: Option<(f64, f64, Float, Float)>,
    last_cursor: (f64, f64),
}

impl App {
    pub fn new() -> Self {
        Self {
            state: FractalState::new(),
            init: None,
            drag: None,
            last_cursor: (0.0, 0.0),
        }
    }

    fn render(&mut self) {
        let Some(init) = &mut self.init else { return };
        let state = &mut self.state;
        let size = init.window.inner_size();
        if size.width == 0 || size.height == 0 {
            return;
        }

        let render_w = (size.width / 2).max(1);
        let render_h = (size.height / 2).max(1);
        init.renderer.resize(&init.gpu, render_w, render_h);

        if state.needs_recompute {
            let frame_start = Instant::now();
            let effective_iter = state.effective_max_iter();

            state.update_precision();
            let orbit = ReferenceOrbit::compute(
                &state.center_re,
                &state.center_im,
                effective_iter,
                state.zoom,
            );
            // Compute at half resolution for speed (~4x fewer pixels)
            let render_w = (size.width / 2).max(1);
            let render_h = (size.height / 2).max(1);
            let result = perturbation::compute(
                &orbit,
                render_w,
                render_h,
                state.zoom,
                effective_iter,
            );

            // Boundary seeking: steer center toward highest-variance region
            if state.auto_zoom {
                if let Some((ndc_x, ndc_y)) =
                    perturbation::find_boundary_target(&result, effective_iter)
                {
                    state.boundary_lost = 0;
                    let aspect = size.width as f64 / size.height as f64;
                    let scale = 2.0 / state.zoom;
                    let delta_re = ndc_x * aspect * scale;
                    let delta_im = ndc_y * scale;

                    let off_center = (ndc_x * ndc_x + ndc_y * ndc_y).sqrt();
                    let blend = (off_center * 0.25).clamp(0.02, 0.15);
                    state.center_re += delta_re * blend;
                    state.center_im += delta_im * blend;
                } else {
                    state.boundary_lost += 1;
                    if state.boundary_lost > 30 {
                        state.auto_zoom = false;
                        state.boundary_lost = 0;
                        log::info!("Auto-explore stopped: boundary lost");
                    }
                }
            }

            init.renderer.upload_iterations(&init.gpu, &result);
            state.frame_time_ms = frame_start.elapsed().as_secs_f64() * 1000.0;
            state.needs_recompute = false;
        }

        // Egui
        let raw_input = init.egui_state.take_egui_input(&init.window);
        let egui_output = init.egui_ctx.run(raw_input, |ctx| {
            Self::draw_ui(state, ctx);
        });
        init.egui_state
            .handle_platform_output(&init.window, egui_output.platform_output);
        let egui_primitives = init
            .egui_ctx
            .tessellate(egui_output.shapes, egui_output.pixels_per_point);
        let egui_screen = egui_wgpu::ScreenDescriptor {
            size_in_pixels: [size.width, size.height],
            pixels_per_point: init.window.scale_factor() as f32,
        };
        for (id, delta) in &egui_output.textures_delta.set {
            init.egui_renderer
                .update_texture(&init.gpu.device, &init.gpu.queue, *id, delta);
        }

        // GPU rendering
        let output = match init.gpu.surface.get_current_texture() {
            Ok(t) => t,
            Err(_) => {
                init.gpu.resize(size.width, size.height);
                return;
            }
        };
        let view = output.texture.create_view(&Default::default());
        let mut encoder = init
            .gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        init.renderer
            .render(&init.gpu, &mut encoder, &view, state);

        init.egui_renderer.update_buffers(
            &init.gpu.device,
            &init.gpu.queue,
            &mut encoder,
            &egui_primitives,
            &egui_screen,
        );
        {
            let mut pass = encoder
                .begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("egui"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Load,
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    ..Default::default()
                })
                .forget_lifetime();
            init.egui_renderer
                .render(&mut pass, &egui_primitives, &egui_screen);
        }

        init.gpu.queue.submit(std::iter::once(encoder.finish()));
        output.present();

        for id in &egui_output.textures_delta.free {
            init.egui_renderer.free_texture(id);
        }
    }

    fn draw_ui(state: &mut FractalState, ctx: &egui::Context) {
        let cx = state.center_re.to_f64();
        let cy = state.center_im.to_f64();

        egui::SidePanel::right("controls")
            .default_width(220.0)
            .show(ctx, |ui| {
                ui.heading("Deep Fractal");
                ui.separator();

                // Status
                let zoom_exp = state.zoom.log10();
                ui.label(format!("Zoom: 10^{:.1}", zoom_exp));
                ui.label(format!("Iter: {}", state.effective_max_iter()));
                ui.label(format!("Frame: {:.0} ms", state.frame_time_ms));
                ui.label(format!("({:.6e}, {:.6e})", cx, cy));

                ui.colored_label(
                    egui::Color32::from_rgb(100, 200, 100),
                    "Perturbation",
                );

                ui.separator();

                // Auto-zoom controls
                ui.label("Auto Explore");
                if ui
                    .add(
                        egui::Slider::new(&mut state.zoom_rate, 1.005..=1.08)
                            .text("Speed")
                            .fixed_decimals(3),
                    )
                    .changed()
                {}

                let btn_label = if state.auto_zoom {
                    "Stop Exploring"
                } else {
                    "Start Exploring"
                };
                if ui.button(btn_label).clicked() {
                    state.auto_zoom = !state.auto_zoom;
                    state.boundary_lost = 0;
                    if state.auto_zoom {
                        state.needs_render = true;
                        state.needs_recompute = true;
                    }
                }

                ui.separator();

                // Manual controls
                ui.label("Iterations (manual)");
                let mut iter = state.max_iter as i32;
                if ui
                    .add(egui::Slider::new(&mut iter, 100..=10000).logarithmic(true))
                    .changed()
                {
                    state.max_iter = iter as u32;
                    state.needs_render = true;
                    state.needs_recompute = true;
                }

                ui.separator();
                ui.label("Color");
                if ui
                    .add(egui::Slider::new(&mut state.color_speed, 0.0..=2.0).text("Speed"))
                    .changed()
                {}
                if ui.checkbox(&mut state.color_cycling, "Cycle colors").changed() {}

                ui.separator();

                // Preset starting points
                ui.label("Jump to:");
                let targets = [
                    ("Seahorse", -0.7463, 0.1102),
                    ("Spiral", -0.3842, 0.65975),
                    ("Elephant", 0.2821, 0.01),
                    ("Mini-brot", -1.7490863, 0.0000002),
                    ("Scepter", -0.1011, 0.9563),
                ];

                ui.horizontal_wrapped(|ui| {
                    for (name, x, y) in &targets {
                        if ui.button(*name).clicked() {
                            let prec = state.center_re.prec().max(DEFAULT_PRECISION);
                            state.center_re = Float::with_val(prec, *x);
                            state.center_im = Float::with_val(prec, *y);
                            state.zoom = 5.0; // Start close enough to see detail
                            state.auto_zoom = true;
                            state.needs_render = true;
                            state.needs_recompute = true;
                        }
                    }
                });

                ui.separator();
                if ui.button("Reset").clicked() {
                    state.center_re.assign(-0.5);
                    state.center_im.assign(0.0);
                    state.zoom = 0.5;
                    state.auto_zoom = false;
                    state.needs_render = true;
                    state.needs_recompute = true;
                }
            });
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.init.is_some() {
            return;
        }

        let attrs = Window::default_attributes()
            .with_title("Deep Fractal")
            .with_inner_size(PhysicalSize::new(1280, 800));
        let window = Arc::new(event_loop.create_window(attrs).expect("create window"));
        let gpu = pollster::block_on(GpuContext::new(Arc::clone(&window)));
        let renderer = Renderer::new(&gpu);
        let egui_ctx = egui::Context::default();
        let egui_state =
            egui_winit::State::new(egui_ctx.clone(), egui_ctx.viewport_id(), &window, None, None, None);
        let egui_renderer =
            egui_wgpu::Renderer::new(&gpu.device, gpu.surface_format, None, 1, false);

        self.init = Some(Initialized {
            window,
            gpu,
            renderer,
            egui_ctx,
            egui_state,
            egui_renderer,
        });
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        // Always clear drag on left mouse release, even if egui would consume it
        if let WindowEvent::MouseInput {
            state: ElementState::Released,
            button: MouseButton::Left,
            ..
        } = event
        {
            self.drag = None;
        }

        // Track cursor position for drag start
        if let WindowEvent::CursorMoved { position, .. } = event {
            self.last_cursor = (position.x, position.y);
        }

        // Let egui handle events
        let mut egui_consumed = false;
        if let Some(init) = &mut self.init {
            let response = init.egui_state.on_window_event(&init.window, &event);
            if response.consumed {
                egui_consumed = true;
                self.state.needs_render = true;
                self.state.needs_recompute = true;
            }
        }

        if !egui_consumed {
            match event {
                WindowEvent::CloseRequested => event_loop.exit(),
                WindowEvent::Resized(size) => {
                    if let Some(init) = &mut self.init {
                        init.gpu.resize(size.width, size.height);
                        self.state.needs_render = true;
                        self.state.needs_recompute = true;
                    }
                }
                WindowEvent::MouseInput {
                    state: ElementState::Pressed,
                    button: MouseButton::Left,
                    ..
                } => {
                    self.drag = Some((
                        self.last_cursor.0,
                        self.last_cursor.1,
                        self.state.center_re.clone(),
                        self.state.center_im.clone(),
                    ));
                }
                WindowEvent::CursorMoved { position, .. } => {
                    if let Some(init) = &self.init {
                        let size = init.window.inner_size();
                        if let Some(ref drag) = self.drag {
                            let dx = position.x - drag.0;
                            let dy = position.y - drag.1;
                            let scale = 2.0 / (self.state.zoom * size.height as f64);
                            self.state.center_re.assign(&drag.2 - dx * scale);
                            self.state.center_im.assign(&drag.3 + dy * scale);
                            self.state.needs_render = true;
                            self.state.needs_recompute = true;
                        }
                    }
                }
                WindowEvent::MouseWheel { delta, .. } => {
                    let dy = match delta {
                        MouseScrollDelta::LineDelta(_, y) => y as f64 * 100.0,
                        MouseScrollDelta::PixelDelta(p) => p.y,
                    };
                    let factor = (1.001_f64).powf(dy);
                    self.state.zoom *= factor;
                    self.state.needs_render = true;
                    self.state.needs_recompute = true;
                }
                WindowEvent::KeyboardInput { ref event, .. } => {
                    if event.state == ElementState::Pressed {
                        match event.logical_key {
                            Key::Character(ref c) if c.as_str() == "r" => {
                                self.state.center_re.assign(-0.5);
                                self.state.center_im.assign(0.0);
                                self.state.zoom = 0.5;
                                self.state.auto_zoom = false;
                                self.state.needs_render = true;
                                self.state.needs_recompute = true;
                            }
                            Key::Named(NamedKey::Space) => {
                                self.state.auto_zoom = !self.state.auto_zoom;
                                self.state.needs_render = true;
                                self.state.needs_recompute = true;
                            }
                            Key::Named(NamedKey::Escape) => {
                                if self.state.auto_zoom {
                                    self.state.auto_zoom = false;
                                } else {
                                    event_loop.exit();
                                }
                            }
                            _ => {}
                        }
                    }
                }
                _ => {}
            }
        }

        if matches!(event, WindowEvent::RedrawRequested) {
            // Auto-zoom: only advance zoom when boundary is tracked
            if self.state.auto_zoom && self.state.boundary_lost == 0 {
                self.state.zoom *= self.state.zoom_rate;
                self.state.needs_render = true;
                self.state.needs_recompute = true;
            } else if self.state.auto_zoom {
                // Boundary lost — still re-render to let seeker try again
                self.state.needs_render = true;
                self.state.needs_recompute = true;
            }

            if self.state.color_cycling {
                self.state.color_shift += self.state.color_speed / 60.0;
                self.state.needs_render = true;
            }

            if self.state.needs_render {
                self.render();
                self.state.needs_render = false;
            }
        }

        if let Some(init) = &self.init {
            init.window.request_redraw();
        }
    }
}
