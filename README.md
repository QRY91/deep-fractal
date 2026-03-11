# Deep Fractal

Real-time Mandelbrot explorer using **perturbation theory** for arbitrary-depth zoom. Built in Rust with wgpu.

Reached **10^63 zoom** and counting — far beyond what GPU float64 can do alone.

## How it works

Traditional Mandelbrot renderers hit a precision wall around 10^15 zoom (float64 limit). This app breaks through using perturbation theory:

1. **Reference orbit** — One orbit computed at the screen center using arbitrary-precision arithmetic (GMP via `rug`). Precision auto-scales with zoom depth.
2. **Delta iteration** — Each pixel iterates only the tiny difference from the reference: `δ_{n+1} = 2·Z_n·δ_n + δ_n² + δ₀`. This stays in float64 range even at extreme zoom.
3. **Parallel CPU** — Pixel perturbation runs across all cores via `rayon`.
4. **GPU coloring** — Iteration data uploaded to a texture, smooth-colored with a palette shader.

## Features

- **Auto-explore** — Boundary-seeking algorithm steers the camera toward complex fractal detail (filaments, spirals, mini-brots) and zooms continuously
- **Arbitrary depth** — Precision scales automatically: 128 bits at shallow zoom, up to 16000 bits at extreme depth
- **Preset locations** — Jump to Seahorse Valley, Elephant Valley, Spiral, Mini-brot, Scepter
- **Color cycling** — Animated palette shifting
- **Interactive** — Click-drag to pan, scroll to zoom, Space to toggle auto-explore, R to reset

## Building

Requires GMP development libraries:

```bash
# Fedora
sudo dnf install gmp-devel

# Ubuntu/Debian
sudo apt install libgmp-dev

# macOS
brew install gmp
```

Then:

```bash
cargo run --release
```

## Controls

| Input | Action |
|-------|--------|
| Scroll | Zoom in/out |
| Click + drag | Pan |
| Space | Toggle auto-explore |
| R | Reset to overview |
| Escape | Stop exploring / quit |

## Tech stack

- **Rust** with `rug` (GMP) for arbitrary precision
- **wgpu** (Vulkan/GL) for GPU rendering
- **rayon** for parallel CPU iteration
- **egui** for the control panel
- **winit** for windowing

## Roadmap

- [ ] Series approximation — skip 90-97% of per-pixel iterations for 5-10x speedup
- [ ] Glitch detection + rebasing for artifact-free rendering
- [ ] Native float64 GPU compute shader for NVIDIA cards
- [ ] Cinematic zoom recording
