use ab_glyph::{FontRef, PxScale};
use async_stream::stream;
use bytes::Bytes;
use cgmath::{Deg, Matrix4, Point3, SquareMatrix, Vector3, perspective};
use futures_intrusive::channel::shared::oneshot_channel;
use hyper::service::{make_service_fn, service_fn};
use hyper::{Body, Request, Response, Server, header};
use image::codecs::jpeg::JpegEncoder;
use image::{DynamicImage, ImageBuffer, Rgba};
use imageproc::drawing::draw_text_mut;
use std::io::Cursor;
use std::{
    convert::Infallible,
    sync::{Arc, atomic::{AtomicU64, Ordering}},
    time::{Duration, Instant},
};
use tokio::sync::RwLock;
use tokio::sync::Notify;
use wgpu::{Backends, InstanceDescriptor, TextureFormat, util::DeviceExt};

#[tokio::main]
async fn main() {
        use std::collections::VecDeque;
    let latest = Arc::new(RwLock::new(Vec::new()));
    let frame_version = Arc::new(AtomicU64::new(0));
    let notify = Arc::new(Notify::new());
    let stream_clone = latest.clone();
    let version_clone = frame_version.clone();
    let notify_clone = notify.clone();

    tokio::spawn(async move {
        let instance = wgpu::Instance::new(InstanceDescriptor {
            backends: Backends::all(),
            dx12_shader_compiler: Default::default(),
            flags: wgpu::InstanceFlags::default(),
            gles_minor_version: wgpu::Gles3MinorVersion::Automatic,
        });

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .expect("No suitable GPU adapters found");

        println!("Using adapter: {:?}", adapter.get_info());

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor::default(), None)
            .await
            .expect("Failed to create device");

        let size = wgpu::Extent3d {
            width: 256,
            height: 256,
            depth_or_array_layers: 1,
        };

        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct Vertex {
            position: [f32; 3],
            color: [f32; 3],
        }

        impl Vertex {
            const ATTRIBS: [wgpu::VertexAttribute; 2] = wgpu::vertex_attr_array![
                0 => Float32x3,
                1 => Float32x3,
            ];

            fn desc<'a>() -> wgpu::VertexBufferLayout<'a> {
                wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<Vertex>() as wgpu::BufferAddress,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &Self::ATTRIBS,
                }
            }
        }

        let vertices = [
            Vertex {
                position: [-1.0, -1.0, 1.0],
                color: [1.0, 0.0, 0.0],
            },
            Vertex {
                position: [1.0, -1.0, 1.0],
                color: [0.0, 1.0, 0.0],
            },
            Vertex {
                position: [1.0, 1.0, 1.0],
                color: [0.0, 0.0, 1.0],
            },
            Vertex {
                position: [-1.0, 1.0, 1.0],
                color: [1.0, 1.0, 0.0],
            },
            Vertex {
                position: [-1.0, -1.0, -1.0],
                color: [1.0, 0.0, 1.0],
            },
            Vertex {
                position: [1.0, -1.0, -1.0],
                color: [0.0, 1.0, 1.0],
            },
            Vertex {
                position: [1.0, 1.0, -1.0],
                color: [1.0, 1.0, 1.0],
            },
            Vertex {
                position: [-1.0, 1.0, -1.0],
                color: [0.0, 0.0, 0.0],
            },
        ];

        let indices: &[u16] = &[
            0, 1, 2, 2, 3, 0, 1, 5, 6, 6, 2, 1, 5, 4, 7, 7, 6, 5, 4, 0, 3, 3, 7, 4, 3, 2, 6, 6, 7,
            3, 4, 5, 1, 1, 0, 4,
        ];

        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Vertex Buffer"),
            contents: bytemuck::cast_slice(&vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });

        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Index Buffer"),
            contents: bytemuck::cast_slice(indices),
            usage: wgpu::BufferUsages::INDEX,
        });

        let num_indices = indices.len() as u32;

        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct Uniforms {
            mvp: [[f32; 4]; 4],
        }

        impl Uniforms {
            fn new() -> Self {
                Self {
                    mvp: Matrix4::identity().into(),
                }
            }

            fn update(&mut self, rotation: f32) {
                let model = Matrix4::from_angle_y(Deg(rotation));
                let view = Matrix4::look_at_rh(
                    Point3::new(0.0, 0.0, 5.0),
                    Point3::new(0.0, 0.0, 0.0),
                    Vector3::unit_y(),
                );
                let proj = perspective(Deg(45.0), 1.0, 0.1, 100.0);
                self.mvp = (proj * view * model).into();
            }
        }

        let mut uniforms = Uniforms::new();

        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Uniform Buffer"),
            size: std::mem::size_of::<Uniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let uniform_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Uniform Bind Group Layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });

        let uniform_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Uniform Bind Group"),
            layout: &uniform_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            }],
        });

        let shader_source = r#"
        struct Uniforms {
            mvp: mat4x4<f32>,
        };

        @group(0) @binding(0)
        var<uniform> uniforms: Uniforms;

        struct VertexInput {
            @location(0) position: vec3<f32>,
            @location(1) color: vec3<f32>,
        };

        struct VertexOutput {
            @builtin(position) position: vec4<f32>,
            @location(0) color: vec3<f32>,
        };

        @vertex
        fn vs_main(input: VertexInput) -> VertexOutput {
            var output: VertexOutput;
            output.position = uniforms.mvp * vec4<f32>(input.position, 1.0);
            output.color = input.color;
            return output;
        }

        @fragment
        fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
            return vec4<f32>(input.color, 1.0);
        }
        "#;

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Cube Shader"),
            source: wgpu::ShaderSource::Wgsl(shader_source.into()),
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Render Pipeline Layout"),
            bind_group_layouts: &[&uniform_bind_group_layout],
            push_constant_ranges: &[],
        });

        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Render Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: &[Vertex::desc()],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format: TextureFormat::Rgba8UnormSrgb,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: Some(wgpu::Face::Back),
                unclipped_depth: false,
                polygon_mode: wgpu::PolygonMode::Fill,
                conservative: false,
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth24PlusStencil8,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
        });

        let depth_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Depth Texture"),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth24PlusStencil8,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });

        let depth_view = depth_texture.create_view(&Default::default());

        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Render Target"),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let view = texture.create_view(&Default::default());

        // FPS tracking
        let mut frame_count = 0u32;
        let mut fps = 0.0;
        let fps_interval = Duration::from_secs(1);
        let mut last_fps_check = Instant::now();
        let start = Instant::now();

        // Load font for FPS counter
        let font_data = include_bytes!("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf");
        let font = FontRef::try_from_slice(font_data).expect("Error constructing Font");
        let scale = PxScale::from(24.0);

        // N-buffering: create N output buffers up front
        const NUM_BUFFERS: usize = 3;
        let buffer_size = (size.width * size.height * 4) as u64;
        let output_buffers: Vec<wgpu::Buffer> = (0..NUM_BUFFERS)
            .map(|i| device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(&format!("Output Buffer {}", i)),
                size: buffer_size,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            }))
            .collect();
        let mut buffer_index = 0;

        // Track pending readbacks: (buffer_index, oneshot receiver)
        let mut pending: VecDeque<(usize, futures_intrusive::channel::shared::OneshotReceiver<Result<(), wgpu::BufferAsyncError>>)> = VecDeque::new();

        // Reusable buffers to minimize allocations
        let mut rgba = Vec::with_capacity((size.width * size.height * 4) as usize);
        let mut rgb = Vec::with_capacity((size.width * size.height * 3) as usize);
        let mut jpeg_data = Vec::with_capacity((size.width * size.height * 3) as usize); // Over-allocate for safety

        loop {
            frame_count += 1;
            let now = Instant::now();

            // Update FPS counter
            let elapsed_since_last_check = now.duration_since(last_fps_check);
            if elapsed_since_last_check >= fps_interval {
                fps = frame_count as f64 / elapsed_since_last_check.as_secs_f64();
                println!("FPS: {:.2}", fps);
                frame_count = 0;
                last_fps_check = now;
            }

            let elapsed = start.elapsed().as_secs_f32() * 60.0;
            uniforms.update(elapsed);
            queue.write_buffer(&uniform_buffer, 0, bytemuck::cast_slice(&[uniforms]));

            // If all buffers are pending, wait for the oldest to complete
            if pending.len() == NUM_BUFFERS {
                let (idx, rx) = pending.pop_front().unwrap();
                rx.receive().await.unwrap().unwrap();
                // Process completed buffer
                let output_buffer = &output_buffers[idx];
                let data = output_buffer.slice(..).get_mapped_range();
                rgba.clear();
                rgba.extend_from_slice(&data);
                drop(data);
                output_buffer.unmap();

                let mut img = ImageBuffer::<Rgba<u8>, Vec<u8>>::from_raw(size.width, size.height, rgba.clone()).unwrap();
                let fps_text = format!("FPS: {:.1}", fps);
                draw_text_mut(&mut img, Rgba([255u8, 255u8, 255u8, 255u8]), 10, 10, scale, &font, &fps_text);
                rgb.clear();
                for chunk in img.as_raw().chunks_exact(4) {
                    rgb.extend_from_slice(&chunk[0..3]);
                }
                let buffer = ImageBuffer::from_raw(size.width, size.height, rgb.clone()).unwrap();
                jpeg_data.clear();
                {
                    let mut cursor = Cursor::new(&mut jpeg_data);
                    let mut encoder = JpegEncoder::new_with_quality(&mut cursor, 80);
                    encoder.encode_image(&DynamicImage::ImageRgb8(buffer)).unwrap();
                }
                *stream_clone.write().await = jpeg_data.clone();
                version_clone.fetch_add(1, Ordering::SeqCst);
                notify_clone.notify_waiters();
            }

            // Render into the next available buffer
            let output_buffer = &output_buffers[buffer_index];
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Render Encoder"),
            });
            {
                let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("Render Pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color {
                                r: 0.1,
                                g: 0.2,
                                b: 0.3,
                                a: 1.0,
                            }),
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                        view: &depth_view,
                        depth_ops: Some(wgpu::Operations {
                            load: wgpu::LoadOp::Clear(1.0),
                            store: wgpu::StoreOp::Store,
                        }),
                        stencil_ops: None,
                    }),
                    occlusion_query_set: None,
                    timestamp_writes: None,
                });
                render_pass.set_pipeline(&render_pipeline);
                render_pass.set_bind_group(0, &uniform_bind_group, &[]);
                render_pass.set_vertex_buffer(0, vertex_buffer.slice(..));
                render_pass.set_index_buffer(index_buffer.slice(..), wgpu::IndexFormat::Uint16);
                render_pass.draw_indexed(0..num_indices, 0, 0..1);
            }
            encoder.copy_texture_to_buffer(
                wgpu::ImageCopyTexture {
                    texture: &texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::ImageCopyBuffer {
                    buffer: output_buffer,
                    layout: wgpu::ImageDataLayout {
                        offset: 0,
                        bytes_per_row: Some(4 * size.width),
                        rows_per_image: Some(size.height),
                    },
                },
                size,
            );
            queue.submit(Some(encoder.finish()));

            // Start async readback for this buffer
            let buffer_slice = output_buffer.slice(..);
            let (tx, rx) = oneshot_channel();
            buffer_slice.map_async(wgpu::MapMode::Read, move |v| tx.send(v).unwrap());
            pending.push_back((buffer_index, rx));

            // Alternate buffer index for next frame
            buffer_index = (buffer_index + 1) % NUM_BUFFERS;
        }
    });

    let make_svc = make_service_fn(move |_| {
        let latest = latest.clone();
        let frame_version = frame_version.clone();
        let notify = notify.clone();
        async move {
            Ok::<_, Infallible>(service_fn(move |_req: Request<Body>| {
                let latest = latest.clone();
                let frame_version = frame_version.clone();
                let notify = notify.clone();
                async move {
                    let stream = stream! {
                        let mut last_version = frame_version.load(Ordering::SeqCst);
                        loop {
                            // Wait for a new frame version
                            while frame_version.load(Ordering::SeqCst) == last_version {
                                notify.notified().await;
                            }
                            last_version = frame_version.load(Ordering::SeqCst);
                            let frame = latest.read().await.clone();
                            if !frame.is_empty() {
                                let header = format!(
                                    "--frame\r\nContent-Type: image/jpeg\r\nContent-Length: {}\r\n\r\n",
                                    frame.len()
                                );
                                yield Ok::<Bytes, Infallible>(Bytes::from(header));
                                yield Ok::<Bytes, Infallible>(Bytes::from(frame));
                                yield Ok::<Bytes, Infallible>(Bytes::from("\r\n"));
                                // Throttle to ~120 FPS for browser smoothness
                                tokio::time::sleep(Duration::from_millis(8)).await;
                            }
                        }
                    };

                    let mut resp = Response::new(Body::wrap_stream(stream));
                    resp.headers_mut().insert(
                        header::CONTENT_TYPE,
                        header::HeaderValue::from_static(
                            "multipart/x-mixed-replace; boundary=frame",
                        ),
                    );
                    Ok::<_, Infallible>(resp)
                }
            }))
        }
    });

    let addr = ([0, 0, 0, 0], 5000).into();
    println!("🚀 Spinning cube MJPEG stream live at http://{}/", addr);
    Server::bind(&addr).serve(make_svc).await.unwrap();
}
