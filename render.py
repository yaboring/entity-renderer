import moderngl
import numpy as np
from PIL import Image
from pyrr import Matrix44
import time
import threading
import io
from flask import Flask, Response

# --- OpenGL context ---
ctx = moderngl.create_standalone_context(backend='egl')
size = (512, 512)
fbo = ctx.simple_framebuffer(size)
fbo.use()
ctx.enable(moderngl.DEPTH_TEST)

# --- Geometry ---
vertices = np.array([
    -1, -1,  1,        1, 0, 0,
     1, -1,  1,        1, 0, 0,
     1,  1,  1,        1, 0, 0,
    -1,  1,  1,        1, 0, 0,
    -1, -1, -1,        0, 1, 0,
     1, -1, -1,        0, 1, 0,
     1,  1, -1,        0, 1, 0,
    -1,  1, -1,        0, 1, 0,
], dtype='f4')

indices = np.array([
    0, 1, 2, 2, 3, 0,
    4, 5, 6, 6, 7, 4,
    3, 2, 6, 6, 7, 3,
    0, 1, 5, 5, 4, 0,
    1, 2, 6, 6, 5, 1,
    0, 3, 7, 7, 4, 0,
], dtype='i4')

vertex_shader = '''
#version 330
in vec3 in_position;
in vec3 in_color;
uniform mat4 mvp;
out vec3 v_color;
void main() {
    gl_Position = mvp * vec4(in_position, 1.0);
    v_color = in_color;
}
'''

fragment_shader = '''
#version 330
in vec3 v_color;
out vec4 f_color;
void main() {
    f_color = vec4(v_color, 1.0);
}
'''

prog = ctx.program(vertex_shader=vertex_shader, fragment_shader=fragment_shader)
vbo = ctx.buffer(vertices.tobytes())
ibo = ctx.buffer(indices.tobytes())
vao = ctx.vertex_array(prog, [(vbo, '3f 3f', 'in_position', 'in_color')], index_buffer=ibo)

# --- Camera ---
proj = Matrix44.perspective_projection(45.0, size[0] / size[1], 0.1, 100.0)
lookat = Matrix44.look_at((3, 3, 3), (0, 0, 0), (0, 1, 0))

# --- Shared frame storage ---
latest_jpeg = None
lock = threading.Lock()

# --- Flask app ---
app = Flask(__name__)

@app.route('/')
def stream():
    def generate():
        while True:
            with lock:
                frame = latest_jpeg
            if frame:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            time.sleep(1/30)
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

def run_flask():
    app.run(host='0.0.0.0', port=5000, threaded=True)

# --- Start Flask in background ---
threading.Thread(target=run_flask, daemon=True).start()

# --- Main render loop ---
frame_count = 0
while True:
    angle = frame_count * 0.03
    rotation = Matrix44.from_y_rotation(angle)
    mvp = proj * lookat * rotation
    prog['mvp'].write(mvp.astype('f4').tobytes())

    ctx.clear(0.2, 0.4, 0.6)
    ctx.enable(moderngl.DEPTH_TEST)
    vao.render()

    pixels = fbo.read(components=3)
    img = Image.frombytes('RGB', size, pixels)
    img = img.transpose(Image.FLIP_TOP_BOTTOM)

    buf = io.BytesIO()
    img.save(buf, format='JPEG')
    with lock:
        latest_jpeg = buf.getvalue()

    frame_count += 1
    time.sleep(1 / 30)
