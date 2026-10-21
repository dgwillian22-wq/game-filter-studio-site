"""Renderizador OpenGL: desloca a imagem pelo mapa de profundidade e monta a saida 3D."""
import math

import moderngl
import numpy as np

MODES = ["original", "pisca", "vaievem", "sbs", "anaglifo", "profundidade"]
MODE_LABELS = {
    "original": "Original",
    "pisca": "Pisca 3D (alterna esquerda/direita)",
    "vaievem": "Vai-e-vem 3D (suave)",
    "sbs": "Lado a lado (TV/monitor 3D)",
    "anaglifo": "Anaglifo (oculos vermelho/ciano)",
    "profundidade": "Mapa de profundidade",
}

VERT = """#version 330
out vec2 uv;
void main() {
    // triangulo gigante que cobre a tela inteira: ids 0,1,2 -> (0,0) (2,0) (0,2)
    vec2 c = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);
    uv = vec2(c.x, 1.0 - c.y);
    gl_Position = vec4(c * 2.0 - 1.0, 0.0, 1.0);
}
"""

FRAG = """#version 330
uniform sampler2D tex;
uniform sampler2D dep;
uniform int mode;
uniform float strength;
uniform float conv;
uniform float view;
in vec2 uv;
out vec4 frag;

// Deslocamento por profundidade com ray march (como parallax occlusion mapping): varre os
// niveis de profundidade do mais perto ao mais longe e fica com a primeira superficie que
// o raio do olho encontra. Sem buracos, sem recorte e o objeto da frente sempre vence.
vec3 eye(vec2 p, float v) {
    const int N = 24;
    float a = v * strength;
    float fPrev = texture(dep, p).r - 1.0;
    float dPrev = 1.0;
    float level = 0.0;
    for (int i = 1; i <= N; i++) {
        float dk = 1.0 - float(i) / float(N);
        float x = clamp(p.x + a * (dk - conv), 0.0, 1.0);
        float f = texture(dep, vec2(x, p.y)).r - dk;
        if (f >= 0.0) {
            float t = fPrev / (fPrev - f + 1e-6);
            level = mix(dPrev, dk, clamp(t, 0.0, 1.0));
            break;
        }
        fPrev = f;
        dPrev = dk;
        level = dk;
    }
    float qx = clamp(p.x + a * (level - conv), 0.0, 1.0);
    return texture(tex, vec2(qx, p.y)).rgb;
}

void main() {
    vec3 c;
    if (mode == 0) {
        c = texture(tex, uv).rgb;
    } else if (mode == 1 || mode == 2) {
        c = eye(uv, view);
    } else if (mode == 3) {
        if (uv.x < 0.5) c = eye(vec2(uv.x * 2.0, uv.y), -1.0);
        else c = eye(vec2((uv.x - 0.5) * 2.0, uv.y), 1.0);
    } else if (mode == 4) {
        vec3 l = eye(uv, -1.0);
        vec3 r = eye(uv, 1.0);
        c = vec3(l.r, r.g, r.b);
    } else {
        c = vec3(texture(dep, uv).r);
    }
    frag = vec4(c, 1.0);
}
"""

HUD_VERT = """#version 330
uniform vec4 rect;
out vec2 uv;
void main() {
    vec2 c = vec2(gl_VertexID & 1, (gl_VertexID >> 1) & 1);
    uv = vec2(c.x, 1.0 - c.y);
    gl_Position = vec4(mix(rect.xy, rect.zw, c), 0.0, 1.0);
}
"""

HUD_FRAG = """#version 330
uniform sampler2D hud;
in vec2 uv;
out vec4 frag;
void main() { frag = texture(hud, uv); }
"""


def make_context():
    """Contexto OpenGL sem janela (testes e exportacao)."""
    try:
        return moderngl.create_standalone_context()
    except Exception:
        return moderngl.create_standalone_context(backend="egl")


def view_at(mode, t, hz):
    """Posicao da camera (-1 esquerda ... +1 direita) no instante t."""
    if mode == "pisca":
        return 1.0 if int(math.floor(2.0 * hz * t)) % 2 else -1.0
    if mode == "vaievem":
        return math.sin(2.0 * math.pi * hz * t)
    return 0.0


class Renderer:
    def __init__(self, ctx):
        self.ctx = ctx
        self.prog = ctx.program(vertex_shader=VERT, fragment_shader=FRAG)
        self.vao = ctx.vertex_array(self.prog, [])
        self.hud_prog = ctx.program(vertex_shader=HUD_VERT, fragment_shader=HUD_FRAG)
        self.hud_vao = ctx.vertex_array(self.hud_prog, [])
        self.frame_tex = None
        self.depth_tex = None
        self.hud_tex = None
        self._set(self.prog, "tex", 0)
        self._set(self.prog, "dep", 1)
        self._set(self.hud_prog, "hud", 2)
        self.upload_depth(np.full((2, 2), 0.5, np.float32))

    @staticmethod
    def _set(prog, name, value):
        if name in prog:
            prog[name].value = value

    def upload_frame(self, bgr):
        h, w = bgr.shape[:2]
        if self.frame_tex is None or self.frame_tex.size != (w, h):
            self.frame_tex = self.ctx.texture((w, h), 3, alignment=1)
            self.frame_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
            self.frame_tex.repeat_x = self.frame_tex.repeat_y = False
            self.frame_tex.swizzle = "BGR1"
        self.frame_tex.write(np.ascontiguousarray(bgr).tobytes(), alignment=1)

    def upload_depth(self, depth):
        h, w = depth.shape
        if self.depth_tex is None or self.depth_tex.size != (w, h):
            self.depth_tex = self.ctx.texture((w, h), 1, dtype="f4")
            self.depth_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
            self.depth_tex.repeat_x = self.depth_tex.repeat_y = False
        self.depth_tex.write(np.ascontiguousarray(depth, np.float32).tobytes())

    def draw(self, fbo, viewport, mode, strength, conv, view):
        if self.frame_tex is None:
            return
        fbo.use()
        self.ctx.disable(moderngl.BLEND)
        self.frame_tex.use(0)
        self.depth_tex.use(1)
        self._set(self.prog, "mode", MODES.index(mode))
        self._set(self.prog, "strength", float(strength))
        self._set(self.prog, "conv", float(conv))
        self._set(self.prog, "view", float(view))
        fbo.viewport = viewport
        self.vao.render(moderngl.TRIANGLES, vertices=3)

    def set_hud(self, rgba, size):
        w, h = size
        if self.hud_tex is None or self.hud_tex.size != (w, h):
            self.hud_tex = self.ctx.texture((w, h), 4, alignment=1)
            self.hud_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.hud_tex.write(rgba, alignment=1)

    def draw_hud(self, fbo, win_size, margin=12):
        if self.hud_tex is None:
            return
        W, H = win_size
        w, h = self.hud_tex.size
        x0 = -1 + 2 * margin / W
        y1 = 1 - 2 * margin / H
        self._set(self.hud_prog, "rect", (x0, y1 - 2 * h / H, x0 + 2 * w / W, y1))
        fbo.use()
        fbo.viewport = (0, 0, W, H)
        self.ctx.enable(moderngl.BLEND)
        self.hud_tex.use(2)
        self.hud_vao.render(moderngl.TRIANGLE_STRIP, vertices=4)
        self.ctx.disable(moderngl.BLEND)


def fit_viewport(win_w, win_h, vid_w, vid_h):
    """Centraliza o video na janela mantendo a proporcao."""
    scale = min(win_w / vid_w, win_h / vid_h)
    w, h = max(1, int(vid_w * scale)), max(1, int(vid_h * scale))
    return ((win_w - w) // 2, (win_h - h) // 2, w, h)
