"""Player 3D em tempo real: abre qualquer video e mostra com profundidade."""
import argparse
import os
import sys
import time

import moderngl
import pygame

from . import __version__
from .depth import AsyncDepth, load_depth_model
from .render import MODE_LABELS, MODES, Renderer, fit_viewport, view_at
from .source import Playback

VIDEO_EXT = "mp4 mkv avi mov webm m4v wmv flv ts mpg mpeg"
HELP = ("Espaco pausa | <- -> pula 5s | 0-5 modos | +/- forca | [ ] convergencia | , . velocidade do pisca | "
        "I inverte profundidade | F tela cheia | O abrir | H ajuda | Esc sai")


def parse_args(argv=None):
    p = argparse.ArgumentParser(prog="wlss3d", description="Transforma qualquer video em 3D em tempo real")
    p.add_argument("video", nargs="?", help="arquivo de video (ou arraste para a janela)")
    p.add_argument("--mode", default="pisca", choices=MODES)
    p.add_argument("--strength", type=float, default=0.04, help="forca do 3D (fracao da largura)")
    p.add_argument("--convergence", type=float, default=0.5, help="0=tudo salta da tela, 1=tudo entra")
    p.add_argument("--hz", type=float, default=3.0, help="ciclos por segundo do pisca/vai-e-vem")
    p.add_argument("--model", help="caminho de um modelo ONNX de profundidade (padrao: baixa o Depth Anything V2 Small)")
    p.add_argument("--depth-size", type=int, default=308, help="lado maior da imagem enviada a IA (menor = mais rapido)")
    p.add_argument("--no-ai", action="store_true", help="nao usa IA (profundidade aproximada, so para teste)")
    p.add_argument("--invert-depth", action="store_true")
    p.add_argument("--no-audio", action="store_true")
    p.add_argument("--loop", action="store_true")
    p.add_argument("--fullscreen", action="store_true")
    p.add_argument("--export", metavar="SAIDA.mp4", help="converte o video em vez de abrir o player")
    return p.parse_args(argv)


def pick_file():
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        path = filedialog.askopenfilename(
            title="Escolha um video",
            filetypes=[("Videos", " ".join(f"*.{e}" for e in VIDEO_EXT.split())), ("Todos", "*.*")])
        root.destroy()
        return path or None
    except Exception:  # noqa: BLE001
        return None


class Hud:
    def __init__(self):
        pygame.font.init()
        self.font = pygame.font.SysFont("segoeui,arial,dejavusans", 18)
        self.text = None

    def build(self, lines):
        if lines == self.text:
            return None
        self.text = lines
        surfs = [self.font.render(s, True, (255, 255, 255)) for s in lines]
        w = max(s.get_width() for s in surfs) + 20
        h = sum(s.get_height() for s in surfs) + 16
        box = pygame.Surface((w, h), pygame.SRCALPHA)
        box.fill((0, 0, 0, 170))
        y = 8
        for s in surfs:
            box.blit(s, (10, y))
            y += s.get_height()
        return pygame.image.tostring(box, "RGBA"), (w, h)


def main(argv=None):
    args = parse_args(argv)
    if args.export:
        return run_export(args)
    return run_player(args)


def make_model(args):
    if args.no_ai:
        from .depth import FallbackDepth

        return FallbackDepth(args.invert_depth)
    return load_depth_model(args.model, args.depth_size, args.invert_depth,
                            progress=lambda f: print(f"\rBaixando modelo de IA... {f * 100:4.0f}%", end="", flush=True))


def run_export(args):
    from .export import export_video

    if not args.video:
        sys.exit("Informe o video de entrada para exportar.")
    model = make_model(args)
    n = export_video(args.video, args.export, model, args.mode, args.strength, args.convergence, args.hz,
                     progress=lambda f: print(f"\rConvertendo... {f * 100:4.0f}%", end="", flush=True))
    print(f"\n{n} quadros -> {args.export}")


def run_player(args):
    pygame.init()
    pygame.display.set_caption(f"WLSS 3D Video {__version__}")
    flags = pygame.OPENGL | pygame.DOUBLEBUF | pygame.RESIZABLE
    if args.fullscreen:
        flags |= pygame.FULLSCREEN
    pygame.display.set_mode((1280, 720), flags, vsync=1)
    ctx = moderngl.create_context()
    renderer = Renderer(ctx)
    screen = ctx.screen
    hud = Hud()
    model = make_model(args)
    depth = AsyncDepth(model)

    mode, strength, conv, hz = args.mode, args.strength, args.convergence, args.hz
    playback = None
    vid = (16, 9)
    path = args.video
    last_ver = -1
    show_until = time.perf_counter() + 6.0
    help_on = True
    fps_t, fps_n, fps = time.perf_counter(), 0, 0.0

    def open_video(p):
        nonlocal playback, vid, last_ver, path
        if playback:
            playback.close()
        playback = Playback(p, use_audio=not args.no_audio)
        vid = (playback.decoder.width, playback.decoder.height)
        path, last_ver = p, -1

    if path:
        open_video(path)
    elif (picked := pick_file()):
        open_video(picked)

    running = True
    while running:
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False
            elif e.type == pygame.DROPFILE:
                open_video(e.file)
            elif e.type == pygame.KEYDOWN:
                show_until = time.perf_counter() + 3.0
                k = e.key
                if k in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
                elif k == pygame.K_SPACE and playback:
                    playback.toggle()
                elif k == pygame.K_RIGHT and playback:
                    playback.seek_by(5)
                elif k == pygame.K_LEFT and playback:
                    playback.seek_by(-5)
                elif pygame.K_0 <= k <= pygame.K_5:
                    mode = MODES[k - pygame.K_0]
                elif k in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
                    strength = min(0.15, strength + 0.005)
                elif k in (pygame.K_MINUS, pygame.K_KP_MINUS):
                    strength = max(0.0, strength - 0.005)
                elif k == pygame.K_LEFTBRACKET:
                    conv = max(0.0, conv - 0.05)
                elif k == pygame.K_RIGHTBRACKET:
                    conv = min(1.0, conv + 0.05)
                elif k == pygame.K_COMMA:
                    hz = max(0.5, hz - 0.5)
                elif k == pygame.K_PERIOD:
                    hz = min(15.0, hz + 0.5)
                elif k == pygame.K_i:
                    model.invert = not getattr(model, "invert", False)
                elif k == pygame.K_h:
                    help_on = not help_on
                elif k in (pygame.K_f, pygame.K_F11, pygame.K_RETURN):
                    pygame.display.toggle_fullscreen()
                elif k == pygame.K_o and (picked := pick_file()):
                    open_video(picked)

        win = pygame.display.get_window_size()
        screen.viewport = (0, 0, *win)
        screen.clear(0.0, 0.0, 0.0, 1.0)

        if playback:
            playback.update()
            item = playback.decoder.get(playback.time())
            if item:
                renderer.upload_frame(item[1])
                depth.submit(item[1])
            ver, d = depth.latest()
            if d is not None and ver != last_ver:
                renderer.upload_depth(d)
                last_ver = ver
            if playback.decoder.at_end:
                if args.loop:
                    playback.seek(0.0)
                elif playback.playing:
                    playback.toggle()
            t = time.perf_counter()
            renderer.draw(screen, fit_viewport(*win, *vid), mode, strength, conv, view_at(mode, t, hz))

        fps_n += 1
        now = time.perf_counter()
        if now - fps_t >= 0.5:
            fps, fps_n, fps_t = fps_n / (now - fps_t), 0, now

        if now < show_until or not playback:
            if playback:
                lines = [os.path.basename(path), MODE_LABELS[mode],
                         f"Forca {strength * 100:.1f}%   Convergencia {conv:.2f}   Pisca {hz:.1f} Hz",
                         f"Profundidade: {model.provider} {depth.ms:.0f} ms   Tela {fps:.0f} fps"
                         f"   {'PAUSADO' if not playback.playing else ''}"]
            else:
                lines = ["WLSS 3D Video", "Arraste um video para esta janela ou aperte O para abrir"]
            if help_on:
                lines.append(HELP)
            built = hud.build(lines)
            if built:
                renderer.set_hud(*built)
            renderer.draw_hud(screen, win)

        pygame.display.flip()

    depth.stop()
    if playback:
        playback.close()
    pygame.quit()
