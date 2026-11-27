"""Rebuild the original supplementary illustrations. Shisei demos are checked in."""

import json
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2] / "assets/mini-apps/people-studio"
SAMPLES = ROOT / "samples"
BG, INK, SKIN = "#f3eee6", "#263043", "#bd8063"


def draw_reference(path, kind):
    im = Image.new("RGB", (384, 512), BG)
    draw = ImageDraw.Draw(im)
    draw.rounded_rectangle((20, 20, 364, 492), radius=24, fill="#e8dece")
    if kind in {"anime", "dyn"}:
        fabric = {"anime": "#71527c", "dyn": "#a44841"}[kind]
        hem = 445
        draw.polygon([(145, 118), (119, 134), (54, 251), (105, 281), (139, 224),
                      (117, hem), (268, hem), (246, 224), (280, 281), (332, 251),
                      (265, 134), (237, 118), (192, 151)], fill=fabric, outline=INK, width=3)
        draw.line([(152, 126), (192, 177), (237, 126)], fill="#e4c79e", width=7)
        draw.line([(192, 177), (192, hem - 4)], fill="#e4c79e", width=4)
        draw.polygon([(143, 251), (187, 251), (187, 276), (143, 276)], outline="#e4c79e", width=3)
    elif kind in {"expr", "hair"}:
        draw.ellipse((99, 85, 285, 310), fill=INK)
        draw.polygon([(165, 250), (219, 250), (225, 310), (305, 348), (337, 476),
                      (46, 476), (81, 348), (159, 310)], fill="#805343")
        draw.ellipse((132, 102, 250, 288), fill=SKIN)
        draw.pieslice((102, 69, 276, 211), 180, 332, fill=INK)
        for x in (164, 221):
            draw.ellipse((x - 10, 177, x + 10, 190), fill="#eee4d4")
            draw.ellipse((x - 4, 177, x + 4, 190), fill=INK)
        draw.line([(191, 188), (183, 218), (194, 218)], fill="#805343", width=3)
        if kind == "expr":
            draw.arc((161, 213, 224, 259), 8, 172, fill=INK, width=5)
        else:
            draw.line([(177, 243), (208, 241)], fill=INK, width=3)
    else:
        draw.rectangle((21, 319, 362, 469), fill="#cdbfae")
        if kind in {"scene", "light"}:
            draw.rectangle((78, 65, 294, 285), fill="#b4cad3", outline=INK, width=8)
            draw.line([(184, 67), (184, 285)], fill=INK, width=6)
            draw.line([(78, 187), (293, 187)], fill=INK, width=6)
            if kind == "light":
                draw.polygon([(106, 287), (292, 287), (355, 446), (199, 446)], fill="#f5dc9d")
        if kind in {"furn", "scene"}:
            draw.rounded_rectangle((96, 174, 239, 323), radius=18, fill="#8f6254", outline=INK, width=5)
            draw.rounded_rectangle((75, 298, 269, 335), radius=10, fill="#bd8268", outline=INK, width=5)
            draw.line([(101, 333), (89, 423)], fill=INK, width=9)
            draw.line([(240, 332), (252, 423)], fill=INK, width=9)
        if kind == "prop":
            draw.line([(192, 181), (192, 411)], fill=INK, width=10)
            draw.polygon([(152, 105), (232, 105), (265, 192), (118, 192)], fill="#bc673d", outline=INK, width=5)
            draw.ellipse((142, 401, 242, 426), fill=INK)
    im.save(SAMPLES / path, optimize=True)


def urls(name):
    path = f"samples/{name}.png"
    return dict(t=path, m=path, i=path, o=path)


def catalog(file, item):
    (ROOT / (file + ".json")).write_text(json.dumps([item], ensure_ascii=False) + "\n", encoding="utf-8")


def normal(kind, label, description, **metadata):
    draw_reference(f"{kind}.png", kind)
    return dict(id="studio-" + kind, ti=label, zh=label, en=description, **urls(kind), **metadata)


SAMPLES.mkdir(parents=True, exist_ok=True)
catalog("anime", normal("anime", "紫色动画礼服", "Purple illustrated costume", cat="礼服", col="紫色", item=["连衣裙"]))
dyn = normal("dyn", "赭红交领长袍", "Rust cross-collar robe", era="古风")
dyn["angles"] = [dict(zh="正面", en="front angle", **urls("dyn"))]
catalog("dynasties", dyn)
catalog("expressions", normal("expr", "微笑", "Gentle smile", emo="开心", int="轻"))
catalog("hair", normal("hair", "深色斜刘海", "Dark hair with side-swept bangs", cat="刘海"))
catalog("scenes", normal("scene", "窗边工作室", "Studio interior next to a window", cat="室内", ang="正面"))
catalog("lights", normal("light", "午后窗光", "Late afternoon window lighting", cat="自然光"))
catalog("props", normal("prop", "橙色立灯", "Orange floor lamp", cat="灯具"))
catalog("furniture", normal("furn", "木色软椅", "Upholstered chair", cat="椅子"))
