#!/usr/bin/env python3
"""
Streamlit · SRT ▶︎ Gemini Flash Experimental ▶︎ Zoom Clips
=============================================
Web-app que:
1. Carrega um arquivo `.srt` (upload).
2. Segmenta o texto a cada *N* palavras (padrão 20).
3. Usa **Gemini 2.0 Flash Experimental** para criar uma imagem por bloco.
4. Salva imagens, gera vídeos com zoom progressivo 100 → 130 % via FFmpeg.
5. (Opcional) Concatena todos os clipes em um vídeo final.

Autor: ChatGPT (abril 2025)
"""
from __future__ import annotations
import os
import io
import subprocess
from pathlib import Path
from typing import List

import streamlit as st
import pysrt
import tqdm
import google.genai as genai
from google.genai import types
from PIL import Image
from io import BytesIO

# ───────────────────────────────────────────
STYLE_BLOCK = (
    "Ultra-realistic, cinematic lighting, volumetric light, "
    "shallow depth-of-field, 35 mm lens, biblical times, "
    "ancient Middle-East setting, 16:9, no text"
)

# ───────────────────────────────────────────
# Helpers
# ───────────────────────────────────────────

def time_tag(t: pysrt.SubRipTime) -> str:
    return f"{t.hours:02d}_{t.minutes:02d}_{t.seconds:02d}_{int(t.milliseconds):03d}"

def duration_sec(start: pysrt.SubRipTime, end: pysrt.SubRipTime) -> float:
    return (end.ordinal - start.ordinal) / 1000.0

def srt_to_blocks(subs: List[pysrt.SubRipItem], words_per_img: int):
    blocks, curr_words, blk_start = [], [], None
    for sub in subs:
        words = sub.text.replace("\n", " ").split()
        if not words:
            continue
        blk_start = blk_start or sub.start
        curr_words.extend(words)
        while len(curr_words) >= words_per_img:
            text = " ".join(curr_words[:words_per_img])
            blocks.append({"start": blk_start, "end": sub.end, "text": text})
            curr_words = curr_words[words_per_img:]
            blk_start = sub.end if curr_words else None
    if curr_words:
        blocks.append({"start": blk_start, "end": subs[-1].end, "text": " ".join(curr_words)})
    return blocks

def gemini_image(client, context_blocks: List[str], text: str) -> bytes:
    ctx = "\n\n".join(context_blocks)
    prompt = (
        f"Previous scenes for continuity:\n{ctx}\n\n"
        f"Current scene:\n{text}\n\n"
        f"Generate ONE 16:9 IMAGE capturing ONLY the current scene. "
        f"Style: {STYLE_BLOCK}"
    )
    response = client.models.generate_content(
        model="gemini-2.0-flash-exp-image-generation",
        contents=prompt,
        config=types.GenerateContentConfig(
            response_modalities=["TEXT", "IMAGE"]
        )
    )
    for part in response.candidates[0].content.parts:
        if part.inline_data:
            return part.inline_data.data
    raise RuntimeError("Nenhuma imagem retornada.")

def ensure_ffmpeg():
    try:
        subprocess.run(["ffmpeg", "-version"], stdout=subprocess.DEVNULL, check=True)
    except Exception:
        st.error("FFmpeg não encontrado na PATH.")
        st.stop()

def zoom_filter(dur: float, fps: int, zoom_end: float, res: str):
    inc = (zoom_end - 1.0) / (dur * fps)
    return f"zoompan=z='if(lte(on,1),1,min(pzoom+{inc:.6f},{zoom_end}))':d={int(dur*fps)}:s={res},fps={fps}"

# ───────────────────────────────────────────
# Streamlit UI
# ───────────────────────────────────────────

st.set_page_config(page_title="SRT ▶︎ Gemini Flash", layout="wide")
st.title("🎞️ SRT → Gemini Flash → Zoom Clips")

# Sidebar
with st.sidebar:
    st.header("Configurações")
    api_key = st.secrets.get("GEMINI_API_KEY")
    if not api_key:
        st.error("A chave de API não foi encontrada. Configure em Settings > Secrets.")
        st.stop()

    client = genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(api_version='v1alpha')
    )

    words_per_img    = st.number_input("Palavras por imagem", 5, 50, 20)
    context_segments = st.slider("Blocos de contexto", 0, 5, 3)
    fps              = st.number_input("FPS", 15, 60, 30)
    resolution       = st.text_input("Resolução WxH", "1920x1080")
    zoom_end         = st.slider("Zoom final (%)", 110, 200, 130)
    concat_all       = st.checkbox("Gerar vídeo final", value=True)

uploaded = st.file_uploader("📂 Carregue um arquivo .srt", type="srt")
run_btn  = st.button("🚀 Processar")

if run_btn:
    if not uploaded:
        st.warning("Faça upload de um .srt primeiro.")
        st.stop()

    ensure_ffmpeg()

    out_dir  = Path("output")
    imgs_dir = out_dir / "images"
    clips_dir = out_dir / "clips"
    for p in (imgs_dir, clips_dir): p.mkdir(parents=True, exist_ok=True)

    subs = pysrt.from_string(uploaded.getvalue().decode("utf-8"))
    blocks = srt_to_blocks(subs, words_per_img)

    st.info(f"{len(blocks)} blocos de imagem serão gerados.")
    prog = st.progress(0.0)

    context_buffer: List[str] = []
    concat_entries: List[Path] = []

    for i, blk in enumerate(blocks, 1):
        img_bytes = gemini_image(client, context_buffer[-context_segments:], blk["text"])
        img_name = f"{time_tag(blk['start'])}-{time_tag(blk['end'])}.png"
        img_path = imgs_dir / img_name
        img_path.write_bytes(img_bytes)
        context_buffer.append(blk["text"])

        dur = duration_sec(blk["start"], blk["end"])
        clip_path = clips_dir / img_name.replace(".png", ".mp4")
        vf = zoom_filter(dur, fps, zoom_end/100, resolution)
        subprocess.run([
            "ffmpeg", "-y", "-loop", "1", "-i", str(img_path),
            "-vf", vf, "-t", f"{dur:.3f}", "-pix_fmt", "yuv420p",
            str(clip_path)
        ], check=True)
        concat_entries.append(clip_path)
        prog.progress(i/len(blocks))

    # Download
    st.header("🎁 Downloads")
    with st.expander("Imagens geradas"):
        for p in imgs_dir.iterdir():
            st.image(p.read_bytes(), caption=p.name)
            st.download_button(f"Baixar {p.name}", p.read_bytes(), file_name=p.name, mime="image/png")

    with st.expander("Clipes individuais"):
        for p in clips_dir.iterdir():
            st.video(str(p))
            st.download_button(f"Baixar {p.name}", p.read_bytes(), file_name=p.name, mime="video/mp4")

    if concat_all:
        concat_txt = out_dir / "concat.txt"
        concat_txt.write_text("\n".join([f"file '{p.as_posix()}'" for p in concat_entries]))
        final = out_dir / "final_video.mp4"
        subprocess.run([
            "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(concat_txt),
            "-c", "copy", str(final)
        ], check=True)
        st.success("🎬 Vídeo final gerado!")
        st.video(str(final))
        st.download_button("Baixar Vídeo Final", final.read_bytes(), file_name=final.name, mime="video/mp4")
    else:
        st.success("Processamento concluído.")

    st.write("Arquivos gerados em:", out_dir.resolve())
