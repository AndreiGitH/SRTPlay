#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
from typing import List

import streamlit as st
import pysrt
import google.genai as genai
from google.genai import types
from PIL import Image
from io import BytesIO

STYLE_SUFFIX = (
    "Ultra-realistic, cinematic lighting, volumetric light, shallow depth-of-field, "
    "35 mm lens, biblical times, ancient Middle-East setting, 16:9, no text."
)

def tag(t: pysrt.SubRipTime) -> str:
    return f"{t.hours:02d}_{t.minutes:02d}_{t.seconds:02d}_{int(t.milliseconds):03d}"

def agrupar_blocos(subs: List[pysrt.SubRipItem], min_w=20, max_w=30):
    blocos, txt, start = [], [], None
    for s in subs:
        words = s.text.replace("\n", " ").split()
        if not words:
            continue
        start = start or s.start
        txt.extend(words)
        if len(txt) >= min_w:
            blocos.append({"start": start, "end": s.end, "text": " ".join(txt)})
            txt, start = [], None
    if txt:
        blocos.append({"start": start, "end": subs[-1].end, "text": " ".join(txt)})
    return blocos

def gerar_prompt(client, texto: str) -> str:
    pedido = (
        "Create a concise, vivid, ultra-realistic image-generation prompt that represents "
        "this biblical scene:\n\n"
        f"{texto}\n\n"
        f"End the prompt with these quality parameters:\n{STYLE_SUFFIX}"
    )
    resp = client.models.generate_content(model="gemini-2.0-flash", contents=pedido)
    return resp.candidates[0].content.parts[0].text.strip()

def gerar_imagem(client, prompt: str) -> bytes:
    resp = client.models.generate_content(
        model="gemini-2.0-flash-exp-image-generation",
        contents=prompt,
        config=types.GenerateContentConfig(response_modalities=["IMAGE"]),
    )
    for part in resp.candidates[0].content.parts:
        if part.inline_data:
            return part.inline_data.data
    raise RuntimeError("No image returned.")

# ─── Streamlit UI ──────────────────────────
st.set_page_config(page_title="SRT ▶︎ Gemini Imagens", layout="wide")
st.title("🎞️  SRT → Gemini Flash → Imagens")

api_key = st.secrets.get("GEMINI_API_KEY")
if not api_key:
    st.error("Configure GEMINI_API_KEY em Settings ▸ Secrets")
    st.stop()

client_txt = genai.Client(api_key=api_key)  # v1beta
client_img = genai.Client(api_key=api_key, http_options=types.HttpOptions(api_version="v1alpha"))

min_words = st.sidebar.number_input("Mín. palavras/bloco", 10, 30, 20)
max_words = st.sidebar.number_input("Máx. palavras/bloco", 20, 60, 30)

uploaded = st.file_uploader("📂  Envie o arquivo .srt", type="srt")
if st.button("🚀 Gerar Imagens"):

    if not uploaded:
        st.warning("Faça upload do .srt primeiro.")
        st.stop()

    subs = pysrt.from_string(uploaded.getvalue().decode("utf-8"))
    blocos = agrupar_blocos(subs, min_words, max_words)
    st.info(f"{len(blocos)} blocos serão processados.")
    prog = st.progress(0.0)

    out_dir = Path("output_images"); out_dir.mkdir(exist_ok=True)

    for i, blk in enumerate(blocos, 1):
        prompt = gerar_prompt(client_txt, blk["text"])
        img_bytes = gerar_imagem(client_img, prompt)

        name = f"{tag(blk['start'])}-{tag(blk['end'])}.png"
        (out_dir / name).write_bytes(img_bytes)

        st.image(img_bytes, caption=name, use_column_width=True)
        st.download_button(f"Baixar {name}", img_bytes, file_name=name, mime="image/png")
        prog.progress(i / len(blocos))

    st.success("✔️ Todas as imagens geradas!")
    st.write("Arquivos salvos em:", out_dir.resolve())
