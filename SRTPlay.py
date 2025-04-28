#!/usr/bin/env python3
"""
Streamlit · SRT ▶︎ Gemini Flash ▶︎ Imagens
------------------------------------------
• Agrupa legendas até ≥20 e ≤30 palavras, terminando no próximo timestamp.
• Usa Gemini 2 Flash (texto) → cria prompt cinematográfico (EN).
• Usa Gemini 2 Flash Experimental (v1alpha) → gera imagem PNG.
• Salva imagem como  HH_MM_SS_mmm-HH_MM_SS_mmm.png .
• Exibe imagem + botão de download.
"""
from __future__ import annotations
from pathlib import Path
from typing import List

import streamlit as st
import pysrt
import google.genai as genai
from google.genai import types
from PIL import Image
from io import BytesIO

# —— Sufixo de qualidade fixo ——
STYLE_SUFFIX = (
    "Ultra-realistic, cinematic lighting, volumetric light, shallow depth-of-field, "
    "35 mm lens, biblical times, ancient Middle-East setting, 16:9, no text."
)

# —— Helpers de tempo ——
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

        # fecha bloco se atingiu min_w e estamos num marcador de tempo
        if len(txt) >= min_w:
            blocos.append({"start": start, "end": s.end, "text": " ".join(txt)})
            txt, start = [], None
    if txt:
        blocos.append({"start": start, "end": subs[-1].end, "text": " ".join(txt)})
    return blocos

# —— Gemini chamadas ——
def gerar_prompt(client_txt, texto: str) -> str:
    pedido = (
        "Create a concise, vivid, ultra-realistic image-generation prompt that represents "
        "this biblical scene:\n\n"
        f"{texto}\n\n"
        f"End the prompt with these quality parameters:\n{STYLE_SUFFIX}"
    )
    resp = client_txt.models.generate_content(
        model="gemini-2.0-flash",  # texto
        contents=pedido
    )
    return resp.candidates[0].content.parts[0].text.strip()

def gerar_imagem(client_img, prompt: str) -> bytes:
    resp = client_img.models.generate_content(
        model="gemini-2.0-flash-exp-image-generation",
        contents=[prompt],  # tem que ser lista
        config=types.GenerateContentConfig(
            response_modalities=["TEXT", "IMAGE"]  # TEM que ter TEXT e IMAGE
        ),
    )
    for part in resp.candidates[0].content.parts:
        if part.inline_data:           # bytes PNG
            return part.inline_data.data
    raise RuntimeError("No image returned.")

# —— Streamlit UI ——
st.set_page_config(page_title="SRT ▶︎ Gemini Imagens", layout="wide")
st.title("🎞️  SRT → Gemini Flash → Imagens Cinematográficas")

api_key = st.secrets.get("GEMINI_API_KEY")
if not api_key:
    st.error("Configure GEMINI_API_KEY em Settings ▸ Secrets.")
    st.stop()

# cliente texto (v1beta) e imagem (v1alpha)
client_txt = genai.Client(api_key=api_key)  # default v1beta
client_img = genai.Client(
    api_key=api_key,
    http_options=types.HttpOptions(api_version="v1alpha")
)

min_words = st.sidebar.number_input("Mín. palavras/bloco", 10, 30, 20)
max_words = st.sidebar.number_input("Máx. palavras/bloco", 20, 60, 30)

uploaded = st.file_uploader("📂 Faça upload do .srt", type="srt")

if st.button("🚀 Gerar Imagens"):
    if not uploaded:
        st.warning("Envie um arquivo .srt primeiro.")
        st.stop()

    subs = pysrt.from_string(uploaded.getvalue().decode("utf-8"))
    blocos = agrupar_blocos(subs, min_words, max_words)
    st.info(f"{len(blocos)} blocos serão processados.")
    prog = st.progress(0.0)

    out_dir = Path("output_images"); out_dir.mkdir(exist_ok=True)

    for i, blk in enumerate(blocos, 1):
        # 1) Prompt cinematográfico
        prompt = gerar_prompt(client_txt, blk["text"])
        # 2) Imagem PNG
        img_bytes = gerar_imagem(client_img, prompt)

        fname = f"{tag(blk['start'])}-{tag(blk['end'])}.png"
        (out_dir / fname).write_bytes(img_bytes)

        # Exibe + download
        st.image(img_bytes, caption=fname, use_column_width=True)
        st.download_button(
            label=f"Baixar {fname}",
            data=img_bytes,
            file_name=fname,
            mime="image/png"
        )
        prog.progress(i / len(blocos))

    st.success("✔️  Todas as imagens geradas!")
    st.write("Arquivos salvos em:", out_dir.resolve())
