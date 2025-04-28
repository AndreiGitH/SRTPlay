#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
from typing import List
import io, zipfile                                  # ← novo
import streamlit as st
import pysrt
import google.genai as genai
from google.genai import types
from PIL import Image
from io import BytesIO

# —— Estilo turbinado ——
STYLE_SUFFIX = (
    "Ultra-realistic, cinematic lighting, volumetric light, dramatic contrast, "
    "film still, epic composition, highly detailed, 4K HDR, masterpiece, "
    "shallow depth-of-field, 35 mm lens, photorealistic, biblical times, "
    "ancient Middle-East setting, 16:9, no text."
)

# … (funções tag, agrupar_blocos, gerar_prompt, gerar_imagem idênticas) …

# —— Streamlit UI ——
st.set_page_config(page_title="SRT ▶︎ Gemini Imagens", layout="wide")
st.title("🎞️  SRT → Gemini Flash → Imagens Cinematográficas")

api_key = st.secrets.get("GEMINI_API_KEY")
if not api_key:
    st.error("Configure GEMINI_API_KEY em Settings ▸ Secrets.")
    st.stop()

client_txt = genai.Client(api_key=api_key)  # v1beta
client_img = genai.Client(api_key=api_key,
                          http_options=types.HttpOptions(api_version="v1alpha"))

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
    zip_buffer = io.BytesIO()                 # buffer zip em memória
    zip_file = zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED)

    for i, blk in enumerate(blocos, 1):
        prompt = gerar_prompt(client_txt, blk["text"])
        img_bytes = gerar_imagem(client_img, prompt)
        fname = f"{tag(blk['start'])}-{tag(blk['end'])}.png"

        # salva em disco e adiciona ao zip
        (out_dir / fname).write_bytes(img_bytes)
        zip_file.writestr(fname, img_bytes)

        # exibe / download individual
        st.image(img_bytes, caption=fname, use_column_width=True)
        st.download_button(f"Baixar {fname}", img_bytes,
                           file_name=fname, mime="image/png")
        prog.progress(i / len(blocos))

    zip_file.close()                          # finaliza o zip
    zip_buffer.seek(0)

    # botão para baixar todas as imagens
    st.download_button(
        "⬇️ Baixar todas as imagens (.zip)",
        data=zip_buffer.read(),
        file_name="todas_as_imagens.zip",
        mime="application/zip"
    )

    st.success("✔️  Todas as imagens geradas!")
    st.write("Arquivos salvos em:", out_dir.resolve())
