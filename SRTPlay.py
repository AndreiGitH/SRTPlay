#!/usr/bin/env python3
"""
Streamlit · SRT ▶︎ Gemini Flash ▶︎ Imagens
------------------------------------------
• Agrupa legendas até ≥20 e ≤30 palavras, terminando no próximo timestamp.
• Gemini 2 Flash (texto) cria prompt cinematográfico (EN).
• Gemini 2 Flash-Exp (v1alpha) gera imagem. Tenta até 3×; se falhar, pula bloco.
• Salva PNG:  HH_MM_SS_mmm-HH_MM_SS_mmm.png .
• Exibe imagem + download individual + botão ZIP com todas as imagens.
"""
from __future__ import annotations
from pathlib import Path
from typing import List
import time, io, zipfile                       # zip buffer
import streamlit as st
import pysrt
import google.genai as genai
from google.genai import types
from PIL import Image
from io import BytesIO   # (Pillow já no requirements)

# —— Sufixo de qualidade turbinado ——
STYLE_SUFFIX = (
    "Ultra-realistic, cinematic lighting, volumetric light, dramatic contrast, "
    "film still, epic composition, highly detailed, 4K HDR, masterpiece, "
    "shallow depth-of-field, 35 mm lens, photorealistic, biblical times, "
    "ancient Middle-East setting, 16:9, no text."
)

# —— Helpers de tempo ——————————————————
def tag(t: pysrt.SubRipTime) -> str:
    return f"{t.hours:02d}_{t.minutes:02d}_{t.seconds:02d}_{int(t.milliseconds):03d}"

def agrupar_blocos(subs: List[pysrt.SubRipItem], min_w=20, max_w=30):
    """Agrupa textos até passar de min_w palavras; fecha no marcador atual."""
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

# —— Gemini helpers ————————————————————
def gerar_prompt(client_txt, texto: str) -> str:
    """
    Pede ao modelo um prompt cinematográfico em inglês.
    Se o modelo não devolver content.parts, retorna fallback.
    """
    pedido = (
        "Create a concise, vivid, ultra-realistic image-generation prompt that represents "
        "this biblical scene:\n\n"
        f"{texto}\n\n"
        f"End the prompt with these quality parameters:\n{STYLE_SUFFIX}"
    )

    try:
        resp = client_txt.models.generate_content(
            model="gemini-2.0-flash",
            contents=pedido
        )
        parts = resp.candidates[0].content.parts if resp.candidates else None
        if parts:
            return parts[0].text.strip()
    except Exception as exc:
        st.warning(f"⚠️  Falha ao criar prompt: {exc}")

    # Fallback: devolve o próprio texto + estilo
    return f"{texto}, {STYLE_SUFFIX}"


def gerar_imagem(client_img, prompt: str, tentativas: int = 3) -> bytes | None:
    """
    Tenta até `tentativas` vezes gerar PNG. Devolve bytes ou None.
    """
    for _ in range(tentativas):
        resp = client_img.models.generate_content(
            model="gemini-2.0-flash-exp-image-generation",
            contents=[prompt],
            config=types.GenerateContentConfig(
                response_modalities=["TEXT", "IMAGE"]
            ),
        )
        for part in resp.candidates[0].content.parts:
            if part.inline_data:
                return part.inline_data.data
        time.sleep(1.2)   # breve pausa antes de tentar novamente
    return None

# —— Streamlit UI ————————————————————
st.set_page_config(page_title="SRT ▶︎ Gemini Imagens", layout="wide")
st.title("🎞️  SRT → Gemini Flash → Imagens Cinematográficas")

api_key = st.secrets.get("GEMINI_API_KEY")
if not api_key:
    st.error("Configure GEMINI_API_KEY em Settings ▸ Secrets.")
    st.stop()

client_txt = genai.Client(api_key=api_key)  # v1beta
client_img = genai.Client(
    api_key=api_key,
    http_options=types.HttpOptions(api_version="v1alpha")
)

min_w = st.sidebar.number_input("Mín. palavras/bloco", 10, 30, 20)
max_w = st.sidebar.number_input("Máx. palavras/bloco", 20, 60, 30)

uploaded = st.file_uploader("📂  Faça upload do .srt", type="srt")

# ————— PROCESSAR —————
if st.button("🚀 Gerar Imagens"):
    if not uploaded:
        st.warning("Envie um arquivo .srt primeiro.")
        st.stop()

    subs   = pysrt.from_string(uploaded.getvalue().decode("utf-8"))
    blocos = agrupar_blocos(subs, min_w, max_w)
    st.info(f"{len(blocos)} blocos serão processados.")
    prog = st.progress(0.0)

    out_dir = Path("output_images"); out_dir.mkdir(exist_ok=True)
    zip_buffer = io.BytesIO()
    zip_file   = zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED)

    geradas = 0
    for i, blk in enumerate(blocos, 1):
        prompt = gerar_prompt(client_txt, blk["text"])
        img_bytes = gerar_imagem(client_img, prompt)

        if img_bytes is None:
            st.warning(f"⚠️  Bloco {i}: nenhuma imagem retornada, pulado.")
            prog.progress(i / len(blocos))
            continue  # pula para o próximo bloco

        fname = f"{tag(blk['start'])}-{tag(blk['end'])}.png"
        (out_dir / fname).write_bytes(img_bytes)
        zip_file.writestr(fname, img_bytes)
        geradas += 1

        st.image(img_bytes, caption=fname, use_column_width=True)
        st.download_button(f"Baixar {fname}", img_bytes,
                           file_name=fname, mime="image/png")
        prog.progress(i / len(blocos))

    zip_file.close(); zip_buffer.seek(0)

    if geradas:
        st.download_button(
            "⬇️ Baixar todas as imagens (.zip)",
            data=zip_buffer.read(),
            file_name="todas_as_imagens.zip",
            mime="application/zip"
        )
        st.success(f"✔️  {geradas} imagens geradas! (pulou {len(blocos)-geradas})")
    else:
        st.error("Nenhuma imagem foi gerada com sucesso.")

    st.write("Arquivos salvos em:", out_dir.resolve())
