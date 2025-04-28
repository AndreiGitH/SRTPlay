#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
from typing import List
import time, io, zipfile

import streamlit as st
import pysrt
import google.genai as genai
from google.genai import types
from PIL import Image
from io import BytesIO

# —— estilo turbinado ——
STYLE_SUFFIX = (
    "Ultra-realistic, cinematic lighting, volumetric light, dramatic contrast, "
    "film still, epic composition, highly detailed, 4K HDR, masterpiece, "
    "shallow depth-of-field, 35 mm lens, photorealistic, biblical times, "
    "ancient Middle-East setting, 16:9 aspect ratio, no text."
)

# —— session state init ——
if "imgs" not in st.session_state:
    st.session_state["imgs"] = []  # armazena dicts {"name","bytes"}

# —— helpers de tempo ——
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
            blocos.append({
                "start": start,
                "end": s.end,
                "text": " ".join(txt[:max_w])
            })
            txt, start = [], None
    if txt:
        blocos.append({
            "start": start,
            "end": subs[-1].end,
            "text": " ".join(txt)
        })
    return blocos

# —— gerar prompt cinematográfico ——
def gerar_prompt(client_txt, texto: str) -> str:
    pedido = (
        "Create a concise, vivid, image generation prompt that represents "
        "this biblical scene. The prompt must end with the quality parameters and explicitly "
        "keep 16:9 aspect ratio.\n\n"
        f"Scene:\n{texto}\n\n"
        f"Quality parameters:\n{STYLE_SUFFIX}"
    )
    try:
        resp = client_txt.models.generate_content(
            model="gemini-2.0-flash",
            contents=pedido
        )
        cand = resp.candidates
        if cand and cand[0].content.parts:
            return cand[0].content.parts[0].text.strip()
    except Exception:
        pass
    # fallback
    return f"{texto}, {STYLE_SUFFIX}"

# —— gerar imagem com retentativas ——
def gerar_imagem(client_img, prompt: str, tries: int = 6) -> bytes | None:
    for _ in range(tries):
        try:
            resp = client_img.models.generate_content(
                model="gemini-2.0-flash-exp-image-generation",
                contents=[prompt],
                config=types.GenerateContentConfig(response_modalities=["TEXT", "IMAGE"])
            )
        except Exception:
            time.sleep(1.0)
            continue

        cand = resp.candidates
        if cand and cand[0].content.parts:
            for part in cand[0].content.parts:
                if part.inline_data:
                    return part.inline_data.data
        time.sleep(1.2)
    return None

# —— Streamlit UI ——
st.set_page_config(page_title="SRT ▶︎ Gemini Imagens", layout="wide")
st.title("🎞️ SRT → Gemini Flash → Imagens Cinematográficas")

api_key = st.secrets.get("GEMINI_API_KEY")
if not api_key:
    st.error("Configure GEMINI_API_KEY em Settings ▸ Secrets.")
    st.stop()

client_txt = genai.Client(api_key=api_key)
client_img = genai.Client(
    api_key=api_key,
    http_options=types.HttpOptions(api_version="v1alpha")
)

min_w = st.sidebar.number_input("Mín. palavras/bloco", 10, 30, 20)
max_w = st.sidebar.number_input("Máx. palavras/bloco", 20, 60, 30)

uploaded = st.file_uploader("📂 Faça upload do .srt", type="srt")

if st.button("🚀 Gerar Imagens"):
    # -> limpa estado antigo para evitar duplicação de keys
    st.session_state["imgs"] = []

    if not uploaded:
        st.warning("Envie um .srt primeiro.")
        st.stop()

    subs   = pysrt.from_string(uploaded.getvalue().decode("utf-8"))
    blocos = agrupar_blocos(subs, min_w, max_w)
    st.info(f"{len(blocos)} blocos serão processados.")
    prog = st.progress(0.0)

    out_dir = Path("output_images"); out_dir.mkdir(exist_ok=True)

    for i, blk in enumerate(blocos, 1):
        prompt = gerar_prompt(client_txt, blk["text"])
        img_bytes = gerar_imagem(client_img, prompt)

        if img_bytes is None:
            st.warning(f"⚠️ Bloco {i}: sem imagem, pulado.")
            prog.progress(i / len(blocos))
            continue

        fname = f"{tag(blk['start'])}-{tag(blk['end'])}.png"
        (out_dir / fname).write_bytes(img_bytes)

        # guarda em session_state
        st.session_state["imgs"].append({"name": fname, "bytes": img_bytes})
        prog.progress(i / len(blocos))

    st.success("✔️ Processamento concluído! Veja a galeria abaixo.")

# — galeria persistente —
if st.session_state["imgs"]:
    st.header("📸 Imagens Geradas")
    for idx, item in enumerate(st.session_state["imgs"]):
        st.image(item["bytes"], caption=item["name"], use_column_width=True)
        # key única combina índice + nome
        st.download_button(
            label=f"Baixar {item['name']}",
            data=item["bytes"],
            file_name=item["name"],
            mime="image/png",
            key=f"dl-{idx}-{item['name']}"
        )

    # ZIP on-the-fly
    zbuf = io.BytesIO()
    with zipfile.ZipFile(zbuf, "w", zipfile.ZIP_DEFLATED) as zf:
        for item in st.session_state["imgs"]:
            zf.writestr(item["name"], item["bytes"])
    zbuf.seek(0)

    st.download_button(
        "⬇️ Baixar todas as imagens (.zip)",
        data=zbuf,
        file_name="todas_as_imagens.zip",
        mime="application/zip",
        key="zip-all"
    )
