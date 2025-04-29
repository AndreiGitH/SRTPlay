#!/usr/bin/env python3
"""
Streamlit · SRT ▶︎ Gemini (prompt) ▶︎ Replicate (imagem)
-----------------------------------------------------
• Agrupa legendas em blocos de 20–30 palavras.
• Gera prompt cinematográfico com Gemini 2.0 Flash (texto).
• Gera imagem via Replicate API (minimax/image-01).
• Exibe galeria, download individual, ZIP e TXT de prompts.
"""
from __future__ import annotations
import os, io, zipfile, time, re
from pathlib import Path
from typing import List

import streamlit as st
import pysrt
import replicate                                # pip install replicate
from PIL import Image
from io import BytesIO

import google.genai as genai
from google.genai import types

# ─── Configurações ─────────────────────────────
STYLE_SUFFIX = (
    "Ultra-realistic, cinematic lighting, volumetric light, dramatic contrast, "
    "film still, epic composition, highly detailed, 4K HDR, masterpiece, "
    "shallow depth-of-field, 35 mm lens, photorealistic, biblical times, "
    "ancient Middle-East setting, 16:9 aspect ratio, no text overlay."
)

# ─── session_state ─────────────────────────────
if "imgs" not in st.session_state:
    st.session_state["imgs"] = []  # [{"name","bytes","prompt"}]

# ─── Helpers ────────────────────────────────────
def tag(t: pysrt.SubRipTime) -> str:
    return f"{t.hours:02d}_{t.minutes:02d}_{t.seconds:02d}_{int(t.milliseconds):03d}"

def agrupar_blocos(subs: List[pysrt.SubRipItem], min_w=20, max_w=30):
    blocos, txt, start = [], [], None
    for s in subs:
        words = s.text.replace("\n"," ").split()
        if not words: continue
        start = start or s.start
        txt.extend(words)
        if len(txt) >= min_w:
            blocos.append({
                "start": start,
                "end":   s.end,
                "text":  " ".join(txt[:max_w])
            })
            txt, start = [], None
    if txt:
        blocos.append({
            "start": start,
            "end":   subs[-1].end,
            "text":  " ".join(txt)
        })
    return blocos

def clean_prompt(raw: str) -> str:
    parts = re.split(r"\*{0,2}Prompt\*{0,2}:\s*", raw)
    body = parts[-1] if len(parts)>1 else raw
    body = re.sub(r"^Here(?:'|’)s a [^:]+:\s*","", body)
    body = body.replace("*","").strip()
    return re.sub(r"\s+", " ", body)

def gerar_prompt(client_txt, texto: str) -> str:
    pedido = (
        "Create a concise, vivid, ultra-realistic image generation prompt that represents "
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
        raw = resp.candidates[0].content.parts[0].text or ""
        prompt = clean_prompt(raw)
        if prompt and not prompt.startswith(texto[:10]):
            return prompt
    except Exception:
        pass
    # fallback
    return f"{texto}, {STYLE_SUFFIX}"

# ─── Replicate image generation ──────────────────
def gerar_imagem_replicate(prompt: str, aspect_ratio: str="16:9") -> bytes:
    # assume REPLICATE_API_TOKEN está em st.secrets
    output = replicate.run(
        "minimax/image-01",
        input={"prompt": prompt, "aspect_ratio": aspect_ratio}
    )
    # output é lista de file-like ou HTTPResponse
    first = output[0]
    return first.read()

# ─── Streamlit UI ───────────────────────────────
st.set_page_config(page_title="SRT ▶︎ Replicate Imagens", layout="wide")
st.title("🎞️ SRT → Gemini (prompt) → Replicate (imagem)")

# token do Replicate no Secrets
rep_token = st.secrets.get("REPLICATE_API_TOKEN")
if not rep_token:
    st.error("Defina REPLICATE_API_TOKEN em Settings → Secrets.")
    st.stop()
# exporta pro ambiente (opcional, replicate pode ler sozinho)
os.environ["REPLICATE_API_TOKEN"] = rep_token

# Cliente Gemini para prompts
api_key = st.secrets.get("GEMINI_API_KEY","")
client_txt = genai.Client(api_key=api_key) if api_key else None

# Controles
min_w = st.sidebar.number_input("Mín. palavras/bloco", 10, 30, 20)
max_w = st.sidebar.number_input("Máx. palavras/bloco", 20, 60, 30)

uploaded = st.file_uploader("📂 Envie seu arquivo .srt", type="srt")

if st.button("🚀 Gerar Imagens"):
    st.session_state["imgs"] = []

    if not uploaded:
        st.warning("Faça upload do .srt primeiro.")
        st.stop()

    subs   = pysrt.from_string(uploaded.getvalue().decode("utf-8"))
    blocos = agrupar_blocos(subs, min_w, max_w)
    st.info(f"{len(blocos)} blocos serão processados.")
    prog = st.progress(0.0)
    out_dir = Path("output_images"); out_dir.mkdir(exist_ok=True)

    for i, blk in enumerate(blocos,1):
        # 1) gerar prompt (ou use blk["text"] diretamente)
        prompt = gerar_prompt(client_txt, blk["text"]) if client_txt else blk["text"]
        # 2) gerar imagem via Replicate
        try:
            img_bytes = gerar_imagem_replicate(prompt, aspect_ratio="16:9")
        except Exception as e:
            st.warning(f"Bloco {i} falhou: {e}")
            prog.progress(i/len(blocos))
            continue

        name = f"{tag(blk['start'])}-{tag(blk['end'])}.png"
        (out_dir/name).write_bytes(img_bytes)
        st.session_state["imgs"].append({"name":name,"bytes":img_bytes,"prompt":prompt})
        prog.progress(i/len(blocos))

    st.success("✔️ Imagens geradas!")

# galeria + downloads
if st.session_state["imgs"]:
    st.header("📸 Galeria de Imagens")
    for idx,item in enumerate(st.session_state["imgs"]):
        st.image(item["bytes"], caption=item["name"], use_column_width=True)
        st.download_button(f"Baixar {item['name']}",
                           item["bytes"],
                           file_name=item["name"],
                           mime="image/png",
                           key=f"img-{idx}")
    # ZIP on-the-fly
    buf = io.BytesIO()
    with zipfile.ZipFile(buf,"w",zipfile.ZIP_DEFLATED) as zf:
        for item in st.session_state["imgs"]:
            zf.writestr(item["name"], item["bytes"])
    buf.seek(0)
    st.download_button("⬇️ Baixar ZIP (.zip)", buf, "all_images.zip", "application/zip")
    # prompts.txt
    txt = "\n\n".join(f"{itm['name']}: {itm['prompt']}" for itm in st.session_state["imgs"])
    st.download_button("⬇️ Baixar Prompts (.txt)", txt, "prompts.txt", "text/plain")
