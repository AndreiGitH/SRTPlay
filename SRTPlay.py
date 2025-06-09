#!/usr/bin/env python3
from __future__ import annotations
import os, io, zipfile, time, re
from pathlib import Path
from typing import List

import streamlit as st
import pysrt
import google.genai as genai
from google.genai import types
from PIL import Image
from io import BytesIO

# ─── Configurações ─────────────────────────────
STYLE_SUFFIX = (
    "abstract design, black background, "
    #"pencil sketch, colored, high detailed, black background, realistic, "
    #"ancient Middle-East setting, wide, "
    "aspect_ratio=16:9, size=1024x574."
    #"pencil sketch colored, textured paper, high detailed, "
    #"pencil sketch colored, visible strokes, high detailed, textured paper, "
    #"Ultra-realistic, "
    #"pencil sketch, colored pencil style, high detailed,  "
    #"Ultra-realistic, cinematic lighting, volumetric light, dramatic contrast, "
    #"film still, epic composition, highly detailed, masterpiece, "
    #"shallow depth-of-field, 35 mm lens, biblical times, "
)

# ─── session_state ─────────────────────────────
if "imgs" not in st.session_state:
    st.session_state["imgs"] = []    # [{"name","bytes","prompt"}]
if "blocos" not in st.session_state:
    st.session_state["blocos"] = []  # guarda blocos após agrupar

# ─── Helpers de tempo e prompt ──────────────────
def tag(t: pysrt.SubRipTime) -> str:
    return f"{t.hours:02d}_{t.minutes:02d}_{t.seconds:02d}_{int(t.milliseconds):03d}"

def agrupar_blocos(subs: List[pysrt.SubRipItem], min_w=20, max_w=30):
    blocos, txt, start = [], [], None
    for s in subs:
        words = s.text.replace("\n"," ").split()
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


def clean_prompt(raw: str) -> str:
    parts = re.split(r"\*{0,2}Prompt\*{0,2}:\s*", raw)
    body = parts[-1] if len(parts) > 1 else raw
    body = re.sub(r"^Here(?:'|’)s a [^:]+:\s*", "", body)
    body = body.replace("*", "").strip()
    return re.sub(r"\s+", " ", body)


def gerar_prompt(client_txt, texto: str) -> str:
    pedido = (
        #"Create a concise, vivid, image generation prompt, that represents "
        #"this scene, with no text overlay. "
        "Create an abstract and concise image generation prompt (only one option ready to go) with black background, that represents "
        "the principal words of this text (subject verb predicate): "
        #"The prompt must end with the quality parameters and explicitly "
        #f"Scene:\n{texto}\n\n"
        f"{texto}" + " aspect_ratio=16:9."
        #f"Quality parameters:\n{STYLE_SUFFIX}"
    )
    try:
        resp = client_txt.models.generate_content(
            model="gemini-2.0-flash",
            contents=pedido
        )
        raw = resp.candidates[0].content.parts[0].text or ""
        prompt = clean_prompt(raw)
        #prompt = pedido
        if prompt and not prompt.startswith(texto[:10]):
            return prompt
    except Exception:
        pass
    return f"{texto}, {STYLE_SUFFIX}"


def gerar_imagem(client_img, prompt: str, tries: int = 50) -> bytes | None:
    for _ in range(tries):
        try:
            resp = client_img.models.generate_content(
                model="gemini-2.0-flash-exp-image-generation",
                contents=[prompt],
                config=types.GenerateContentConfig(response_modalities=["TEXT", "IMAGE"])
            )
        except Exception:
            time.sleep(2.0)
            continue
        if resp and resp.candidates:
            cand0 = resp.candidates[0]
            if cand0.content and getattr(cand0.content, "parts", None):
                for part in cand0.content.parts:
                    if part.inline_data:
                        return part.inline_data.data
        time.sleep(1.2)
    return None

# ─── Streamlit UI ───────────────────────────────
st.set_page_config(page_title="SRT ▶︎ Gemini Imagens", layout="wide")
st.title("🎞️ SRT → Gemini Flash → Imagens Cinematográficas")

# Autenticação
api_key = st.secrets.get("GEMINI_API_KEY")
if not api_key:
    st.error("Configure GEMINI_API_KEY em Settings ▸ Secrets.")
    st.stop()
client_txt = genai.Client(api_key=api_key)
client_img = genai.Client(api_key=api_key, http_options=types.HttpOptions(api_version="v1alpha"))

# Controles
min_w = st.sidebar.number_input("Mín. palavras/bloco", 10, 30, 20)
max_w = st.sidebar.number_input("Máx. palavras/bloco", 20, 60, 30)

block_nums_str = st.sidebar.text_input("Blocos a (re)processar (índices, ex: 51,75):", "")
timestamps_str = st.sidebar.text_area("Timestamps a (re)processar (uma por linha, sem .png):", "")

# Converte filtros
selected_idxs = set()
if block_nums_str.strip():
    try:
        selected_idxs = {int(x) for x in re.split(r"[,\s]+", block_nums_str) if x.strip()}
    except ValueError:
        st.error("Formato inválido em 'Blocos'. Use números separados por vírgula.")
selected_ts = {line.strip() for line in timestamps_str.splitlines() if line.strip()}

uploaded = st.file_uploader("📂 Faça upload do .srt", type="srt")

# ─── Botão: Gerar Imagens ──────────────────────
if st.button("🚀 Gerar Imagens"):
    st.session_state["imgs"] = []
    st.session_state["blocos"] = []

    if not uploaded:
        st.warning("Envie um .srt primeiro.")
        st.stop()

    subs = pysrt.from_string(uploaded.getvalue().decode("utf-8"))
    blocos = agrupar_blocos(subs, min_w, max_w)
    st.session_state["blocos"] = blocos
    st.info(f"{len(blocos)} blocos total; aplicando filtros...")
    prog = st.progress(0.0)
    out_dir = Path("output_images"); out_dir.mkdir(exist_ok=True)

    for i, blk in enumerate(blocos, 1):
        key = f"{tag(blk['start'])}-{tag(blk['end'])}"
        if selected_idxs and i not in selected_idxs:
            continue
        if selected_ts and key not in selected_ts:
            continue

        prompt = gerar_prompt(client_txt, blk["text"])
        img_bytes = gerar_imagem(client_img, prompt)
        if img_bytes is None:
            st.warning(f"⚠️ Bloco {i} ({key}): sem imagem, pulado.")
            prog.progress(i/len(blocos))
            continue

        # Inclui número do bloco no nome do arquivo
        fname = f"{key}_B{i}.png"
        (out_dir/fname).write_bytes(img_bytes)
        st.session_state["imgs"].append({"name": fname, "bytes": img_bytes, "prompt": prompt})
        prog.progress(i/len(blocos))

    st.success("✔️ Processamento concluído!")

# ─── Botão: Reprocessar falhas ─────────────────
if st.session_state["blocos"] and st.button("🔄 Reprocessar falhas"):
    prog = st.progress(0.0)
    total = len(st.session_state["blocos"])
    for i, blk in enumerate(st.session_state["blocos"], 1):
        key = f"{tag(blk['start'])}-{tag(blk['end'])}"
        # Verifica se já existe com sufixo de bloco
        if any(item["name"] == f"{key}_B{i}.png" for item in st.session_state["imgs"]):
            continue
        if selected_idxs and i not in selected_idxs:
            continue
        if selected_ts and key not in selected_ts:
            continue

        prompt = gerar_prompt(client_txt, blk["text"])
        img_bytes = gerar_imagem(client_img, prompt)
        if img_bytes:
            fname = f"{key}_B{i}.png"
            st.session_state["imgs"].append({"name": fname, "bytes": img_bytes, "prompt": prompt})
        prog.progress(i/total)
    st.success("🔄 Reprocessamento concluído!")

# ─── Galeria + downloads ──────────────────────
if st.session_state["imgs"]:
    st.header("📸 Imagens Geradas")
    for idx, item in enumerate(st.session_state["imgs"]):
        st.image(item["bytes"], caption=item["name"], use_column_width=True)
        st.download_button(
            f"Baixar {item['name']}",
            item["bytes"],
            file_name=item["name"],
            mime="image/png",
            key=f"dl-{idx}-{item['name']}"
        )

    # ZIP on-the-fly
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for itm in st.session_state["imgs"]:
            zf.writestr(itm["name"], itm["bytes"])
    buf.seek(0)
    st.download_button(
        "⬇️ Baixar todas as imagens (.zip)", buf, "todas_as_imagens.zip", "application/zip"
    )

    # Prompts.txt
    txt = "\n\n".join(f"{itm['name']}: {itm['prompt']}" for itm in st.session_state["imgs"])
    st.download_button(
        "⬇️ Baixar Prompts (.txt)", txt, "prompts.txt", "text/plain"
    )
