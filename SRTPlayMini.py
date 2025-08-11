#!/usr/bin/env python3
"""
Streamlit · SRT ▶︎ Gemini (prompt) ▶︎ Replicate (imagem)
-----------------------------------------------------
• Agrupa legendas em blocos de 20–30 palavras.
• Gera prompt cinematográfico com Gemini 2.0 Flash (texto).
• Gera imagem via Replicate API (minimax/image-01).
• Permite (re)processar só blocos específicos por índice ou timestamp.
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

import time

# ─── Configurações ─────────────────────────────
STYLE_SUFFIX = (
    #"D.C. Comics, black background, ancient Middle-East setting, no text overlay."
    #"Cinematic and Photorealistic, Photography high detailed, 4k, no text overlay."
    "Cinematic and Photorealistic, Photography high detailed, 4k, Chiaroscuro, no text overlay."
    #"dark gothic atmosphere, dramatic shadows, deep reds and browns, cinematic high contrast, 4K detail, photorealistic, photography"
    #"high detailed, no text overlay." 
)
# ─── session_state ─────────────────────────────
if "imgs" not in st.session_state:
    st.session_state["imgs"] = []  # [{"name","bytes","prompt"}]
if "blocos" not in st.session_state:
    st.session_state["blocos"] = []  # guarda blocos para reprocessar

# ─── Helpers ────────────────────────────────────
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
            blocos.append({"start": start, "end": s.end, "text": " ".join(txt[:max_w])})
            txt, start = [], None
    if txt:
        blocos.append({"start": start, "end": subs[-1].end, "text": " ".join(txt)})
    return blocos

def clean_prompt(raw: str) -> str:
    parts = re.split(r"\*{0,2}Prompt\*{0,2}:\s*", raw)
    body = parts[-1] if len(parts)>1 else raw
    body = re.sub(r"^Here(?:'|’)s a [^:]+:\s*","", body)
    body = body.replace("*","").strip()
    return re.sub(r"\s+", " ", body)

def gerar_prompt(client_txt, texto: str) -> str:
    pedido = (
        "Create a complete, creative, image generation prompt that represents "
        #"this biblical scene. Always bring a biblical setting, an environment of the time. The prompt must end with the quality parameters. "
        #"this biblical scene, with a beautiful ancient Middle Eastern setting. Capture the character emotion. The prompt must end with the quality parameters." # and only one part of image in blue, red ou yellow color.
        #"This biblical scene, set against a beautiful ancient Middle Eastern backdrop. Capture the emotion of the character or simply the beauty of the historical setting. The prompt should end with the quality parameters."
        #This biblical scene (biblical times), sharpness, set against a beautiful ancient Middle Eastern backdrop. If there is a man in the scene, he should be: 35 years old (has a beard and mustache). If there is a woman in the scene, she should be: 30 years old and is very beautiful and has black hair. The prompt should end with the quality parameters."
        "This scene with no text, no hands, no chairs, no sofa, no armchair, no tree, in the context for seniors, focus on food and health. The prompt should end with the quality parameters."
        #"This scene with no text. The prompt should end with the quality parameters."
        f"\n\nScene:\n{texto}\n\nQuality parameters:\n{STYLE_SUFFIX}"
    )
    try:
        time.sleep(2) # espera 3 segundos RPM de 10 para o modelo 2.5 flash e 15 para o 2.0 flash
        #resp = client_txt.models.generate_content(model="gemini-2.5-flash-preview-05-2", contents=pedido) # gemini-2.0-flash
        resp = client_txt.models.generate_content(model="gemini-2.5-flash-lite-preview-06-17", contents=pedido)
        raw = resp.candidates[0].content.parts[0].text or ""
        prompt = clean_prompt(raw)
        if prompt and not prompt.startswith(texto[:10]):
            return prompt
    except Exception:
        pass
    return f"{texto}, {STYLE_SUFFIX}"

def gerar_imagem_replicate(prompt: str, aspect_ratio: str="16:9") -> bytes:
    #output = replicate.run("prunaai/flux.1-dev:970a966e3a5d8aa9a4bf13d395cf49c975dc4726e359f982fb833f9b100f75d5", input={"seed": -1, "prompt": prompt, "guidance": 3.5, "image_size": 1024, "speed_mode": "Juiced 🔥 (default)", "aspect_ratio": aspect_ratio, "output_format": "png", "output_quality": 100, "num_inference_steps": 30})
    output = replicate.run("black-forest-labs/flux-schnell", input={"prompt": prompt, "aspect_ratio": aspect_ratio, "output_format": "png", "output_quality": 100, "seed":41270, "disable_safety_checker": True, "go_fast": False}) 
    #output = replicate.run("minimax/image-01", input={"prompt": prompt, "aspect_ratio": aspect_ratio})
    return output[0].read()
    #return output.read()

# ─── Streamlit UI ───────────────────────────────
st.set_page_config(page_title="SRT ▶︎ Replicate Imagens", layout="wide")
st.title("🎞️ SRT → Gemini (prompt) → Replicate (imagem)")

# Autenticação
rep_token = st.secrets.get("REPLICATE_API_TOKEN")
if not rep_token:
    st.error("Defina REPLICATE_API_TOKEN em Settings → Secrets.")
    st.stop()
os.environ["REPLICATE_API_TOKEN"] = rep_token
api_key = st.secrets.get("GEMINI_API_KEY2","")
client_txt = genai.Client(api_key=api_key) if api_key else None

# Controles de bloco
min_w = st.sidebar.number_input("Mín. palavras/bloco", 10, 100, 20)
max_w = st.sidebar.number_input("Máx. palavras/bloco", 20, 150, 30)
block_nums_str = st.sidebar.text_input("Blocos a (re)processar (índices, ex: 51,75):", "")
timestamps_str = st.sidebar.text_area("Timestamps a (re)processar (uma por linha, sem .png):", "")
selected_idxs = set()
if block_nums_str.strip():
    try:
        selected_idxs = {int(x) for x in re.split(r"[,\s]+", block_nums_str) if x.strip()}
    except ValueError:
        st.error("Formato inválido em 'Blocos'. Use números separados por vírgula.")
selected_ts = {line.strip() for line in timestamps_str.splitlines() if line.strip()}
uploaded = st.file_uploader("📂 Envie seu arquivo .srt", type="srt")

# ─── Botão: gerar tudo ───────────────────────────
if st.button("🚀 Gerar Imagens"):
    st.session_state["imgs"] = []
    st.session_state["blocos"] = []
    if not uploaded:
        st.warning("Faça upload do .srt primeiro.")
        st.stop()
    subs = pysrt.from_string(uploaded.getvalue().decode("utf-8"))
    blocos = agrupar_blocos(subs, min_w, max_w)
    st.session_state["blocos"] = blocos
    st.info(f"{len(blocos)} blocos serão processados.")
    prog = st.progress(0.0)
    out_dir = Path("output_images"); out_dir.mkdir(exist_ok=True)

    for i, blk in enumerate(blocos, 1):
        key = f"{tag(blk['start'])}-{tag(blk['end'])}"
        if selected_idxs and i not in selected_idxs: continue
        if selected_ts and key not in selected_ts: continue
        prompt = gerar_prompt(client_txt, blk["text"]) if client_txt else blk["text"]
        try:
            img_bytes = gerar_imagem_replicate(prompt)
        except Exception as e:
            st.warning(f"Bloco {i} ({key}) falhou: {e}")
            prog.progress(i/len(blocos)); continue

        name = f"{key}_B{i}.png"
        (out_dir/name).write_bytes(img_bytes)
        st.session_state["imgs"].append({"name":name,"bytes":img_bytes,"prompt":prompt})
        prog.progress(i/len(blocos))
    st.success("✔️ Imagens geradas!")

# ─── Botão: reprocessar falhas ───────────────────
if st.session_state["blocos"] and st.button("🔄 Reprocessar blocos falhos"):
    prog = st.progress(0.0)
    for i, blk in enumerate(st.session_state["blocos"], 1):
        key = f"{tag(blk['start'])}-{tag(blk['end'])}"
        if any(item["name"] == f"{key}_B{i}.png" for item in st.session_state["imgs"]): continue
        if selected_idxs and i not in selected_idxs: continue
        if selected_ts and key not in selected_ts: continue
        prompt = gerar_prompt(client_txt, blk["text"]) if client_txt else blk["text"]
        try:
            img_bytes = gerar_imagem_replicate(prompt)
        except Exception as e:
            st.warning(f"Retry {i} ({key}) falhou: {e}"); prog.progress(i/len(st.session_state["blocos"])); continue

        name = f"{key}_B{i}.png"
        st.session_state["imgs"].append({"name":name,"bytes":img_bytes,"prompt":prompt})
        prog.progress(i/len(st.session_state["blocos"]))
    st.success("🔄 Reprocessamento concluído!")

# ─── Galeria + downloads ────────────────────────
if st.session_state["imgs"]:
    st.header("📸 Galeria de Imagens")
    for idx,item in enumerate(st.session_state["imgs"]):
        st.image(item["bytes"], caption=item["name"], use_column_width=True)
        st.download_button(f"Baixar {item['name']}", item["bytes"], file_name=item["name"], mime="image/png", key=f"img-{idx}")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf,"w",zipfile.ZIP_DEFLATED) as zf:
        for item in st.session_state["imgs"]:
            zf.writestr(item["name"], item["bytes"])
    buf.seek(0)
    st.download_button("⬇️ Baixar ZIP (.zip)", buf, "output_images.zip", "application/zip")

    txt = "\n\n".join(f"{itm['name']}: {itm['prompt']}" for itm in st.session_state["imgs"])
    st.download_button("⬇️ Baixar Prompts (.txt)", txt, "prompts.txt", "text/plain")
