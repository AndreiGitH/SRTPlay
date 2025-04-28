import streamlit as st
import google.genai as genai
from google.genai import types
from PIL import Image
from io import BytesIO

# Carrega API Key
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    st.error("Chave de API não encontrada.")
    st.stop()

# Cria o client corretamente
client = genai.GenerativeServiceClient(
    api_key=GEMINI_API_KEY,
    transport="rest",
    http_options=types.HttpOptions(api_version="v1alpha"),
)

st.title("Geração de Imagem com Gemini API")

prompt = st.text_area(
    "Digite o prompt para gerar a imagem:",
    value="A pig with wings and a top hat flying over a happy futuristic scifi city with lots of greenery.",
)

if st.button("Gerar Imagem"):
    if prompt:
        with st.spinner("Gerando imagem..."):
            try:
                response = client.generate_content(
                    model="models/gemini-1.5-flash",
                    prompt=prompt,
                    generation_config={"response_mime_type": "image/png"},
                )

                if response.candidates and response.candidates[0].content.parts:
                    for part in response.candidates[0].content.parts:
                        if part.inline_data is not None:
                            image = Image.open(BytesIO(part.inline_data.data))
                            st.image(image, caption=prompt, use_column_width=True)
                        elif part.text is not None:
                            st.info("Texto gerado:")
                            st.write(part.text)
                else:
                    st.warning("Resposta vazia.")
                    st.write(response)

            except Exception as e:
                st.error(f"Ocorreu um erro: {e}")
    else:
        st.warning("Digite um prompt!")

st.sidebar.header("Configurações")
st.sidebar.markdown("Usa a nova API Gemini via google-genai.")
