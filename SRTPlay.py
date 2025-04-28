import streamlit as st
import google.generativeai as genai
from PIL import Image
import io
import os

# Carrega a chave de API do Streamlit Secrets
GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]

if not GEMINI_API_KEY:
    st.error("A chave de API do Google Gemini não foi encontrada nos Streamlit Secrets. Certifique-se de configurá-la na seção 'Secrets' do seu aplicativo Streamlit.")
    st.stop()

genai.configure(api_key=GEMINI_API_KEY)
model_name = 'gemini-2.0-flash-exp-image-generation'
model = genai.GenerativeModel(model_name)

st.title("Gerador de Imagens com Gemini API")
st.subheader(f"Usando o modelo: {model_name}")

prompt = st.text_input("Digite um prompt para gerar a imagem:")

if st.button("Gerar Imagem"):
    if prompt:
        with st.spinner("Gerando imagem..."):
            try:
                response = model.generate_content([prompt])

                if response.parts and response.parts[0].data and 'mime_type' in response.parts[0].data and 'data' in response.parts[0].data:
                    mime_type, base64_data = response.parts[0].data['mime_type'], response.parts[0].data['data']
                    image_bytes = io.BytesIO(base64_data)
                    image = Image.open(image_bytes)
                    st.image(image, caption=f"Imagem gerada para: '{prompt}'", use_column_width=True)
                else:
                    st.error("Não foi possível gerar a imagem. Verifique a resposta da API.")
                    st.write(response)  # Exibe a resposta completa para debugging
            except Exception as e:
                st.error(f"Ocorreu um erro: {e}")
    else:
        st.warning("Por favor, digite um prompt.")

st.sidebar.header("Configurações")
st.sidebar.markdown("Este aplicativo usa a API Gemini para gerar imagens a partir de prompts de texto.")
st.sidebar.markdown("Configure sua chave de API do Google Cloud na seção 'Secrets' do seu aplicativo Streamlit.")
st.sidebar.markdown("Para mais informações, consulte a [documentação da API Gemini](https://ai.google.dev/docs).")
