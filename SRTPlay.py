import streamlit as st
import google.genai as genai
from google.genai import types
from PIL import Image
from io import BytesIO

# Carrega a chave de API do Streamlit Secrets
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    st.error("A chave de API do Google Gemini não foi encontrada nos Streamlit Secrets.")
    st.stop()

# Cria o client corretamente
client = genai.Client(
    api_key=GEMINI_API_KEY,
    http_options=types.HttpOptions(api_version='v1alpha')
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
                response = client.models.generate_content(
                    model="gemini-2.0-flash-exp-image-generation",
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_modalities=["TEXT", "IMAGE"]
                    )
                )

                if response.candidates and response.candidates[0].content.parts:
                    for part in response.candidates[0].content.parts:
                        if part.text is not None:
                            st.info("Texto gerado:")
                            st.write(part.text)
                        elif part.inline_data is not None:
                            image = Image.open(BytesIO(part.inline_data.data))
                            st.image(image, caption=prompt, use_column_width=True)
                else:
                    st.warning("Resposta vazia.")
                    st.write(response)

            except Exception as e:
                st.error(f"Ocorreu um erro: {e}")
    else:
        st.warning("Por favor, digite um prompt.")

st.sidebar.header("Configurações")
st.sidebar.markdown(
    "Este aplicativo usa a API Gemini para gerar imagens a partir de prompts de texto."
)
st.sidebar.markdown(
    "Certifique-se de ter configurado sua chave de API na seção 'Secrets' do Streamlit."
)
st.sidebar.markdown(
    "Para mais informações, consulte a [documentação da API Gemini](https://ai.google.dev/docs)."
)
