import streamlit as st
import google.genai as genai
from PIL import Image
from io import BytesIO

# Carrega a chave de API do Streamlit Secrets
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    st.error(
        "A chave de API do Google Gemini não foi encontrada nos Streamlit Secrets. "
        "Certifique-se de configurá-la na seção 'Secrets' do seu aplicativo Streamlit."
    )
    st.stop()

st.title("Geração de Imagem com Gemini API")

prompt = st.text_area(
    "Digite um prompt para gerar a imagem:",
    value="A pig with wings and a top hat flying over a happy futuristic scifi city with lots of greenery.",
)

if st.button("Gerar Imagem"):
    if prompt:
        with st.spinner("Gerando imagem..."):
            try:
                # Chamada direta com API KEY
                response = genai.generate_content(
                    model="gemini-1.5-flash",  # ou outro modelo disponível
                    prompt=prompt,
                    api_key=GEMINI_API_KEY,
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
                    st.warning("A resposta não continha partes válidas. Verifique o prompt e a resposta da API.")
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
    "Certifique-se de ter configurado sua chave de API do Google Cloud na seção 'Secrets' do seu aplicativo Streamlit."
)
st.sidebar.markdown(
    "Para mais informações, consulte a [documentação da API Gemini](https://ai.google.dev/docs)."
)
