import streamlit as st
import google.generativeai as genai
from PIL import Image
from io import BytesIO
 

# Carrega a chave de API do Streamlit Secrets
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY")
 

if not GEMINI_API_KEY:
 st.error(
 "A chave de API do Google Gemini não foi encontrada nos Streamlit Secrets. Certifique-se de configurá-la na seção 'Secrets' do seu aplicativo Streamlit."
 )
 st.stop()
 
genai.configure(api_key=GEMINI_API_KEY)
 

st.title("Geração de Imagem com Gemini API")
 

prompt = st.text_area(
 "Digite um prompt para gerar a imagem:",
 value="A pig with wings and a top hat flying over a happy futuristic scifi city with lots of greenery.",
)
 

if st.button("Gerar Imagem"):
 if prompt:
  with st.spinner("Gerando imagem..."):
   try:
    model = genai.GenerativeModel(
     model_name="gemini-2.0-flash-exp-image-generation"
    )
    response = model.generate_content(
     contents=prompt
    )
    if response.parts:
     for part in response.parts:
      if hasattr(part, "data"):
       image = Image.open(BytesIO((part.data)))
       st.image(image, caption=prompt, use_column_width=True)
      else:
       st.warning(
        "A resposta não continha dados de imagem. Verifique o prompt e a resposta da API."
       )
       st.write(response)
      else:
       st.warning(
        "A resposta não continha partes válidas. Verifique o prompt e a resposta da API."
       )
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
 
