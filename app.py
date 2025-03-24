import streamlit as st
import pandas as pd
from supabase import create_client, Client
from pandasai import SmartDataframe
from pandasai.llm.openai import OpenAI

# Configurações iniciais
st.set_page_config(page_title="Chat com Dados de Malária", page_icon="🦟")
st.title("🤖 Chat com Dados de Malária")

# Conectar ao Supabase
url = st.secrets["SUPABASE_URL"]
key = st.secrets["SUPABASE_KEY"]
supabase: Client = create_client(url, key)

# OpenAI LLM
openai_api_key = st.secrets["OPENAI_API_KEY"]
llm = OpenAI(api_token=openai_api_key)

# Carregar dados do Supabase
@st.cache_data
def carregar_dados():
    response = supabase.table("malaria").select("*").execute()
    return pd.DataFrame(response.data)

df = carregar_dados()

# Mostrar amostra dos dados
with st.expander("👀 Ver dados brutos"):
    st.dataframe(df.head())

# Criar dataframe inteligente com pandasai
sdf = SmartDataframe(df, config={"llm": llm})

# Caixa de pergunta
st.markdown("### Faça uma pergunta sobre os dados 👇")
query = st.text_input("Exemplo: Quantos casos ocorreram em janeiro de 2024?")

# Responder com IA
if query:
    with st.spinner("Consultando os dados com IA..."):
        try:
            resposta = sdf.chat(query)
            st.success(resposta)
        except Exception as e:
            st.error(f"Erro ao processar a pergunta: {e}")
