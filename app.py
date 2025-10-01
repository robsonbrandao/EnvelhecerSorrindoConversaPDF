import os
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import streamlit as st
from dotenv import load_dotenv

# LangChain / loaders / vetores
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.vectorstores import Chroma

# OpenAI via langchain-openai (NÃO é langchain_community)
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

# -------------------------------------------------------------------
# Configurações básicas
# -------------------------------------------------------------------
st.set_page_config(page_title="Chat com PDFs 📚🤖", layout="centered")
st.title("🤖 Converse com seus PDFs")
st.markdown("Faça perguntas com base no conteúdo dos documentos da pasta `/fo`.")

# Diretórios
PDF_DIR = Path("fo")
CHROMA_DIR = Path("chroma_db")

# Carrega variáveis locais (.env) para rodar no PC; no Cloud use st.secrets
load_dotenv()
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))

if not OPENAI_API_KEY:
    st.error("Defina OPENAI_API_KEY em Settings → Secrets (Streamlit Cloud) ou .env local.")
    st.stop()

# -------------------------------------------------------------------
# Cache de recursos (evita reinstanciar a cada interação)
# -------------------------------------------------------------------
@st.cache_resource(show_spinner=True)
def get_embeddings():
    # Use o modelo novo de embedding por padrão
    return OpenAIEmbeddings(api_key=OPENAI_API_KEY, model="text-embedding-3-small")

@st.cache_resource(show_spinner=True)
def get_llm():
    # Chat model moderno. Ajuste o modelo se quiser.
    return ChatOpenAI(api_key=OPENAI_API_KEY, model="gpt-4o-mini", temperature=0)

@st.cache_resource(show_spinner="Lendo PDFs e construindo índice...")
def carregar_base_vetorial():
    # 1) Carrega documentos da pasta ./fo
    loader = PyPDFDirectoryLoader(str(PDF_DIR))
    documents = loader.load()

    # 2) Cria/atualiza a base vetorial persistente em ./chroma_db
    embeddings = get_embeddings()
    vectordb = Chroma.from_documents(
        documents,
        embedding=embeddings,
        persist_directory=str(CHROMA_DIR)
    )
    return vectordb

# -------------------------------------------------------------------
# Construção/Carregamento do índice
# -------------------------------------------------------------------
if not CHROMA_DIR.exists() or not any(CHROMA_DIR.iterdir()):
    st.info("Inicializando índice (ChromaDB) a partir dos PDFs...")
db = carregar_base_vetorial()
st.success("Base vetorial pronta!")

# -------------------------------------------------------------------
# UI de pergunta → resposta
# -------------------------------------------------------------------
pergunta = st.text_input("Digite sua pergunta:")

# Histórico
if "historico" not in st.session_state:
    st.session_state.historico = []

if pergunta:
    with st.spinner("Consultando os documentos..."):
        # 1) Recupera trechos similares
        docs = db.similarity_search(pergunta, k=3)

        # 2) Concatena contexto e pergunta para o LLM
        contexto = "\n\n".join(d.page_content for d in docs)
        prompt = f"""
Responda de forma objetiva com base APENAS no contexto abaixo.
Se a resposta não estiver no contexto, diga que não encontrou informação suficiente.

Contexto:
{contexto}

Pergunta:
{pergunta}
        """.strip()

        llm = get_llm()
        resposta = llm.invoke(prompt).content

        # 3) Salva no histórico
        st.session_state.historico.append({"pergunta": pergunta, "resposta": resposta})

# Exibir histórico
if st.session_state.historico:
    st.markdown("---")
    st.subheader("Histórico de perguntas e respostas")
    for i, item in enumerate(reversed(st.session_state.historico), 1):
        st.markdown(f"**🧠 Pergunta {len(st.session_state.historico) - i + 1}:** {item['pergunta']}")
        st.markdown(f"**📌 Resposta:** {item['resposta']}")
        st.markdown("---")
