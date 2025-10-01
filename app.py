import streamlit as st
import os
import openai
import chromadb
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from dotenv import load_dotenv
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import streamlit as st
from src.ingest import build_or_update_vectorstore

PDF_DIR = Path("fo")
CHROMA_DIR = Path("chroma_db")

if not CHROMA_DIR.exists() or not any(CHROMA_DIR.iterdir()):
    st.info("ChromaDB não encontrado. Construindo a partir dos PDFs...")
    vectordb = build_or_update_vectorstore(PDF_DIR, CHROMA_DIR)
else:
    st.success("ChromaDB já existe. Pronto para usar!")



from pathlib import Path
from src.ingest import build_or_update_vectorstore

CHROMA_DIR = Path("chroma_db")
PDF_DIR = Path("fo")

if not CHROMA_DIR.exists() or not any(CHROMA_DIR.iterdir()):
    st.info("Inicializando índice (ChromaDB) a partir dos PDFs...")
    build_or_update_vectorstore(pdf_dir=PDF_DIR, chroma_dir=CHROMA_DIR)


# Carrega variáveis de ambiente do .env
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Configuração da página
st.set_page_config(page_title="Chat com PDFs 📚🤖", layout="centered")
st.title("🤖 Converse com seus PDFs")
st.markdown("Faça perguntas com base no conteúdo dos documentos da pasta `/fo`.")

# Inicializar histórico na sessão
if "historico" not in st.session_state:
    st.session_state.historico = []

# Carregar a base vetorial (com cache para evitar recarregamento desnecessário)
@st.cache_resource(show_spinner="Indexando os PDFs, aguarde...") 
def carregar_base_vetorial():
    loader = PyPDFDirectoryLoader("./fo")
    documents = loader.load()
    embeddings = OpenAIEmbeddings()
    vectordb = Chroma.from_documents(documents, embedding=embeddings, persist_directory="./chroma_db")
    return vectordb

db = carregar_base_vetorial()

# Entrada do usuário
pergunta = st.text_input("Digite sua pergunta:")

if pergunta:
    with st.spinner("Consultando os documentos..."):
        docs = db.similarity_search(pergunta, k=3)
        llm = OpenAI(temperature=0)
        chain = load_qa_chain(llm, chain_type="stuff")
        resposta = chain.run(input_documents=docs, question=pergunta)

        # Salva no histórico
        st.session_state.historico.append({"pergunta": pergunta, "resposta": resposta})

# Exibir histórico como um chat
if st.session_state.historico:
    st.markdown("---")
    st.subheader("Histórico de perguntas e respostas")
    for i, item in enumerate(reversed(st.session_state.historico), 1):
        st.markdown(f"**🧠 Pergunta {len(st.session_state.historico) - i + 1}:** {item['pergunta']}")
        st.markdown(f"**📌 Resposta:** {item['resposta']}")
        st.markdown("---")
