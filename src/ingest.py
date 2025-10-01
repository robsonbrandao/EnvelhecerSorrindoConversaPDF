# src/ingest.py
from pathlib import Path
import chromadb
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

def build_or_update_vectorstore(pdf_dir: Path, chroma_dir: Path):
    """
    Lê todos os PDFs em pdf_dir, gera embeddings e salva no Chroma persistido em chroma_dir.
    Se já existir um índice, ele será atualizado.
    """
    pdf_files = list(pdf_dir.glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(f"Nenhum PDF encontrado em {pdf_dir}")

    docs = []
    for pdf in pdf_files:
        loader = PyPDFLoader(str(pdf))
        docs.extend(loader.load())

    # quebra em chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_documents(docs)

    # embeddings
    embeddings = OpenAIEmbeddings()

    # cria / atualiza chroma
    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(chroma_dir)
    )

    vectordb.persist()
    print(f"Base Chroma atualizada com {len(chunks)} chunks a partir de {len(pdf_files)} PDFs.")
    return vectordb
