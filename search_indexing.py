import PyPDF2
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders.pdf import PyPDFium2Loader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

chunk_size = 500
chunk_overlap = 50

number_snippets_to_retrieve = 3


def download_and_index_pdf(filepaths: list[str], session) -> FAISS:
    def __update_metadata(pages, filepath):

        for page in pages:
            try:
                with open(filepath, "rb") as f:
                    pdf_reader = PyPDF2.PdfReader(f)
                    title = pdf_reader.metadata.title if '/title' in pdf_reader.metadata else ""
                    page.metadata['source'] = filepath
                    page.metadata['title'] = title
            except Exception as e:
                print(f"Error while processing PDF from {filepath}: {e}")

        return pages

    all_pages = []
    for filepath in filepaths:
        loader = PyPDFium2Loader(filepath)
        splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        pages = loader.load_and_split(splitter)
        pages = __update_metadata(pages, filepath)
        all_pages += pages

    faiss_index = FAISS.from_documents(all_pages, OpenAIEmbeddings())

    return faiss_index


def search_faiss_index(faiss_index: FAISS, query: str, top_k: int = number_snippets_to_retrieve) -> list:
    docs = faiss_index.similarity_search(query, k=top_k)

    return docs
