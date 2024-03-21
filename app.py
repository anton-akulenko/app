import os
import tempfile
import random
import string
from datetime import datetime
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain_community.callbacks import get_openai_callback

from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
import openai

from langchain.vectorstores.faiss import FAISS
from flask import Flask, render_template, request, flash, session, redirect, url_for
from io import BytesIO

from langchain_utils import initialize_chat_conversation
from search_indexing import download_and_index_pdf
import re
from werkzeug.utils import secure_filename
from langchain.vectorstores import VectorStore
import pickle

load_dotenv()


app = Flask(__name__)
app.secret_key = os.getenv("SECRET_FLASK")
api_key = os.getenv("OPENAI_API_KEY")
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

search_number_messages = 1

uploaded_files = []

# def get_pdf_text(file):
#     text = None
#     if file:
#         pdf_reader = PdfReader(file)
#         text = ''.join(page.extract_text() for page in pdf_reader.pages)
#     return text


# def get_text_chunks(text):
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
#     chunks = text_splitter.split_text(text)
#     return chunks


# def get_vectorstore(text_chunks):
#     api_key = os.getenv("OPENAI_API_KEY")
#     embeddings = OpenAIEmbeddings(model="text-embedding-ada-002", openai_api_key=api_key)
#     vectorstore = FAISS.from_texts(text_chunks, embeddings)
#     return vectorstore


# def get_conversation_chain(vectorstore):
#     llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)
#     memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
#     conversation_chain = ConversationalRetrievalChain.from_llm(
#         llm=llm,
#         retriever=vectorstore.as_retriever(),
#         memory=memory
#     )
#     return conversation_chain


def prepare_session():
    session.permanent = True
    
    if 'conversation_memory' not in session:
        session['conversation_memory'] = None

    # Initialize chat history used by StreamLit (for display purposes)
    if "messages" not in session:
        session["messages"] = []

    if 'files' not in session:
        session['files'] = []

    if 'last_update' not in session:
        session['last_update'] = None

    if 'last_indexed' not in session:
        session['last_indexed'] = None


    if "faiss_index" not in session:
        session['faiss_index'] = ''.join(random.choices(string.ascii_uppercase + string.digits, k=20))

@app.route('/')
def index():
    prepare_session()
    return render_template('upload.html', files=session['files'])

@app.route('/clear', methods=['GET'])
def clear_session():
    session.clear()
    # TODO: clear /tmp folder
    return redirect('/')

@app.route('/upload', methods=['POST'])
def upload_file():
    urls = []
    prepare_session()
    
    pdf_file = request.files['pdf_to_upload']

    if pdf_file.filename != '':
        filename = secure_filename(pdf_file.filename)
        print(UPLOAD_FOLDER, filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        pdf_file.save(filepath)
        urls.append(filepath)

    if len(urls) > 0:
        if pdf_file.filename not in session['files']:
            session['files'].append(pdf_file.filename)
            session['last_update'] = datetime.now()
            session.modified = True

    return render_template('upload.html', files=session['files'])

@app.route('/ask_question', methods=['POST', 'GET'])
def ask_question():
    prepare_session()
    sn_list = []
    # if session['last_update'] != session['last_indexed']:
    urls = []
    for file_name in session['files']:
        urls.append(os.path.join("uploads", secure_filename(file_name)))
    faiss_index = download_and_index_pdf(urls, session)
        # session['last_indexed'] = session['last_update']

    if request.method == 'POST':
        query = request.form.get('question')
        # if query:
            # session["messages"].append({"role": "user", "content": query})
            # session.modified = True
    else:
        query = request.args.get('query')
    
    # conversation_memory = session.get('conversation_memory')
    # user_messages_history = [message['content'] for message in session.get('messages')[-search_number_messages:] if message['role'] == 'user']
    # user_messages_history = '\n '.join(user_messages_history)

    response = ""

    if len(session.get('files')) == 0:
        response = "FAISS index not found. Please upload PDFs first."
    else:
        if query:
            
            session["messages"].append({"role": "user", "content": query})

            # faiss_index = None




            # with open(os.path.join('tmp', session['faiss_index']), "rb") as f:
                
            #     local_vectorstore: VectorStore = pickle.load(f)





            # with open(os.path.join('tmp', session['faiss_index']), 'rb') as f:
            #     faiss_index = FAISS.deserialize_from_bytes(embeddings=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"), serialized=f.read())
            # conversation_memory = session['conversation_memory']
            
            user_messages_history = [message['content'] for message in session.get('messages')[-search_number_messages:] if message['role'] == 'user']
            user_messages_history = '\n'.join(user_messages_history)

            response = ""

            # if not faiss_index:
            #     response = "FAISS index not found. Please upload PDFs first."
            # else:
                # if not conversation_memory:
                    # faiss_index = FAISS.read_index("index_filename.index")
            conversation = initialize_chat_conversation(faiss_index)
                    # session['conversation_memory'] = conversation
            conversation_memory = conversation
                # else:
                #     conversation = session['conversation_memory']
                
            # print("------------")
            # print(type(conversation_memory))
            # print("------------")
            response = conversation_memory.predict(input=query, user_messages_history=user_messages_history)
            session.modified = True
                
            snippet_memory = conversation.memory.memories[1]
            
            for page_number, snippet in zip(snippet_memory.pages, snippet_memory.snippets):
                snippet = re.sub("<START_SNIPPET_PAGE_\d+>", '', snippet)
                snippet = re.sub("<END_SNIPPET_PAGE_\d+>", '', snippet)
                session["messages"].append({"role": "snippets", "content": f' >>> Snippet from page {page_number + 1} \n\n  + {snippet}'})


            # session["messages"].append({"role": "user", "content": query})
            session["messages"].append({"role": "assistant", "content": response})

            print(session["messages"])
            print("-------------------------------------------------------")
            print(session["messages"]["role" == "snippets"])
    return render_template('ask_question.html', files=session['files'], session=reversed(session["messages"]))

if __name__ == '__main__':
    app.run(debug=True)
