import os
import random
import re
import string
from datetime import datetime

from dotenv import load_dotenv
from flask import Flask, render_template, request, session, redirect
from werkzeug.utils import secure_filename

from langchain_utils import initialize_chat_conversation
from search_indexing import download_and_index_pdf

load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_FLASK")
api_key = os.getenv("OPENAI_API_KEY")
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

search_number_messages = 3

uploaded_files = []


def prepare_session():
    session.permanent = True

    if 'conversation_memory' not in session:
        session['conversation_memory'] = None

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
    urls = []
    for file_name in session['files']:
        urls.append(os.path.join("uploads", secure_filename(file_name)))
    faiss_index = download_and_index_pdf(urls, session)

    if request.method == 'POST':
        query = request.form.get('question')
    else:
        query = request.args.get('query')

    response = ""
    if len(session.get('files')) == 0:
        response = "FAISS index not found. Please upload PDFs first."
    else:
        if query:

            session["messages"].append({"role": "user", "content": query})
            user_messages_history = [message['content'] for message in session.get('messages')[-search_number_messages:] if message['role'] == 'user']
            user_messages_history = '\n'.join(user_messages_history)

            conversation = initialize_chat_conversation(faiss_index)
            conversation_memory = conversation
            response = conversation_memory.predict(input=query, user_messages_history=user_messages_history)
            session.modified = True

            snippet_memory = conversation.memory.memories[1]

            for page_number, snippet in zip(snippet_memory.pages, snippet_memory.snippets):
                snippet = re.sub("<START_SNIPPET_PAGE_\d+>", '', snippet)
                snippet = re.sub("<END_SNIPPET_PAGE_\d+>", '', snippet)
                session["messages"].append(
                    {"role": "snippets", "content": f' \n>>> Snippet from page {page_number + 1} \n {snippet}'})

            session["messages"].append({"role": "assistant", "content": response})

    return render_template('ask_question.html', files=session['files'], session=reversed(session["messages"]))


if __name__ == '__main__':
    app.run(debug=False)
