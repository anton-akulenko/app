
![Screenshot 2024-03-21 204751](https://github.com/anton-akulenko/app/assets/7303380/aaec3ea5-e48b-4ec8-a7c8-00aaf68b7fc3)

## Overview:

Simple web-based chat app, built using Flask and [Langchain](https://python.langchain.com/). The app backend follows the Retrieval Augmented Generation (RAG) framework.

Allows the user to provide a list of PDFs, and ask questions to a LLM (OpenAI GPT is implemented) that can be answered by these PDF documents.

User needs to provide their own OpenAI API key (.env).

## Instalation:

Just clone the repo and install the requirements using ```pip install -r requirements.txt```
You need to re-create .env file from the uploaded .env-example. 
Replace your OPENAI_API_KEY

## How to run locally:

Run ```python app.py``` in your terminal.

Upload the PDF documents that are relevant to your queries, and start chatting with the bot. 

## How it works:

The provided PDFs will be uploaded and properly split into chunks, and finally embedding vectors for each chunk will be generated using OpenAI service. These vectors are then indexed using FAISS, and can be quickly retrieved.

As the user interacts with the bot, relevant document chunks/snippets are retrieved and added to the memory, alongside the past few messages. These snippets and messages are part of the prompt sent to the LLM; this way, the model will have as context not just the latest message and retrieved snippet, but past ones as well.
