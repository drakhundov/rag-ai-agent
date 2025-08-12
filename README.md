# PDF RAG AI Agent

This project utilizes Retrieval-Augmented Generation (RAG) to enable large language models (LLMs) to interact effectively with PDF documents. It includes OCR support to process scanned or corrupted PDFs.

# Features

- PDF ingestion and processing  
- OCR for scanned or illegible documents  
- Question answering based on PDF content  
- Chatbot integration for interactive use  

# Requirements

Python 3.9+

```bash
pip install -r requirements.txt
```
*Installs all required dependencies.*

# Usage

You need a HuggingFace API token to start. Get it from [the official website](https://huggingface.co/settings/tokens).
Add it to the ".env" file before starting.

```bash
chainlit run main.py
```