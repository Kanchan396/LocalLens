\# LocalLens



A fully local, private AI second brain.  

Turn your personal notes, PDFs, and documents into a searchable knowledge base — completely offline.



\## Features



\- 100% local (no data leaves your machine)

\- Works with `.txt`, `.md`, `.pdf`, `.docx`

\- Semantic search + local LLM chat

\- Folder watcher for automatic re-indexing

\- Free and open source



\## Requirements



\- Python 3.10+

\- \[Ollama](https://ollama.com) installed

\- \~2 GB free disk space for a small model

\- 8–16 GB RAM recommended



\## Quick Start



1\. Clone the repository

2\. Create and activate a virtual environment

3\. Install dependencies:

	pip install -r requirements.txt

4\. Install Ollama and pull a model:

	ollama pull llama3.2:3b

5\. Put your documents in the documents/ folder

6\. Build the index:

	 src/indexer.py

7\. Start chatting: 
	
	src/chat.py

