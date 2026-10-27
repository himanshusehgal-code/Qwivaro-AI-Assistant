# Qwivaro AI Assistant — Free/Open-Source Starter

A document-grounded ERP assistant built with Python, Qwen, Gradio, Sentence Transformers and FAISS.

## Features

- Chat interface for ERP manuals.
- Admin tab to add PDF, DOCX and TXT documents.
- Semantic retrieval with source filenames.
- Delete a document and rebuild the index.
- Google Drive persistence for source files, FAISS index, chunks and manifest.
- Local-storage fallback for development.

> This is a starter project, not a secured production ERP application. Add authentication/access controls before publishing the Admin tab publicly. Do not upload confidential company documents until you have reviewed access and storage permissions.

## 1. Run in Google Colab

1. Open a new Colab notebook.
2. Upload/extract this project ZIP, or copy `app.py` and `requirements.txt` into the notebook environment.
3. Install dependencies:

```python
!pip install -r requirements.txt
```

4. For Google Drive persistence in Colab, the simplest development option is to mount Drive and set `DATA_DIR` to a folder in your Drive **before** importing/running the app. Example:

```python
from google.colab import drive
drive.mount('/content/drive')
import os
os.environ["DATA_DIR"] = "/content/drive/MyDrive/QwivaroAI/data"
```

5. Run:

```python
!python app.py
```

Colab runtimes are temporary. Mounting Drive preserves files, but does not make Colab a reliable permanent public server. If you use the app in Colab, access the Gradio interface through the URL printed by Gradio; this starter defaults to `share=False`, so for temporary Colab access you can change the launch line to `share=True`. A temporary share URL is not production hosting.

## 2. Google Drive persistence for hosted deployment

For deployment outside Colab, this project uses the Google Drive API with a Google Cloud service account. This avoids asking the end user to upload all old documents after a restart.

1. Create a Google Cloud project and enable **Google Drive API**.
2. Create a service account and generate a JSON key.
3. Create a folder in Google Drive for the knowledge base.
4. Share that folder with the service account's `client_email` as **Editor**.
5. Copy the folder ID from its URL.
6. In your hosting provider's secret/environment-variable settings, set:
   - `GOOGLE_DRIVE_FOLDER_ID` = the Drive folder ID
   - `GOOGLE_SERVICE_ACCOUNT_JSON` = the full contents of the service-account JSON key as one secret value
7. Deploy `app.py` with the requirements in `requirements.txt`.

The app stores originals in Drive with names prefixed `doc__` and keeps `qwivaro_faiss.index`, `qwivaro_chunks.json`, and `qwivaro_manifest.json` in the same folder. Do not commit the service-account JSON key to GitHub/Bitbucket.

**Google Drive API note:** Drive storage itself has no per-request API charge for ordinary use, but Google quotas and account storage limits apply. Google Cloud may ask you to configure billing for some project services even though this app does not call a paid LLM API.

## 3. Deploy on Hugging Face Spaces (free-tier availability/limits apply)

1. Create a Gradio Space at https://huggingface.co/new-space and choose Gradio.
2. Upload `app.py` and `requirements.txt` (and this README if desired).
3. In Space Settings → Variables and secrets, add `GOOGLE_DRIVE_FOLDER_ID` and secret `GOOGLE_SERVICE_ACCOUNT_JSON`.
4. Space will install requirements and launch the Gradio app.

This model is loaded on the host at runtime. The `Qwen/Qwen2.5-1.5B-Instruct` model may be slow or may exceed memory limits on free CPU hardware. If it fails, use a smaller compatible model or a host with more RAM/GPU. Free hardware and storage quotas can change. Keep the Admin tab private or add authentication before using it with real documents.

## 4. Model settings

Defaults:
- LLM: `Qwen/Qwen2.5-1.5B-Instruct`
- Embeddings: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`

You can override them with `MODEL_NAME` and `EMBEDDING_MODEL` environment variables, but choose models compatible with the available memory.

## 5. How adding/deleting documents works

- Upload: the original file is saved; the app extracts text and rebuilds the FAISS index using all current documents; the new index and metadata are synced to Drive.
- Duplicate filename: skipped to avoid accidental replacement. Delete the old file before uploading a replacement.
- Delete: removes the original from local storage and Drive, then rebuilds and saves the index.
- Startup: the app downloads saved index/metadata and source documents from Drive.

## 6. Important limitations / security

- The Admin tab is not password-protected in this starter. Do not deploy it publicly for real company use until you add authentication or restrict access.
- The app retrieves document excerpts; it does not access live SQL Server or Google Sheets business data.
- Scanned PDFs may need OCR; this version extracts selectable PDF text only.
- FAISS search is in-memory during each running process. Drive files are the persistent copy.
- Google Drive file permissions are controlled by the account/folder owner and service account sharing.
- Do not use this starter for sensitive data without reviewing security, retention, backups and access control.
