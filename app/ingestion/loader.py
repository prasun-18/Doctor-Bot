import os
import pandas as pd
from PyPDF2 import PdfReader
from docx import Document

def load_pdf(file):
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

def load_docx(file):
    doc = Document(file)
    return "\n".join([para.text for para in doc.paragraphs])

def load_csv(file):
    df = pd.read_csv(file)
    return df.to_string()

def load_txt(file):
    return file.read().decode("utf-8")

def load_document(file):
    filename = file.name.lower()

    if filename.endswith(".pdf"):
        return load_pdf(file)
    elif filename.endswith(".docx"):
        return load_docx(file)
    elif filename.endswith(".csv"):
        return load_csv(file)
    elif filename.endswith(".txt"):
        return load_txt(file)
    else:
        raise ValueError("Unsupported file format")