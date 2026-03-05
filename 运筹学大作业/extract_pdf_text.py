import pypdf
import os

def extract_text_from_pdf(pdf_path):
    text = ""
    try:
        reader = pypdf.PdfReader(pdf_path)
        for page in reader.pages:
            text += page.extract_text() + "\n"
    except Exception as e:
        return f"Error reading {pdf_path}: {e}"
    return text

pdf_files = ["实验1&2要求.pdf", "实验要求3.pdf"]
base_dir = "/Users/halo/Desktop/运筹学大作业"

for pdf_file in pdf_files:
    file_path = os.path.join(base_dir, pdf_file)
    print(f"--- Start of {pdf_file} ---")
    if os.path.exists(file_path):
        print(extract_text_from_pdf(file_path))
    else:
        print(f"File not found: {file_path}")
    print(f"--- End of {pdf_file} ---\n")
