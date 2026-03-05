"""
文件处理工具函数
"""
import os
import hashlib
from pathlib import Path
from typing import Optional, Tuple
import mimetypes


def get_file_hash(file_path: str) -> str:
    """
    计算文件的MD5哈希值
    """
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def get_file_type(file_path: str) -> str:
    """
    获取文件类型
    """
    ext = Path(file_path).suffix.lower()
    mime_type, _ = mimetypes.guess_type(file_path)
    
    if ext in ['.pdf']:
        return 'pdf'
    elif ext in ['.doc', '.docx']:
        return 'doc'
    elif ext in ['.xls', '.xlsx']:
        return 'xls'
    elif ext in ['.txt', '.text']:
        return 'txt'
    elif ext in ['.html', '.htm']:
        return 'html'
    else:
        return mime_type or 'unknown'


def extract_text_from_file(file_path: str) -> Optional[str]:
    """
    从文件中提取文本内容
    支持PDF、DOC、DOCX、TXT等格式
    """
    file_type = get_file_type(file_path)
    
    try:
        if file_type == 'pdf':
            import PyPDF2
            text = ""
            with open(file_path, 'rb') as f:
                pdf_reader = PyPDF2.PdfReader(f)
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
            return text
        
        elif file_type == 'doc':
            # .doc格式需要特殊处理，这里简化处理
            # 可以使用python-docx2txt或其他库
            return None
        
        elif file_path.lower().endswith('.docx'):
            try:
                from docx import Document
                doc = Document(file_path)
                text = "\n".join([para.text for para in doc.paragraphs])
                return text
            except ImportError:
                print("python-docx not installed, skipping .docx extraction")
                return None
        
        elif file_type == 'txt':
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                return f.read()
        
        elif file_type in ['html', 'htm']:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                from utils.text_processor import extract_text_from_html
                return extract_text_from_html(f.read())
        
        else:
            return None
    
    except Exception as e:
        print(f"Error extracting text from {file_path}: {e}")
        return None


def ensure_dir(dir_path: str):
    """
    确保目录存在
    """
    Path(dir_path).mkdir(parents=True, exist_ok=True)


def get_file_size(file_path: str) -> int:
    """
    获取文件大小（字节）
    """
    return os.path.getsize(file_path)

