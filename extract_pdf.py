#!/usr/bin/env python3
try:
    import PyPDF2
    
    with open('Research_proposal (3) (1).pdf', 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        for page_num in range(len(reader.pages)):
            page = reader.pages[page_num]
            print(f"--- Page {page_num+1} ---")
            print(page.extract_text())
            print("\n")
except ImportError:
    print("PyPDF2 not installed. Trying alternative method.")
    try:
        import fitz  # PyMuPDF
        
        doc = fitz.open('Research_proposal (3) (1).pdf')
        for page_num in range(len(doc)):
            page = doc[page_num]
            print(f"--- Page {page_num+1} ---")
            print(page.get_text())
            print("\n")
    except ImportError:
        print("Neither PyPDF2 nor PyMuPDF is installed.")
        print("Please install one with: pip install PyPDF2 or pip install pymupdf")