"""
step2_keyword_extraction.py

This script extracts core keywords from a collection of research papers to identify
key research areas and themes. It implements the following keyword extraction
technique: Term Frequency-Inverse Document Frequency (TF-IDF)
"""

# --- Configuration ---
PDF_FOLDER = 'raw_data/articles'
NUM_KEYWORDS = 20                # Increase initial selection slightly to account for filtering
FINAL_NUM_KEYWORDS = 10          # Target number of keywords after filtering
OUTPUT_EXCEL_FILE = r'processed_data\article_keywords_tfidf_filtered.xlsx'

# Words to filter out (will be checked in lowercase)
WORDS_TO_FILTER = ['arxiv', 'ieee', 'proc'] 
# --- End Configuration ---

import os
import PyPDF2
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import pandas as pd

def extract_text_from_pdf(pdf_path):
    """Extracts text from a PDF file using PyPDF2."""
    text = ""
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            if reader.is_encrypted:
                 try:
                    reader.decrypt('')
                 except Exception as e:
                    print(f"Warning: Could not decrypt {os.path.basename(pdf_path)}. Skipping. Error: {e}")
                    return None

            num_pages = len(reader.pages)
            for page_num in range(num_pages):
                page = reader.pages[page_num]
                page_text = page.extract_text()
                if page_text:
                    page_text = re.sub(r'\s+', ' ', page_text).strip()
                    text += page_text + " "
    except FileNotFoundError:
        print(f"Error: File not found at {pdf_path}")
        return None
    except Exception as e:
        print(f"Error reading {os.path.basename(pdf_path)}: {e}")
        return text if text else None
    return text.strip() if text else None


def extract_keywords_tfidf(pdf_folder, num_keywords_initial=15, num_keywords_final=10, filter_words=None):
    """
    Extracts text from PDFs, computes TF-IDF keywords, and filters them.
    Assumes PDF filenames are the desired identifiers.
    """
    if filter_words is None:
        filter_words = []

    if not os.path.isdir(pdf_folder):
        print(f"Error: Folder not found at {pdf_folder}")
        return None

    pdf_files = [f for f in os.listdir(pdf_folder) if f.lower().endswith('.pdf')]
    if not pdf_files:
        print(f"No PDF files found in {pdf_folder}")
        return None

    print(f"Found {len(pdf_files)} PDF files. Starting text extraction...")
    documents = []
    processed_filenames = []
    failed_files = []

    for i, filename in enumerate(pdf_files):
        pdf_path = os.path.join(pdf_folder, filename)
        print(f"Processing file {i+1}/{len(pdf_files)}: {filename}") # Uses the actual filename
        extracted_text = extract_text_from_pdf(pdf_path)
        if extracted_text:
            documents.append(extracted_text)
            processed_filenames.append(filename)
        else:
            print(f"Warning: Could not extract text from {filename}. Skipping.")
            failed_files.append(filename)

    if not documents:
        print("Error: No text could be extracted from any PDF files.")
        return None

    print(f"\nSuccessfully extracted text from {len(documents)} out of {len(pdf_files)} files.")
    if failed_files: print(f"Failed to extract text from: {', '.join(failed_files)}")

    print("\nCalculating TF-IDF scores...")
    vectorizer = TfidfVectorizer(stop_words='english', max_df=0.85, min_df=2,
                                 lowercase=True, ngram_range=(1, 1),
                                 use_idf=True, smooth_idf=True)

    try:
        tfidf_matrix = vectorizer.fit_transform(documents)
    except ValueError as e:
        print(f"\nError during TF-IDF calculation: {e}")
        return None

    feature_names = vectorizer.get_feature_names_out()
    print("Extracting and filtering top keywords for each document...")
    keywords_per_file = {}

    for i in range(tfidf_matrix.shape[0]):
        doc_vector = tfidf_matrix[i, :]
        term_scores = []
        row_data = doc_vector.toarray().flatten()
        non_zero_indices = row_data.nonzero()[0]

        for term_index in non_zero_indices:
            score = row_data[term_index]
            term = feature_names[term_index]
            term_scores.append((score, term))

        sorted_term_scores = sorted(term_scores, key=lambda item: item[0], reverse=True)
        top_keywords_initial = [term for score, term in sorted_term_scores[:num_keywords_initial]]

        # --- Filtering Step ---
        filtered_keywords = []
        for kw in top_keywords_initial:
            contains_digit = any(char.isdigit() for char in kw)
            is_filtered_word = kw.lower() in filter_words
            if not contains_digit and not is_filtered_word:
                filtered_keywords.append(kw)
        final_keywords = filtered_keywords[:num_keywords_final]
        # --- End Filtering Step ---

        current_filename = processed_filenames[i]
        keywords_per_file[current_filename] = final_keywords # Store keywords against the actual filename

    print("\nKeyword extraction and filtering complete.")
    return keywords_per_file

# --- Main execution ---
if __name__ == "__main__":
    pdf_directory = PDF_FOLDER
    output_file = OUTPUT_EXCEL_FILE
    filter_set = set(w.lower() for w in WORDS_TO_FILTER)

    keyword_results = extract_keywords_tfidf(
        pdf_directory,
        num_keywords_initial=NUM_KEYWORDS,
        num_keywords_final=FINAL_NUM_KEYWORDS,
        filter_words=filter_set
    )

    if keyword_results:
        print(f"\nPreparing results for Excel export...")
        output_data = []
        for filename, keywords in keyword_results.items():
            # Use filename (without extension) as the identifier
            article_name = os.path.splitext(filename)[0]
            keywords_str = ', '.join(keywords)
            # Use the actual filename (without .pdf) as the identifier
            output_data.append({'Article Name': article_name, 'Keywords': keywords_str})

        df = pd.DataFrame(output_data)

        try:
            df.to_excel(output_file, index=False, engine='openpyxl')
            print(f"\nSuccessfully saved filtered keywords to: {output_file}")
        except ImportError:
             print("\nError: 'openpyxl' library needed. Install using: pip install openpyxl")
        except Exception as e:
            print(f"\nError saving Excel file: {e}")

        print("\n--- Top Filtered Keywords per Article (TF-IDF) ---")
        # Print using the article name derived from the filename
        for item in output_data:
             print(f"\n📄 {item['Article Name']}:")
             print(f"   Keywords: {item['Keywords']}")
    else:
        print("\nNo keyword results generated.")