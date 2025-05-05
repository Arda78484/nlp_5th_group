import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
from langdetect import detect, LangDetectException
from rake_nltk import Rake

# Name of the file and sheet
EXCEL_FILE_PATH = r'raw_data\nlp_hw1_raw.xlsx' 
SHEET_NAME = 'Sayfa1'

# Initilize constants for column names
COL_PAPER_NAME = 'Paper Name'
COL_ABSTRACT = 'Abstract'
COL_TECHNIQUES = 'Techniques'

# Download nltk components if not found
try:
    nltk.data.find('corpora/stopwords')
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('tokenizers/punkt_tab')
    print("NLTK data found.")
except LookupError:
    print("Downloading NLTK data (stopwords, punkt)...")
    try:
        nltk.download('stopwords', quiet=True)
        nltk.download('punkt', quiet=True)
        nltk.download('punkt_tab', quiet=True)
        print("NLTK data downloaded.")
    except Exception as e:
        print(f"Error downloading NLTK data: {e}")
        print("Please check your internet connection and try again.")

# Detecting language function
def detect_language(text):
    if pd.isna(text) or not isinstance(text, str):
        return 'unknown' # If there is no text return string 'unknown'
    try:
        return detect(text)
    except LangDetectException:
        return 'error'
    except Exception as e:
        print(f"Unexpected error during language detection: {e}")
        return 'error'

LANG_MAP_NLTK = {
    'en': 'english',
    'fr': 'french',
    'es': 'spanish',
    'de': 'german',
    'it': 'italian',
    'pt': 'portuguese',
    'nl': 'dutch',
}

def clean_text(text, language_code='en'):
    if pd.isna(text) or not isinstance(text, str):
        return "" # Return empty string for non string inputs!

    # Turn all text to lowercase
    text = text.lower()

    # Remove punctuation with a translation table
    translator = str.maketrans('', '', string.punctuation)
    text = text.translate(translator)

    # Tokenize text
    tokens = word_tokenize(text)

    # Based on language filter stopwords
    lang_name = LANG_MAP_NLTK.get(language_code)
    if lang_name:
        try:
            stop_words = set(stopwords.words(lang_name))
            filtered_tokens = [word for word in tokens if word not in stop_words and word.isalnum()]
        except OSError:
            print(f"Warning: Stopwords for language '{lang_name}' ({language_code}) not found. Skipping stopword removal.")
            filtered_tokens = [word for word in tokens if word.isalnum()]
    else:
        if language_code not in ['unknown', 'error']:
             print(f"Warning: Language code '{language_code}' not mapped for stopword removal. Skipping.")
        filtered_tokens = [word for word in tokens if word.isalnum()]

    return " ".join(filtered_tokens)

LANG_MAP_RAKE = {
    'en': 'english',
    'fr': 'french',
    'es': 'spanish',
    'de': 'german',
    'it': 'italian',
    'pt': 'portuguese',
    'nl': 'dutch',
    'ru': 'russian'
}

def extract_keywords_rake(text, language_code='en'):
    """Extracts keywords using RAKE algorithm."""
    if pd.isna(text) or not isinstance(text, str) or len(text.strip()) < 10:
        return []

    rake_language = LANG_MAP_RAKE.get(language_code, 'english') # Default to English if unknown/unsupported
    try:
        r = Rake(language=rake_language)
        # Extract keywords from the *original* (but lowercased) text for better context
        r.extract_keywords_from_text(text.lower())
        # Get ranked phrases (keywords). Returns list of strings.
        keywords = r.get_ranked_phrases() # You can also use get_ranked_phrases_with_scores()
        return keywords
    except Exception as e:
        print(f"Error during keyword extraction for language {language_code}: {e}")
        return []


print(f"Reading Excel file: {EXCEL_FILE_PATH}...")
try:
    # Read the excel file
    df = pd.read_excel(EXCEL_FILE_PATH, sheet_name=SHEET_NAME)
    print(f"Successfully read {len(df)} rows.")
except FileNotFoundError:
    print(f"Error: File not found at {EXCEL_FILE_PATH}")
    exit()
except Exception as e:
    print(f"Error reading Excel file: {e}")
    exit()

required_cols = [COL_PAPER_NAME, COL_ABSTRACT, COL_TECHNIQUES]
missing_cols = [col for col in required_cols if col not in df.columns]
if missing_cols:
    print(f"Error: Missing required columns in the Excel file: {missing_cols}")
    print(f"Available columns are: {list(df.columns)}")
    exit()

print("Processing articles...")

print("- Detecting language...")
df['language_code'] = df[COL_ABSTRACT].fillna('').apply(detect_language)

print("- Extracting keywords...")
df['keywords'] = df.apply(
    lambda row: extract_keywords_rake(row[COL_ABSTRACT], row['language_code']),
    axis=1
)

print("- Cleaning text (title and abstract)...")
df['cleaned_title'] = df.apply(
    lambda row: clean_text(row[COL_PAPER_NAME], row['language_code']),
    axis=1
)
df['cleaned_abstract'] = df.apply(
    lambda row: clean_text(row[COL_ABSTRACT], row['language_code']),
    axis=1
)

print("Processing complete.")
output_columns = [
    COL_PAPER_NAME,
    COL_ABSTRACT,
    'cleaned_title',
    'cleaned_abstract',
    COL_TECHNIQUES, 
    'keywords',     
    'language_code' 
]

output_df = df[output_columns]

print("\n--- Sample Processed Data (First 5 Rows) ---")
print(output_df.head())

SAVE_OUTPUT = True
OUTPUT_FILE_PATH = r'processed_data\preprocessed_nlp_articles.xlsx'

if SAVE_OUTPUT:
    print(f"\nSaving processed data to {OUTPUT_FILE_PATH}...")
    try:
        output_df.to_excel(OUTPUT_FILE_PATH, index=False, engine='openpyxl')
        print("File saved successfully.")
    except Exception as e:
        print(f"Error saving output file: {e}")
