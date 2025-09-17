import re
import unicodedata
import pandas as pd

# -------------------------------------- Normalize the Text -------------------------------------- #
# handles the different aliases of name entities
VARIANT_MAP = {
    "seetha": "sita", "seeta": "sita", "maithili": "sita", "janaki": "sita", "vaidehi": "sita",
    "ram": "rama", "raghava": "rama",
    "hanuma": "hanuman", 
    "lakshman": "lakshmana"
}

# normalize the text for model input
def normalize(text):
    # Unicode normalization
    text = unicodedata.normalize("NFKC", text).encode('ASCII', 'ignore').decode('utf-8')
    
    # Lowercase
    text = text.lower()

    # Remove quotes not part of contractions or possessives
    text = re.sub(r"(?<=\s)'|'(?=\s)|^'|'$", "", text)

    # Remove unwanted punctuation, keeping . , / - ' : ; ? ! " ( ) and apostrophes in contractions/in-text
    text = re.sub(r"[^\w\s'.,/\-:;?!\()]", "", text)

    # Normalize ellipses
    text = re.sub(r"\.{2,}", ".", text)

    # Remove punctuation at start/end if isolated
    text = re.sub(r"^[.,\s]+|[.,\s]+$", "", text)

    # 4. Apply variant mapping
    for variant, canonical in VARIANT_MAP.items():
        # Match variant in word-boundary-aware regex, allow possessive `'s`
        pattern = re.compile(fr"\b{variant}\b(?=(?:[']s)?[.,;:?!\"]*|\s|$)", flags=re.IGNORECASE)
        text = pattern.sub(canonical, text)

    return text

# -------------------------------------- Load verse dataset -------------------------------------- #
def load_dataset(DATA_PATH):
    # Load data
    df = pd.read_csv(DATA_PATH)
    # renaming the columns for easy handling
    df = df.rename(columns={ 
                'Kanda/Book': 'Book',
                'Sarga/Chapter': 'Chapter',
                'Shloka/Verse Number': 'Verse_number', 
                'English Translation': 'Verse_Text',
                })
    df['ID'] = df['Book'].astype(str) + '-' + df['Chapter'].astype(str) + '-' + df['Verse_number'].astype(str)
    df = df[['ID'] + [col for col in df.columns if col != 'ID']]

    return df