import re

# -------------------------------------- Cleaned Parsed Output -------------------------------------- #
# Function to map the supporting verse back to the reference in a described format Book, Sarga _, Shloka _ e.g. Bala Kanda, Sarga 5, Shloka 9
def extract_ref(json, top_verses, chunks):
    if not json.get('reference'):
        return json  # If the list is empty or key doesn't exist, return as-is

    updated_refs = []
    for s in json['reference']:
        if 1 <= s <= len(top_verses):  # Ensure index is valid
            idx = top_verses[s-1]['idx']
            meta = chunks[idx].get("meta", {})

            book = meta.get("book", "Unknown Book")
            chapter = meta.get("chapter", "?")
            verse_num = meta.get("verse_num", "?")

            formatted_ref = f"{book}, Sarga {chapter}, Shloka {verse_num}"
            updated_refs.append((s, formatted_ref))

    json['reference'] = updated_refs
    return json

# Regex to match references like "verse 1", "verses 2 and 3", "verse 4-5", etc.
reference_pattern = re.compile(r"""
    (?:
        [\(\[\{]?\s*                         # optional opening ( [ {
        (?:(?i:verses?|references?)[:\s]*)? # optional prefix
        (
            \d+(?:\s*-\s*\d+)?              # number or range
            (?:\s*,\s*\d+(?:\s*-\s*\d+)?)*  # , more numbers or ranges
            (?:\s*(?:,?\s*and\s*)\d+(?:\s*-\s*\d+)?)?  # optional and N
        )
        \s*[\)\]\}]?                         # optional closing ) ] }
    )
""", flags=re.VERBOSE | re.IGNORECASE)

# Helper to extract list of verse numbers
def expand_numbers(num_str):
    nums = []
    for part in re.split(r",|\band\b", num_str):
        part = part.strip()
        if "-" in part:
            start, end = map(int, part.split("-"))
            nums.extend(range(start, end + 1))
        elif part.isdigit():
            nums.append(int(part))
    return nums

# Replace matched reference with metadata
def replacement_func(match):
    num_str = match.group(1)
    verse_nums = expand_numbers(num_str)
    replacements = [dict(predicted_ref).get(num, f"[Verse {num} missing]") for num in verse_nums]
    return "(" + "; ".join(replacements) + ")"

# processing the explanation for replacing the verse no. with proper reference
def explanation_processing(predicted_json):
    global predicted_ref
    predicted_exp = predicted_json['explanation']
    if 'reference' in predicted_json:
        predicted_ref = predicted_json['reference']
    else:
        predicted_ref = [] 
    
    # Replace all references in text
    final_exp = re.sub(reference_pattern, replacement_func, predicted_exp)
    return final_exp