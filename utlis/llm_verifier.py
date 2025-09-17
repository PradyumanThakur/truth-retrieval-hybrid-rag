import re
import json

from models.load_models import load_all_models

models = load_all_models()

llm = models["mistral_llm"]
chunks = models["verses"]

def build_prompt(claim, verses):
    verses_text = "\n".join([f"[{i}]- {v['text']}" for i, v in enumerate(verses, 1)])
    prompt = f"""
        <s>[INST] 
        You are a scholarly expert in the Valmiki Ramayana.  
        Your task is to fact-check the given CLAIM using the provided VERSES and return a valid JSON object.  

        The JSON must follow this format:
        {{
            "relevance": "RAMAYANA_RELATED" | "NOT_RAMAYANA_RELATED",
            "label": "TRUE" | "FALSE",
            "reference": [list of verse numbers],
            "explanation": "1–2 sentences why the claim is TRUE or FALSE, citing verse numbers."
        }}
        [/INST]
        CLAIM: {claim}
        VERSES: {verses_text}
        </s>
        [INST] 
        Now return the JSON object only.  
        [/INST]
    """
    return prompt

# Streaming generator
def generate_stream(claim, verses):
    prompt = build_prompt(claim, verses)
    stream = llm(
        prompt, 
        max_tokens=256, 
        temperature=0.2, 
        stream=True, 
        stop=["</s>", "[/INST]"]
    )

    for output in stream: # enables token streaming
        if "choices" in output and len(output["choices"]) > 0:
            token = output["choices"][0]["text"]
            if token:  # avoid None
                yield token