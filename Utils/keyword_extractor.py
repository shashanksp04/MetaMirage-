# ==========================================
# keyword_extractor.py
# ==========================================
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
import re
from chat_models.Client import Client  # <-- your local model client wrapper

class KeywordExtractor:
    """Extracts and organizes keywords from a user query using local LLM via Client."""

    def __init__(self, model_name="Qwen/Qwen2.5-14B-Instruct", openai_api_base="http://127.0.0.1:8000/v1"):
        self.model_name = model_name
        self.openai_api_base = openai_api_base
        # Don't create client here - create fresh one for each call to avoid message accumulation

    def extract_keywords(self, query: str):
        prompt = f"""
        You are an intelligent keyword extraction assistant.
        Your goal is to extract the most important and relevant keywords or short phrases
        from the following user query, and organize them in a logical order that makes
        the resulting string ideal for performing a Google web search.

        Combine related words when appropriate (for example: "soybean pests", "wheat diseases", "Maryland agriculture"),
        and include key topics, entities (like crops, pests, locations, and years), and domain-specific terms.

        If a keyword or phrase contains multiple words that represent a fixed concept or named entity
        (for example: "soybean pests", "climate change", "drought impact"),
        enclose it in double quotes (" ") to make it search-optimized for SerpAPI and Google.
        Do NOT quote single words.

        The output must be a JSON list of strings, ordered from general to specific,
        so it can be joined and passed directly as a search query to the SerpAPI.

        Example:
        Input: "impact of drought on corn and soybean pests in Maryland 2022"
        Output: ["\\"drought impact\\"", "corn", "\\"soybean pests\\"", "Maryland", "2022"]

        Query: {query}
        """

        try:
            # ---- Model Call ----
            # Create fresh client for each call to avoid message accumulation and context window issues
            client = Client(model_name=self.model_name, openai_api_base=self.openai_api_base, messages=[])
            text = client.chat(prompt=prompt, images=[])
            print("🧠 Raw model response:\n", text, "\n")

            # ---- Clean Formatting ----
            # Remove code block markers
            text = re.sub(r"```(?:json)?", "", text).replace("```", "").strip()
            
            # Remove control characters (except newlines and tabs)
            text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)
            
            # Fix common JSON issues
            text = text.replace('\\"', '"')
            text = re.sub(r'""', '"', text)
            
            # Try to extract JSON array from text
            match = re.search(r"\[.*?\]", text, re.DOTALL)
            if match:
                cleaned = match.group(0)
            else:
                cleaned = text
            
            # Fix unclosed strings and brackets
            # Count quotes to ensure they're balanced
            quote_count = cleaned.count('"')
            if quote_count % 2 != 0:
                # Unclosed quote - try to fix by adding closing quote before ]
                cleaned = re.sub(r'(\w+)\s*\]', r'"\1"]', cleaned)
            
            # Ensure proper JSON array format
            if not cleaned.strip().startswith('['):
                cleaned = '[' + cleaned
            if not cleaned.strip().endswith(']'):
                cleaned = cleaned + ']'
            
            # ---- Parse JSON ----
            try:
                keywords = json.loads(cleaned)
            except json.JSONDecodeError as e:
                # Fallback: try to extract strings manually
                print(f"[Warning] JSON parsing failed, trying fallback extraction: {e}")
                # Extract quoted strings
                string_matches = re.findall(r'"([^"]*)"', cleaned)
                if string_matches:
                    keywords = string_matches
                else:
                    # Last resort: split by common delimiters
                    keywords = [w.strip().strip('"\'') for w in re.split(r"[,;\n]", cleaned) if w.strip() and w.strip() not in ['[', ']']]

            # Clean and validate keywords
            keywords = [kw.strip() for kw in keywords if isinstance(kw, str) and kw.strip() and len(kw.strip()) > 0]
            print("✅ Extracted keywords:", keywords)
            return keywords

        except Exception as e:
            print(f"[Error] Keyword extraction failed: {e}")
            return []
