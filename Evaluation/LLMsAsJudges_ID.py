import sys
sys.path.append('../')
from chat_models.OpenAI_Chat import OpenAI_Chat
from chat_models.Reasoning_Client import RClient
from pydantic import BaseModel
import json
import multiprocessing
import os
from tqdm import tqdm
import argparse
import re

class Score(BaseModel):
    identification_accuracy: int
    reasoning_accuracy: int

    def to_json(self):
        return {
            "identification_accuracy": self.identification_accuracy,
            "reasoning_accuracy": self.reasoning_accuracy
        }


class Scorer:
    def __init__(self, raw_data_file, output_file, subject_name, openai_api_base, model_name="gpt-4o", num_processes=None, 
    temperature=1, max_records=None):
        self.raw_data_file = raw_data_file
        self.output_file = output_file
        self.offline_model = model_name
        self.model_name = model_name.split("/")[-1]
        self.num_processes = num_processes if num_processes is not None else os.cpu_count()
        self.subject_name = subject_name
        self.temperature = temperature
        self.openai_api_base = openai_api_base
        self.max_records = max_records

    def get_prompt(self, item):
        user_query = item["question"]
        # Retrieve expert answer
        model_response = item[self.subject_name]
        expert_answer = item["answer"]
        category = item.get("category", "unknown")
        assert category in ["Plant Disease Identification", "Plant Identification", "Insect and Pest Identification"]
        
        correct_entity_name = f"\nEntity Name: {item['entity_name']} \nScientific Name: {item['entity_scientific_name']} \nCommon Names: {', '.join(item['entity_common_names'])}\n"

        entity_type = item["entity_type"]

        prompt = f"""\
You are now required to rate a model’s response to an {entity_type} identification question. \
We have the user’s question, the gold answer (Expert’s Answer), and the correct entity name. \
All answers (gold and model) are provided in a single-paragraph “analysis + result” format. \
You need to score the model’s response according to the Score Criteria.

<User Query> {user_query} </User Query>
<Gold Answer> {expert_answer} </Gold Answer>
<Correct Entity Name> {correct_entity_name} </Correct Entity Name>
<Model Response> {model_response} </Model Response>

<Score Criteria>
**Identification Accuracy Definition**: Identification Accuracy assesses whether the model's identification result is consistent with the expert's conclusion. That is, whether the entity identified by the model matches with the expert’s identification result or appears explicitly within any of the provided fields: entity_name, scientific_name, or common_names (case-insensitive).
**Reasoning Accuracy Definition**: Reasoning Accuracy evaluates how effectively the model’s analysis aligns with the expert’s reasoning. It must reflect the presence of key clues (observable characteristics explicitly stated in the gold answer), accuracy and detail of descriptions, and logical coherence through clear causal links (e.g., "Based on..., therefore...").

### Scoring Guidelines
1. **Identification Accuracy (0 or 1 point)**:
   - **1 point**: if the model’s final identification result matches the expert's identification result, or appears in any of the following fields: entity_name, scientific_name, or common_names (case-insensitive).
   - **0 points**: otherwise.

2. **Reasoning Accuracy (0–4 points)**:
   - **4 points**: Covers all key clues (≥2 key clues such as shape, color, distinctive markings) with precise descriptions and clear causal links.
   - **3 points**: Mentions ≥2 key clues; with precise descriptions and establishes some causal links.
   - **2 points**: Mentions 1–2 key clues; with some descriptions and establishes some incomplete causal links.
   - **1 point**: Mentions ≤1 key clues with some descriptions and no causal links. 
   - **0 points**: No usable observations or completely off-topic.
</Score Criteria>

Please output only the scores in JSON with two keys: `identification_accuracy` and `reasoning_accuracy`.  

Example:
{{ "identification_accuracy": ..., "reasoning_accuracy": ... }}

Please output only the JSON object without any additional text.
"""
        
        return {
            "prompt": prompt,
            "expert_answer": expert_answer,
            "model_response": model_response
        }

    def process_item(self, args):
        item, model_name, output_file, lock = args
        prompt = self.get_prompt(item)
        max_retries = 5
        retries = 0
        while retries < max_retries:
            if self.model_name.startswith("gpt"):
                client = OpenAI_Chat(model_name=model_name, messages=[])
            else:
                client = RClient(model_name=self.offline_model, openai_api_base=self.openai_api_base, messages=[])            
            new_item = {
                "judge": self.model_name,
                "id": item.get('id'),
                "question": item.get('question'),
                "attachments": item.get('attachments'),
                "subject_name": self.subject_name,
                "expert_answer": prompt["expert_answer"],
                "model_response": prompt["model_response"],
                "category": item.get("category"),
            }
            
            try:
                if self.model_name.startswith("gpt"):
                    response = client.chat(prompt=prompt["prompt"], response_format=Score, temperature=0)
                    new_item["score"] = response.to_json()
                    response = response.to_json()
                else:
                    content_and_reasoning = client.chat(prompt=prompt["prompt"], temperature=self.temperature)
                    response = self.extract_json(content_and_reasoning["content"])
                    reasoning = content_and_reasoning["reasoning"]
                    new_item["score"] = response
                    new_item["reasoning"] = reasoning
                
                # Check only the accuracy key is present.
                assert "identification_accuracy" in response
                assert "reasoning_accuracy" in response
                new_item["info"] = client.info()
                break
            except Exception as e:
                print(f"Error: {e}")
                print(f"Error processing item {item.get('id', 'unknown')}")
                retries += 1
                print(f"Retrying... ({retries}/{max_retries})")
                new_item["score"] = -1
                
        with lock:
            with open(output_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(new_item, ensure_ascii=False) + '\n')
        
        return new_item.get('id')

    def extract_json(self, string):
        try:
            json_data = json.loads(string)
            return json_data
        except json.JSONDecodeError:
            if isinstance(string, dict):
                return string
            
            pattern = r'```json\s*(\{[\s\S]*?\})\s*```'
            match = re.search(pattern, string)

            if match:
                json_str = match.group(1)
            else:
                # Look for JSON object in the string
                start = string.find('{')
                end = string.rfind('}') + 1
                if start >= 0 and end > start:
                    json_str = string[start:end]
                else:
                    json_str = string.strip()

            json_data = json.loads(json_str)   
            return json_data   

    def scoring(self):
        # Read the raw data file
        with open(self.raw_data_file, "r", encoding='utf-8') as f:
            data = json.load(f)
        
        if getattr(self, "max_records", None) is not None:
            data = data[: self.max_records]

        # Check if the output file exists and read processed items
        processed_ids = set()
        if os.path.exists(self.output_file):
            with open(self.output_file, "r", encoding='utf-8') as f:
                for line in f:
                    try:
                        item = json.loads(line)
                        if 'score' in item and item['score'] != -1 and item['score'] is not None:
                            processed_ids.add(item['id'])
                    except json.JSONDecodeError:
                        continue
                    
        items_to_process = [item for item in data if item.get('id') not in processed_ids]
        print(f"Processing {len(items_to_process)} items.")
        
        if items_to_process:
            manager = multiprocessing.Manager()
            lock = manager.Lock()
            pool = multiprocessing.Pool(processes=self.num_processes)
            args_list = [(item, self.model_name, self.output_file, lock) for item in items_to_process]
            for _ in tqdm(pool.imap_unordered(self.process_item, args_list), total=len(args_list), desc="Processing items"):
                pass
            pool.close()
            pool.join()
        
        print("Processing completed.")
        self.cleanup_output(len(data))

    def cleanup_output(self, data_length):
        valid_items = []
        
        with open(self.output_file, "r", encoding='utf-8') as f:
            for line in f:
                try:
                    item = json.loads(line)
                    if 'score' in item and item['score'] != -1 and item['score'] is not None:
                        valid_items.append(item)
                except json.JSONDecodeError:
                    continue

        with open(self.output_file, "w", encoding='utf-8') as f:
            for item in valid_items:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')

        print(f"Total successful items: {len(valid_items)}. \nRemaining items to process: {data_length - len(valid_items)}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Score identification responses using LLMs.")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the input JSON file.")
    parser.add_argument("--output_file", type=str, required=True, help="Path to the output JSONL file.")
    parser.add_argument("--model_name", type=str, default="gpt-4o", help="Model name to use.")
    parser.add_argument("--num_processes", type=int, default=os.cpu_count(), help="Number of processes to use.")
    parser.add_argument("--subject_name", type=str, required=True, help="Name of the subject's answer field.")
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature for model response generation.")
    parser.add_argument("--openai_api_base", type=str, default="", help="Base URL for OpenAI API.")
    parser.add_argument("--max_records", type=int, default=None, help="Only process the first N records from the input JSON array.")

    args = parser.parse_args()
    scorer = Scorer(args.input_file, args.output_file, args.subject_name, args.openai_api_base, args.model_name, args.num_processes, 
    args.temperature, args.max_records)
    scorer.scoring()
