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
    accuracy: int
    relevance: int
    completeness: int
    parsimony: int

    def to_json(self):
        return {
            "accuracy": self.accuracy,
            "relevance": self.relevance,
            "completeness": self.completeness,
            "parsimony": self.parsimony
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
        model_response = item[self.subject_name]
        expert_answer = item["answer"]

        score_criteria = """
**Accuracy Definition**: Accuracy evaluates whether the agricultural facts, species identification, diagnostic conclusions, and management recommendations provided by the model align with the expert's response. Emphasis is placed on: 1. Correctness of professional terminology (e.g., precise naming of diseases, pests, or invasive species). 2. Accuracy of key details (e.g., descriptions of lesion characteristics, pest behaviors, or plant symptoms). 3. Logical coherence in describing causal relationships (e.g., disease transmission pathways, pest infestation mechanisms). 4. Appropriateness and effectiveness of the proposed management strategies or interventions.
- 4 points: All agricultural facts, terminologies, diagnostic conclusions, and management recommendations are completely correct, comprehensive, and fully aligned with expert consensus.
- 3 points: Minor inaccuracies or omissions in terminology, descriptive details, or management advice exist, but the core diagnostic conclusions and recommended management practices remain accurate and effective.
- 2 points: Noticeable factual errors, misidentifications (species/disease/pests), or suboptimal management suggestions. However, the response still demonstrates partial accuracy or correctness in key aspects.
- 1 point: Major inaccuracies, such as significant confusion between diseases, pests, or plants, flawed causal logic, or incorrect management practices that could lead to ineffective or detrimental outcomes.
- 0 points: Entirely incorrect, scientifically invalid, or significantly misleading claims without any alignment with expert consensus.

**Relevance Definition**: This measures how closely the model’s response matches the scope and focus of expert answers, ensuring it stays on-topic and avoids tangential information. Responses that digress into unrelated agricultural knowledge or overlook critical points tied to the user’s query are considered less relevant.
- 4 points: The response perfectly mirrors the expert answer and directly addresses the query, using precise terminology and only including question-relevant information.
- 3 points: The answer is mostly aligned with the expert response and user query, with only minor tangents or slight omissions in details.
- 2 points: The response contains noticeable deviations or omissions compared to the expert answer, with several off-topic or less relevant points.
- 1 point: Significant misalignment with the expert answer and the query is evident. The response includes major irrelevant or incorrect content.
- 0 points: The answer is entirely off-topic, failing to reflect the expert response or address the user query.

**Completeness Definition**: Whether the model’s answer covers all key information points mentioned in expert answers to fully address the user’s inquiry. If the model omits critical steps or precautions highlighted in expert answers, it is deemed incomplete. Emphasis is placed on: 1. Professional Terminology: Uses precise terms (e.g., names of diseases, pests, invasive species). 2. Key Details: Includes comprehensive descriptions (e.g., lesion characteristics, pest behaviors, plant symptoms). 3. Logical Causal Relationships: Fully explains connections (e.g., disease transmission, pest infestation mechanisms). 4. Management Recommendations: Details all necessary strategies and precautions.
- 4 points: Covers all key points from the gold answer
- 3 points: Misses 1-2 minor details but addresses core aspects. 
- 2 points: The response contains noticeable deviations or omissions compared to the expert answer.
- 1 point: Omits a major component (e.g.,management recommendations).
- 0 points: Fails to address any key elements of the query.

**Parsimony Definition**: Whether the answer provides actionable guidance that directly addresses the user’s core needs, delivering a concise and unambiguous conclusion and specific recommendations without extraneous technical details. The response should adhere to Occam’s Razor by avoiding unnecessary complexity and focusing only on what is essential for understanding whether intervention is necessary and what exact steps (if any) need to be taken.
- 4 points: The answer is succinct, clear, and directly addresses the user’s concerns. It offers straightforward, practical guidance that is fully aligned with the visible evidence without any unnecessary details. It embodies the principle of Occam’s Razor.
- 3 points: The answer is generally concise and practical, offering useful advice. However, it may include some extraneous details or slight ambiguity that only minimally detracts from its overall clarity and directness.
- 2 points: The answer contains relevant information but is overly theoretical or detailed. Extra technical content obscures the key actionable recommendations, making the response less concise and direct. 
- 1 point: The answer is largely indirect or abstract, with a significant amount of unnecessary information. The lack of clarity in actionable guidance leaves the user uncertain about whether any intervention is needed.
- 0 points: The answer fails to provide practical or actionable recommendations and is cluttered with superfluous details, completely missing the concise, straightforward approach required by Occam’s Razor.
"""

        prompt = f"""
You are now required to rate a model's response to an agriculture-related question. \
We have a gold answer, which is Expert's Answer and based on this gold answer, \
and the user's question, you need to score the model's answer according to the following four scoring criteria.

<User Query>{user_query}</User Query>

<Gold Answer>{expert_answer}</Gold Answer>

<Model Response>{model_response}</Model Response>

<Score Criteria>{score_criteria}</Score Criteria>

Please only output the scores without any other content. You should output JSON with four key, accuracy, relevance, completeness, parsimony. The example is shown below:
{{ "accuracy": ..., "relevance": ..., "completeness": ..., "parsimony": ... }}"""
        
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
                "subject_name": self.subject_name,
                "expert_answer": prompt["expert_answer"],
                "model_response": prompt["model_response"],
                "category": item.get('category'),
            }
            
            try:
                if self.model_name.startswith("gpt"):
                    response = client.chat(prompt=prompt["prompt"], response_format=Score, temperature=self.temperature)
                    new_item["score"] = response.to_json()
                    response = response.to_json()         
                else:
                    content_and_reasoning = client.chat(prompt=prompt["prompt"], temperature=self.temperature)
                    response = self.extract_json(content_and_reasoning["content"])
                    reasoning = content_and_reasoning["reasoning"]
                    new_item["score"] = response
                    new_item["reasoning"] = reasoning

                assert "accuracy" in response and "relevance" in response and "completeness" in response and "parsimony" in response
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
                        if 'score' in item and item['score'] != -1 and item['score'] != None:
                            processed_ids.add(item['id'])
                    except json.JSONDecodeError:
                        # Handle potentially corrupt JSON lines
                        continue
                    
        items_to_process = [item for item in data if item.get('id') not in processed_ids]
        print(f"Processing {len(items_to_process)} items.")
        
        if items_to_process:
            manager = multiprocessing.Manager()
            lock = manager.Lock()
            # Initialize the process pool with the specified number of processes
            pool = multiprocessing.Pool(processes=self.num_processes)
            args_list = [(item, self.model_name, self.output_file, lock) for item in items_to_process]
            # Use tqdm to show progress
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
                    if 'score' in item and item['score'] != -1 and item['score'] != None:
                        valid_items.append(item)
                except json.JSONDecodeError:
                    continue

        with open(self.output_file, "w", encoding='utf-8') as f:
            for item in valid_items:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')

        print(f"Total successful items: {len(valid_items)}. \n Remaining items to process: {data_length - len(valid_items)}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scoring Management Tasks with LLMs")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the input JSON file.")
    parser.add_argument("--output_file", type=str, required=True, help="Path to the output JSONL file.")
    parser.add_argument("--model_name", type=str, default="gpt-4o", help="Model name to use.")
    parser.add_argument("--num_processes", type=int, default=os.cpu_count(), help="Number of processes to use.")
    parser.add_argument("--subject_name", type=str, required=True, help="Name of the subject's answer field.")
    parser.add_argument("--openai_api_base", type=str, default="", help="Base URL for OpenAI API.")
    parser.add_argument("--temperature", type=float, default=0, help="Temperature for the model.")
    parser.add_argument("--max_records", type=int, default=None, help="Only process the first N records from the input JSON array.")


    args = parser.parse_args()
    scorer = Scorer(args.input_file, args.output_file, args.subject_name, args.openai_api_base, args.model_name, args.num_processes, 
    args.temperature, args.max_records)
    scorer.scoring()
