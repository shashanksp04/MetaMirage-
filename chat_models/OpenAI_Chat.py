from openai import OpenAI
import base64
import copy
import math
from PIL import Image

# Function to encode the image
def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

class OpenAI_Chat():
    def __init__(self, model_name="gpt-4o", messages=[]):
        self.client = OpenAI()
        self.model_name = model_name
        self.messages = messages
        self.history_info = copy.deepcopy(messages)
        # Define pricing per model
        self.pricing = {
            "gpt-4o": {
                "input": 2.50 / 1_000_000,  # $2.50 per 1M input tokens
                "output": 10.00 / 1_000_000,  # $10.00 per 1M output tokens
            },
            "gpt-4o-mini": {
                "input": 0.150 / 1_000_000,  # $0.150 per 1M input tokens
                "output": 0.600 / 1_000_000,  # $0.600 per 1M output tokens
            },
            "gpt-3.5-turbo-0125": {
                "input": 0.50 / 1_000_000,  # $0.50 per 1M tokens
                "output": 1.50 / 1_000_000,  # $1.50 per 1M tokens
            },
            "gpt-4.1": {
                "input": 2.00 / 1_000_000,  # $2.00 per 1M tokens
                "output": 8.00 / 1_000_000,  # $8.00 per 1M tokens
            },
            "gpt-4.1-mini": {
                "input": 0.400 / 1_000_000,  # $0.400 per 1M tokens
                "output": 1.600 / 1_000_000,  # $1.600 per 1M tokens
            },
            "gpt-4.1-nano":{
                "input": 0.100 / 1_000_000,  # $0.100 per 1M tokens
                "output": 0.400 / 1_000_000,  # $0.400 per 1M tokens
            }
        }
        self.input_text_tokens = 0
        self.input_image_tokens = 0
        self.output_text_tokens = 0

    def chat(self, prompt, images=[], response_format=None, temperature=1):
        # Images
        if images == []:
            self.messages.append({"role": "user", "content": prompt})
            self.history_info.append({"role": "user", "content": prompt})
        else:
            content = [{"type": "text", "text": prompt}]
            for image in images:
                base64_image = encode_image(image)
                content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}})
            self.messages.append({"role": "user", "content": content})
            self.history_info.append({"role": "user", "content": prompt})
            self.input_image_tokens = self.calculate_vision_pricing(images)
    
        if response_format is None:
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=self.messages,
                temperature=temperature
            )
            response = completion.choices[0].message.content
            self.messages.append({"role": "assistant", "content": response})
            self.history_info.append({"role": "assistant", "content": response})
        else:
            # JSON format
            completion = self.client.beta.chat.completions.parse(                
                model=self.model_name,
                messages=self.messages,
                response_format=response_format,
                temperature=temperature
            )
            response = completion.choices[0].message.parsed
            self.messages.append({"role": "assistant", "content": str(response.to_json())})
            self.history_info.append({"role": "assistant", "content": str(response.to_json())})
            
        self.input_text_tokens = completion.usage.prompt_tokens - self.input_image_tokens
        self.output_text_tokens = completion.usage.completion_tokens
        
        return response

    # Calculate the image tokens
    def calculate_vision_pricing(self, image_paths, detail='high'):
        all_tokens = 0
        if len(image_paths) != 0:
            all_tokens = 85
        # Get the image dimensions
        for image_path in image_paths:
            with Image.open(image_path) as img:
                width, height = img.size
                
            if detail == 'low':
                return 85

            # Scale down to fit within a 2048 x 2048 square if necessary
            if width > 2048 or height > 2048:
                max_size = 2048
                aspect_ratio = width / height
                if aspect_ratio > 1:
                    width = max_size
                    height = int(max_size / aspect_ratio)
                else:
                    height = max_size
                    width = int(max_size * aspect_ratio)

            # Resize such that the shortest side is 768px if the original dimensions exceed 768px
            min_size = 768
            aspect_ratio = width / height
            if width > min_size and height > min_size:
                if aspect_ratio > 1:
                    height = min_size
                    width = int(min_size * aspect_ratio)
                else:
                    width = min_size
                    height = int(min_size / aspect_ratio)

            tiles_width = math.ceil(width / 512)
            tiles_height = math.ceil(height / 512)
            all_tokens += 170 * (tiles_width * tiles_height)
        
        return all_tokens

    def info(self):
        model_name = self.model_name

        model_pricing = self.pricing.get(model_name)
        if not model_pricing:
            raise ValueError(f"Pricing information for model '{model_name}' not found.")

        input_cost = (self.input_text_tokens + self.input_image_tokens) * model_pricing["input"]
        output_cost = (self.output_text_tokens) * model_pricing["output"]

        total_cost = input_cost + output_cost
        info = {
            "model": model_name,
            "input_text_tokens": self.input_text_tokens,
            "input_image_tokens": self.input_image_tokens,
            "output_text_tokens": self.output_text_tokens,
            "input_cost": input_cost,
            "output_cost": output_cost,
            "total_cost": total_cost,
        }
        
        return info    
    
    def get_history(self):
        return self.history_info
