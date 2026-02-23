from openai import OpenAI
import base64

# Function to encode the image
def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

class Client:
    def __init__(self, model_name="Qwen2-VL-7B-Instruct", openai_api_key = "token-abc123", openai_api_base = "None", messages=[]):
        self.client = OpenAI(
            api_key=openai_api_key,
            base_url=openai_api_base,
        )
        self.model_name = model_name
        self.messages = messages
        self.history = []
    
    def chat(self, prompt, images=[], response_format=None):
        # Images
        if images == []:
            self.messages.append({"role": "user", "content": prompt})
            self.history.append({"role": "user", "content": prompt})
        else:
            content = [{"type": "text", "text": prompt}]
            for image in images:
                base64_image = encode_image(image)
                content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}})
            self.messages.append({"role": "user", "content": content})
            self.history.append({"role": "user", "content": prompt})
    
        if response_format is None:
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=self.messages,
                max_completion_tokens=4096,
                temperature=0.8,
                top_p=0.95,
            )
            response = completion.choices[0].message.content
            self.messages.append({"role": "assistant", "content": response})
            self.history.append({"role": "assistant", "content": response})
        else:
            # JSON format
            completion = self.client.beta.chat.completions.parse(                
                model=self.model_name,
                messages=self.messages,
                response_format=response_format
            )
            response = completion.choices[0].message.parsed
            self.messages.append({"role": "assistant", "content": str(response.to_json())})
            self.history.append({"role": "assistant", "content": str(response.to_json())})
        return response
    
    def get_history(self):
        return self.history
