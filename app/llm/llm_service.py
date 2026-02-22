import torch
from .model_loader import load_local_model
from .api_fallback import call_api

class LLMService:
    def __init__(self):
        try:
            self.tokenizer, self.model = load_local_model()
            self.local_available = True
            print("Local model loaded")
        except Exception as e:
            print("Local model failed:", e)
            self.local_available = False

    def generate(self, prompt, max_tokens=512):

        # Try local first
        if self.local_available:
            try:
                inputs = self.tokenizer(prompt, return_tensors="pt")
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=max_tokens
                    )
                return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            except Exception as e:
                print("⚠ Local inference failed, switching to API:", e)

        # Fallback to API
        return call_api(prompt, max_tokens=max_tokens)