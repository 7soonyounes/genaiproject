from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPTNeoForCausalLM
import torch

class PoetryChain:
    def __init__(self, model_name="gpt2"):
        self.model_name = model_name
        
        if model_name == "gpt2":
            self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
            self.model = GPT2LMHeadModel.from_pretrained(model_name)
        elif model_name == "gpt-neo":
            self.tokenizer = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
            self.model = GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-125M")

        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.model.eval()

    def processing_prompt(self, user_input):
        processed_prompt = f'''Write a beautiful and expressive poem based on the words input by the user. 
        Words provided by user: {user_input}.
        '''
        return processed_prompt

    def generate_poem(self, prompt, max_length=50):
        processed_prompt = self.processing_prompt(prompt)

        inputs = self.tokenizer.encode(processed_prompt, return_tensors='pt')
        attention_mask = torch.ones(inputs.shape, dtype=torch.long)

        with torch.no_grad():
            outputs = self.model.generate(
                inputs, 
                max_length=max_length, 
                num_return_sequences=1, 
                attention_mask=attention_mask,
                pad_token_id=self.tokenizer.eos_token_id  
            )
        
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_text


poetry_chain = PoetryChain(model_name="gpt2")
poem = poetry_chain.generate_poem("dark, sad", max_length=100)
print(poem)