# llm.py
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

model_name = "distilgpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

def generate_answer(prompt, max_tokens=512):
    result = pipe(prompt, max_new_tokens=max_tokens, do_sample=True, temperature=0.7)
    return result[0]['generated_text'][len(prompt):].strip()
