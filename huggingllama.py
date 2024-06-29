# Use a pipeline as a high-level helper
from transformers import pipeline

pipe = pipeline("text-generation", model="huggyllama/llama-7b")
# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("huggyllama/llama-7b")
model = AutoModelForCausalLM.from_pretrained("/home/guanyu/ModelCenter-main/results/llama-7b-hf")

# import requests

# API_URL = "https://api-inference.huggingface.co/models/meta-llama/Llama-2-7b-chat-hf"
# headers = {"Authorization": "Bearer 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"}

# def query(payload):
# 	response = requests.post(API_URL, headers=headers, json=payload)
# 	return response.json()
	
# output = query({
# 	"inputs": "Can you please let us know more details about your ", 
# })
# print(output)