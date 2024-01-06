# python3 run.py --model_path /home/maxloo/src/pastoring/llama-13b --adapter_path /home/maxloo/src/pastoring/adapter --weights_path /home/maxloo/src/pastoring/weights --use_tpu
# Import libraries
# from huggingface_hub import login
# # Log in programmatically
# login(token="")
import torch
import transformers
import requests
print(torch.cuda.is_available())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Load model and adapter weights from local directory
model = transformers.AutoModelForCausalLM.from_pretrained("/home/maxloo/src/pastoring/llama/llama-2-13b")
print(model)
model.to(device)
adapter = transformers.AutoModelForCausalLM.from_pretrained("/home/maxloo/src/pastoring/adapter", config=transformers.configuration.AdapterConfig.from_json_file("adapter_config.json"))
adapter.to(device)
model.load_state_dict(adapter.state_dict())
adapter.load_state_dict(model.state_dict())
# Define prompt
prompt = "Hello, I am a chatbot."
# Perform inference
response = model.generate(prompt, max_length=50)
# Print response
print(response)

