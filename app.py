import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load model and tokenizer
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")
    model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1", torch_dtype=torch.float16, device_map="auto")
    return tokenizer, model

tokenizer, model = load_model()

# Streamlit UI
st.title("Local LLM Chatbot")
user_input = st.text_input("Ask me anything:")

if user_input:
    inputs = tokenizer(f"[INST] {user_input} [/INST]", return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=200)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    st.write(response)
