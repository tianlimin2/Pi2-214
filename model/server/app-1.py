import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import StoppingCriteria, StoppingCriteriaList, TextIteratorStreamer
from threading import Thread

# Loading the tokenizer and model from Hugging Face's model hub.
tokenizer = AutoTokenizer.from_pretrained("/Users/tianlimin/LLM/model-parameter/TinyLlama-1.1B-intermediate-step-1431k-3T")
model = AutoModelForCausalLM.from_pretrained("/Users/tianlimin/LLM/model-parameter/TinyLlama-1.1B-intermediate-step-1431k-3T")


model = model.to('mps')


# Defining a custom stopping criteria class for the model's text generation.
class StopOnTokens(StoppingCriteria):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        stop_ids = [2]  # IDs of tokens where the generation should stop.
        for stop_id in stop_ids:
            if input_ids[0][-1] == stop_id:  # Checking if the last generated token is a stop token.
                return True
        return False


# Function to generate model predictions.
def predict(message, history):
    inpi
    inputs = tokenizer(message, return_tensors="pt").to('mps')
    outputs = model.generate(**inputs, max_new_tokens=200)
    out = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return out


# Setting up the Gradio chat interface.
gr.ChatInterface(predict,
                 title="Tinyllama_chatBot",
                 description="Ask Tiny llama any questions",
                 examples=['How to cook a fish?', 'Who is the president of US now?']
                 ).launch()  # Launching the web interface.
