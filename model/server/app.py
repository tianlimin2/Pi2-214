from text2vec import SentenceModel
import gradio as gr
from qdrant_client import QdrantClient
from threading import Thread
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import StoppingCriteria, StoppingCriteriaList, TextIteratorStreamer

t2v_model = SentenceModel("/Users/tianlimin/LLM/model-parameter/text2vec_cmed")

with open('key.txt','r') as f:
    key = f.read()

def to_embeddings(text):
    sentence_embeddings = t2v_model.encode(text)
    return sentence_embeddings

tokenizer = AutoTokenizer.from_pretrained("/Users/tianlimin/LLM/model-parameter/TinyLlama-1.1B-intermediate-step-1431k-3T")
model = AutoModelForCausalLM.from_pretrained("/Users/tianlimin/LLM/model-parameter/TinyLlama-1.1B-intermediate-step-1431k-3T").half().to('mps')

def prompt(question, answers):
    q = 'based on the following question and answer,evaluate the user story \n' 
    for index, answer in enumerate(answers):
        q += str(index + 1) + '. ' + str(answer['text']) + '\n'
    q = q+"user story：%s answer：" % question
    return q

def query(text):
    client = QdrantClient(
        url="https://635ebebf-42a7-4833-8d6c-83d8d7685d5f.us-east4-0.gcp.cloud.qdrant.io", 
        api_key=key,
    )
    collection_name = "questions"
    
    vector = to_embeddings(text)

    search_result = client.search(
        collection_name=collection_name,
        query_vector=vector.tolist(),
        limit=1,
        search_params={"exact": False, "hnsw_ef": 128}
    )
    answers = []

    for result in search_result:
        if len(result.payload["text"]) > 5000:
            summary = result.payload["text"][:5000]
        else:
            summary = result.payload["text"]
        answers.append({ "text": summary})
    promptMessage=prompt(text, answers)
    return promptMessage



# Function to generate model predictions.
def predict(message, history):
    his = query(message)
    inputs = tokenizer(his, return_tensors="pt").to('mps')
    outputs = model.generate(**inputs, max_new_tokens=200)
    out = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return out


# Setting up the Gradio chat interface.
gr.ChatInterface(predict,
                 title="Tinyllama_chatBot",
                 description="Ask Tiny llama any questions",
                 examples=['As a site visitor residing in the borough, I want to effortlessly explore and stay informed about upcoming events and activities in my community.']
                 ).launch()  # Launching the web interface.