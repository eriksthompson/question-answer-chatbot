#Dependencies: pypdf, transformers, sentencepiece, torch, tensorflow,and flax, tf-keras from pip. numba and cudatoolkit from conda.
#First time running the program in 2-book report mode it will take longer to create AI model.
# importing required modules
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow
import torch
from torch.profiler import profile, ProfilerActivity
from pypdf import PdfReader
import re
from transformers import T5Tokenizer, T5ForConditionalGeneration
import sys
from transformers import pipeline, DistilBertTokenizer, DistilBertForQuestionAnswering
from torch.profiler import profile, ProfilerActivity
import time


def log_to_file(filename, context, question, answer):
    """
    Appends the question and answer to the specified text file.

    Args:
        filename (str): The name of the text file.
        context (str): The context or textbook section given for the question.
        question (str): The user's question.
        answer (str): The chatbot's answer.
    """
    with open(filename, "a") as file:
        file.write(f"Context: {context}\n")
        file.write(f"Question: {question}\n")
        file.write(f"Answer: {answer}\n")
        file.write("\n")  # Add a blank line for better readability

def main():
        
    print('GPU Dependency- Is torch CUDA available? ' + 'Yes.' if torch.cuda.is_available() else 'No.')
    print('GPU device count: ' + str(torch.cuda.device_count()))  # Should show the number of GPUs
    print('GPU Device name: ' + str(torch.cuda.get_device_name(0)))  # Name of the GPU

    # Load the tokenizer and model explicitly
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased-distilled-squad")
    model = DistilBertForQuestionAnswering.from_pretrained("distilbert-base-uncased-distilled-squad")

    oracle = pipeline(
        "question-answering",model=model, tokenizer=tokenizer, device=0
    )
    
    save_history = input('Enter y for yes or n for no to save history of questions: \n')
    question = input('Enter a question for chatbot to answer: (exit to quit)\n')
    while(question != 'exit' or question != ''):
        
        context = input('Enter some context/textbook paragraph to your question that may provide information about answer:\n')
        while context == '':
            context = input('Error cannot be empty. Enter some context to your question that may provide information about answer:\n')
        response = oracle(question=question, context=context)
        answer = response['answer'].strip(', ')
        print('Answer: ', answer)
        if save_history == 'y':
            log_to_file('chat_history.txt', context, question, answer)       
        question = input('Enter a question for chatbot to answer: (exit to quit)\n')

if __name__ == "__main__":
    main()