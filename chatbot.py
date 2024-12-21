#Dependencies: pypdf, transformers, sentencepiece, sentence_transformers, cupy, torch, tensorflow,and flax, tf-keras, wikipediaapi, spacy from pip. numba and cudatoolkit from conda.
#First time running the program in 2-book report mode it will take longer to create AI model.
# importing required modules
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import torch
import re
import math
import sys
from transformers import pipeline, DistilBertTokenizer, DistilBertForQuestionAnswering
import time
import wikipediaapi
import spacy
from sentence_transformers import SentenceTransformer, util
import warnings
import tensorflow as tf

# Set TensorFlow logger to only show errors
tf.get_logger().setLevel('ERROR')
# Suppress specific TensorFlow warning
warnings.filterwarnings("ignore", category=UserWarning, message=".*deprecated.*")

def make_valid_string(input_string):
    # Regex pattern to match only alphabetic characters (A-Z, a-z)
    pattern = r'[^a-zA-Z0-9 ×^.-_,!?\n]'  # Match anything that is NOT a letter
    return re.sub(pattern, '', input_string)

def log_to_file(filename, context, question, answer, log_context):
    """
    Appends the question and answer to the specified text file.

    Args:
        filename (str): The name of the text file.
        context (str): The context or textbook section given for the question.
        question (str): The user's question.
        answer (str): The chatbot's answer.
    """
    with open(filename, "a") as file:
        if log_context:
            file.write(f"Context: {context}\n")
        file.write(f"Question: {question}\n")
        file.write(f"Answer: {answer}\n")
        file.write("\n")  # Add a blank line for better readability

def log_wiki(name, filename='chat_history.txt'):
    with open(filename, "a") as file:
        file.write(f"Searched Wikipedia.org page - {name}\n")

def extract_keywords(question, nlp, num_keywords=5):
    # Process the question sentence
    doc = nlp(question)
    keywords = set()

    # Add Named Entities to the list of keywords
    for ent in doc.ents:
        keywords.add(ent.text)
    #if len(keywords) > 0:
        #Return named entities only
        #return list(keywords)[:num_keywords]

    # Find the subject
    for token in doc:
        # The subject is usually tagged as "nsubj" (nominal subject) or "nsubjpass" (subject of passive voice)
        if "subj" in token.dep_:
            keywords.add(token.text)
            #break  # Assuming there's only one subject, but you can modify to get all subjects
    
    # Check for direct objects (dobj) or prepositional objects (pobj)
    for token in doc:
        if token.dep_ in {"dobj", "pobj", "obj"}:
            keywords.add(token.text)
    
    if len(keywords) == 0:
        # Extract nouns, proper nouns, or named entities
        for token in doc:
            if token.pos_ in {"NOUN", "PROPN"}:  # Nouns and Proper Nouns
                keywords.add(token.text)
    
    # Return the top `num_keywords` keywords
    return list(keywords)[:num_keywords]

#Find number of words in a string based on spacing.
def word_count(sentence):
    words = sentence.split(' ')
    return len(words)

def fetch_wikipedia_page(page_title, wiki):
    """Fetch the Wikipedia page and return its title and sections."""
    page = wiki.page(page_title)
    
    if not page.exists():
        raise ValueError(f"Page '{page_title}' does not exist on Wikipedia.")
    log_wiki(page_title)
    return page

def extract_sections_with_titles(page):
    """Extract sections and their corresponding titles from a Wikipedia page."""
    sections = []
    sections.append(('Summary', page.summary))
    def recursive_section_extraction(section, prefix=""):
        section_title = f"{prefix}{section.title}"
        sections.append((section_title, section.text))
        for sub_section in section.sections:
            recursive_section_extraction(sub_section, prefix=section_title + " > ")
    
    for section in (page.sections):
        recursive_section_extraction(section)
    
    return sections

def semantic_search(model, query, sections, top_k=1):
    """Perform semantic search on section titles and return the most relevant sections."""
    section_titles = [section[0] for section in sections]  # Extract section titles
    section_texts = [section[1] for section in sections]  # Extract section texts

    # Compute embeddings for query and section titles
    query_embedding = model.encode(query, convert_to_tensor=True)
    title_embeddings = model.encode(section_titles, convert_to_tensor=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Move query_embedding and title_embeddings to the same device
    query_embedding = query_embedding.to(device)
    title_embeddings = title_embeddings.to(device)
    # Calculate cosine similarity
    similarities = util.pytorch_cos_sim(query_embedding, title_embeddings).squeeze(0)
    top_indices = similarities.topk(top_k).indices.tolist()  # Top K relevant indices

    # Retrieve relevant sections
    relevant_sections = [sections[i] for i in top_indices]
    return relevant_sections

# Main Functionality
def get_relevant_details(model, query, page_title, wiki, top_k=3):
    """Retrieve relevant details from a Wikipedia page based on a query."""
    page = fetch_wikipedia_page(page_title, wiki)
    sections = extract_sections_with_titles(page)
    # Perform semantic search to find relevant sections
    relevant_sections = semantic_search(model, query, sections, top_k)

    # Collect details from relevant sections
    details = []
    for title, text in relevant_sections:
        details.append(f"{text}")

    return details

def rank_sentences(query, sentences, model):
    """
    Rank sentences based on their relevance to the query.
    
    Args:
    - query (str): The query string.
    - sentences (list): List of sentences to rank.
    - model_name (str): Pretrained SentenceTransformer model to use.
    
    Returns:
    - List of tuples: [(sentence, score), ...] ranked by relevance.
    """
    if not sentences:
        print("Error: No sentences to rank.")
        return []
    # Compute embeddings for the query and sentences
    query_embedding = model.encode(query, convert_to_tensor=True)
    sentence_embeddings = model.encode(sentences, convert_to_tensor=True)
    
    # Ensure both tensors are on the same device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    query_embedding = query_embedding.to(device)
    sentence_embeddings = sentence_embeddings.to(device)  # Move sentence_embeddings to the same device

    # Compute cosine similarities
    similarity_scores = util.pytorch_cos_sim(query_embedding, sentence_embeddings).squeeze(0)
    
    # Pair each sentence with its similarity score
    sentence_score_pairs = list(zip(sentences, similarity_scores.tolist()))
    
    # Sort by similarity score (descending order)
    ranked_sentences = sorted(sentence_score_pairs, key=lambda x: x[1], reverse=True)
    return [x[0] for x in ranked_sentences]

def top_sentences_from_wiki(wiki, keywords, query, semantic_search_model, num_sentences):
    
    #Check every sentence from wiki with sentences-transformers semantic search for relevance.
    details = []
    for k in keywords:
        details = details + (get_relevant_details(semantic_search_model, query, k, wiki))

    #Split details into sentences.
    sentences = []
    for d in details:
        sentences = sentences + re.split(r'[.?!]', d)  # Split by '.', '?', or '!'
    
    return rank_sentences(query, sentences, semantic_search_model)[:num_sentences]


def main():
        
    print('GPU Dependency- Is torch CUDA available? ' + 'Yes.' if torch.cuda.is_available() else 'No.')
    print('GPU device count: ' + str(torch.cuda.device_count()))  # Should show the number of GPUs
    print('GPU Device name: ' + str(torch.cuda.get_device_name(0)))  # Name of the GPU

    # Load the tokenizer and model explicitly
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased-distilled-squad")
    model = DistilBertForQuestionAnswering.from_pretrained("distilbert-base-uncased-distilled-squad")

    #Create distlled bert question answering pipeline abstraction.
    qa_bert = pipeline(
        "question-answering",model=model, tokenizer=tokenizer, device=0
    )
    
    # Set up Wikipedia API with a custom user agent
    user_agent = "MyWikipediaBot/1.0 (https://github.com/eriksthompson; erikthompson12345@gmail.com)"
    wiki_wiki = wikipediaapi.Wikipedia(
        language='en',
        user_agent=user_agent
    )

    # Load the spaCy model
    nlp = spacy.load("en_core_web_sm")

    # Initialize the model
    #msmarco distilbert is assymmetric model meaning it finds answers to queries.
    semantic_search_model = SentenceTransformer("msmarco-distilbert-dot-v5", device='cuda') 

    save_history = input('Enter y for yes or n for no to save history of questions: \n')
    print('Rules: Make question factual, only capitalize proper nouns, spell out full names correctly and have fun!')
    question = input('Enter a question for chatbot to answer: (exit to quit)\n')
    start_time = time.time()
    while(question != 'exit' and question != ''):
        
        #context = input('Enter some context/textbook paragraph to your question that may provide information about answer:\n')
        #while context == '':
            #context = input('Error cannot be empty. Enter some context to your question that may provide information about answer:\n')

        #Context discovery:
        #Find keywords to search on wiki from Spacy keyword extraction.
        #word_count1 = word_count(question)
        #Keywords dependent on question length.
        num_keywords = 3
        num_rel_sentences = 20
        #if word_count1 >= 8:
            #num_keywords = 2
        #if word_count1 >= 16:
            #num_keywords = 3
            
        
        #print(keywords)
        keywords = extract_keywords(question, nlp, num_keywords)
        print(keywords)
        #Check every sentence from wiki with sentences-transformers semantic search for relevance.\
        top_sentences = top_sentences_from_wiki(wiki_wiki, keywords, question, semantic_search_model, num_rel_sentences)
        #print(top_sentences)
        top_joined = '\n'.join(top_sentences)
        #Pass top 20 sentences most relevant to context.
        context = make_valid_string(top_joined)
        response = qa_bert(question=question, context=context)
        answer = response['answer'].strip(', ')
        print('Answer: ', answer)
        if save_history == 'y':
            log_to_file('chat_history.txt', context, question, answer, False)       
        question = input('Enter a question for chatbot to answer: (exit to quit)\n')
    print('Time with QA-chatbot: ', str(math.trunc((time.time()-start_time)//60)), ' minutes ' , str(math.trunc((time.time()-start_time)%60)), ' seconds.')
if __name__ == "__main__":
    main()