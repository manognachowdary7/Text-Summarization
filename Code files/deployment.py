import streamlit as st
import re
import pickle
import torch

# Load the model and tokenizer using pickle
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# Text cleaning function
def clean_text(text):
    cleaned_text = re.sub(r'[^a-zA-Z\s]', '', text)
    return cleaned_text

# Summarization function
def summarize_text(text, max_length=150, min_length=20):
    input_text = "summarize: " + text
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True)

    summary_ids = model.generate(
        inputs["input_ids"], 
        max_length=max_length, 
        min_length=min_length, 
        length_penalty=2.0, 
        num_beams=4, 
        early_stopping=True
    )
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

# Streamlit app interface
st.title("Text Summarization App")
st.write("Enter an article below and click 'Summarize' to generate a summary.")

# Text input for a single article
article_text = st.text_area("Enter the article text to summarize")

# Button to generate summary
if st.button("Summarize"):
    if article_text:
        # Clean and summarize the article
        cleaned_article = clean_text(article_text)
        summary = summarize_text(cleaned_article)
        
        # Display only the summary
        st.write("### Summary")
        st.write(summary)
    else:
        st.write("Please enter an article text to summarize.")
