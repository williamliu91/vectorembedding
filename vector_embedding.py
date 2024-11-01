import os
import streamlit as st
import numpy as np
import re
from langchain_groq import ChatGroq

# Load the API key from secret.txt
with open('secret.txt') as f:
    key = f.read().strip()
os.environ["GROQ_API_KEY"] = key

# Initialize ChatGroq model
llm = ChatGroq(
    model="llama3-8b-8192",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

# Streamlit UI
st.title("Text Embedding Generator with ChatGroq")

# Text input
text_input = st.text_area("Enter text to generate embeddings:")

# Function to generate embeddings
def generate_embedding(text):
    response = llm.invoke([("system", "Generate an embedding for the following text and explain the meaning of it."),
                           ("user", text)])
    
    # Print the raw content to understand its format
    st.write("Raw response content:", response.content)

    # Extract numeric values from response.content using regex
    try:
        numbers = re.findall(r"[-+]?\d*\.\d+|\d+", response.content)  # Find all floats and integers
        embedding = np.array([float(num) for num in numbers])
        
        if embedding.size > 0:
            return embedding
        else:
            st.error("No numeric values found in the response content.")
            return None
    except Exception as e:
        st.error(f"Error parsing embedding: {e}")
        return None

# Button to generate embeddings
if st.button("Generate Embedding"):
    if text_input.strip():
        try:
            # Generate embedding
            embedding = generate_embedding(text_input)
            
            # If the embedding is valid, display it
            if embedding is not None:
                st.subheader("Generated Embedding Vector:")
                st.write(embedding)

                # Display vector stats
                st.subheader("Vector Statistics:")
                st.write(f"Vector Dimension: {len(embedding)}")
                st.write(f"Mean Value: {np.mean(embedding):.5f}")
                st.write(f"Standard Deviation: {np.std(embedding):.5f}")
        except Exception as e:
            st.error(f"Error generating embedding: {e}")
    else:
        st.warning("Please enter some text.")
