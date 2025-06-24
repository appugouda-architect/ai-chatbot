import streamlit as st
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
import time

# Initialize LLM
model = OllamaLLM(model="llama3.2")

# Define prompt
template = """
You are a helpful assistant. Answer the following question based on the provided context.
{context}
Question: {question}
Answer:"""
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

# Streamlit UI setup
st.set_page_config(page_title="Langchain Chatbot", layout="centered")
st.title("🧠 AI Chatbot ")

# Session state for chat
if "context" not in st.session_state:
    st.session_state.context = ""
if "messages" not in st.session_state:
    st.session_state.messages = []


# Display past messages
for role, msg in st.session_state.messages:
    st.chat_message(role).markdown(msg)

# User input box
user_input = st.chat_input("Ask me anything...")

# Handle interaction
if user_input:
    st.chat_message("user").markdown(user_input)

    # Get response from model
    response = chain.invoke({
        "context": st.session_state.context,
        "question": user_input
    })

    st.chat_message("assistant").markdown(response)
    with st.sidebar:
        st.write(response)

    # Update session state
    st.session_state.context += f"\nUser: {user_input}\nAI: {response}\n"
    st.session_state.messages.append(("user", user_input))
    st.session_state.messages.append(("assistant", response))
