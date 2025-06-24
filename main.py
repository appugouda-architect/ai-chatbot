from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
import streamlit as st

model = OllamaLLM(model="llama3.2")


template = """
    You are a helpful assistant. Answer the following question based on the provided context.
    {context}
    Question: {question}
    Answer:"""
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model


def handle_conversation():
    context = ""

    while True:
        print("Welcome to AI Chatbot: Type 'q' to exit:")
        user_input = input("You:").strip()
        if user_input.lower() == 'q':
            print("Exiting the chatbot. Goodbye!")
            break

        if user_input:
            response = chain.invoke(
                {"context": context, "question": user_input})
            print(f"Bot: {response}")
            context += f"\nUser: {user_input}\nAI: {response}\n"


if __name__ == "__main__":
    handle_conversation()
