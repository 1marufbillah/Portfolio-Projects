import streamlit as st
import requests
import json
from typing import Dict, Any

class LocalAIAgent:
    def __init__(self, model="llama3.2"):
        self.base_url = "http://localhost:11434/api"
        self.model = model

    def generate_response(self, prompt: str, temperature: float = 0.7) -> str:
        try:
            response = requests.post(
                f"{self.base_url}/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "temperature": temperature,
                    "stream": False
                }
            )
            if response.status_code == 200:
                return response.json()["response"]
            else:
                return f"Error: {response.status_code}"
        except Exception as e:
            return f"Error connecting to Ollama: {str(e)}"

def initialize_session_state():
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'agent' not in st.session_state:
        st.session_state.agent = LocalAIAgent()

def main():
    st.title("Local AI Assistant")

    # Initialize session state
    initialize_session_state()

    # Sidebar controls
    with st.sidebar:
        st.header("Settings")
        temperature = st.slider("Temperature", 0.0, 1.0, 0.7, 0.1)
        if st.button("Clear Chat"):
            st.session_state.messages = []

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("What would you like to know?"):
        # Add user message to chat
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate AI response
        with st.chat_message("assistant"):
            response = st.session_state.agent.generate_response(prompt, temperature)
            st.markdown(response)

        # Add AI response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()

