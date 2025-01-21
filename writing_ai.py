import requests
import streamlit as st
from typing import Dict, Any

class LLMAgent:
    def __init__(self, model="llama3.2"):
        """Initialize the AI agent."""
        self.base_url = "http://localhost:11434/api"  # Ollama's default port
        self.model = model

    def generate_response(self, prompt: str, temperature: float = 0.7) -> str:
        """Generate response from Ollama."""
        try:
            response = requests.post(
                f"{self.base_url}/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "temperature": temperature,
                    "stream": False
                },
                timeout=30
            )
            response.raise_for_status()
            return response.json()["response"]
        except requests.exceptions.RequestException as e:
            return f"Error connecting to Ollama: {str(e)}"
        except Exception as e:
            return f"Unexpected error: {str(e)}"

def generate_essay_prompt(topic: str) -> str:
    """Generate essay prompt."""
    return f"""Write a comprehensive essay on the topic: {topic}.
    Include an introduction, body paragraphs, and conclusion.
    Make it informative and engaging."""

def generate_poem_prompt(topic: str) -> str:
    """Generate poem prompt."""
    return f"""Write a beautiful poem about {topic}.
    Be creative and use metaphors and imagery."""

def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="AI Writing Assistant",
        page_icon="✍️",
        layout="wide"
    )

    # Initialize LLM agent
    agent = LLMAgent()

    # Title and description
    st.title("✍️ AI Writing Assistant")
    st.markdown("Generate essays and poems using LLAMA3!")

    # Sidebar for settings
    with st.sidebar:
        st.header("⚙️ Settings")
        temperature = st.slider(
            "🌡️ Temperature",
            min_value=0.1,
            max_value=1.0,
            value=0.7,
            step=0.1,
            help="Higher values make the output more creative, lower values make it more focused"
        )

    # Create two columns for essay and poem
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📝 Essay Generator")
        essay_topic = st.text_input(
            "Enter essay topic",
            key="essay",
            placeholder="e.g., Climate Change"
        )
        if essay_topic:
            with st.spinner("Generating essay..."):
                prompt = generate_essay_prompt(essay_topic)
                response = agent.generate_response(prompt, temperature)
                st.markdown("### Your Essay:")
                st.markdown(response)

    with col2:
        st.subheader("🎨 Poem Generator")
        poem_topic = st.text_input(
            "Enter poem topic",
            key="poem",
            placeholder="e.g., Nature"
        )
        if poem_topic:
            with st.spinner("Creating poem..."):
                prompt = generate_poem_prompt(poem_topic)
                response = agent.generate_response(prompt, temperature)
                st.markdown("### Your Poem:")
                st.markdown(response)

    # Footer
    st.markdown("---")
    st.markdown("Made with ❤️ using LLAMA3")

if __name__ == "__main__":
    main()