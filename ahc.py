import streamlit as st
from transformers import pipeline
import nltk

# Download necessary NLTK data (quiet to avoid clutter)
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# Load a small pre-trained Hugging Face model (will download on first run)
# Note: This is a general text generator; it is NOT a medical model.
chatbot = pipeline("text-generation", model="distilgpt2")

# Simple healthcare-specific response logic
def healthcare_chatbot(user_input: str) -> str:
    text = (user_input or "").strip().lower()

    if "symptom" in text:
        return ("It sounds like you’re describing symptoms. "
                "For accurate guidance, please consult a qualified doctor.")
    elif "appointment" in text:
        return "Would you like help finding or scheduling a doctor’s appointment?"
    elif "medication" in text:
        return ("Please take prescribed medicines as directed. "
                "If you have concerns or side effects, consult your doctor.")
    elif "emergency" in text:
        return ("If this is an emergency, call your local emergency number immediately.")
    else:
        # For other inputs, use the model to generate a response
        # Keep generation modest to avoid very long outputs
        out = chatbot(user_input, max_length=200, num_return_sequences=1)
        return out[0]["generated_text"]

def main():
    st.title("🩺 AI Healthcare Assistant Chatbot")
    st.write("Hello! I'm here to help with general health questions.")
    st.info("Important: This app does not provide medical diagnosis or treatment. For medical advice, consult a professional. In emergencies, call your local emergency number.")

    user_input = st.text_input("How can I assist you today?", "")

    if st.button("Submit"):
        if user_input.strip():
            st.write("User:", user_input)
            with st.spinner("Processing your query..."):
                response = healthcare_chatbot(user_input)
            st.write("Healthcare Assistant:", response)
        else:
            st.warning("Please enter a query.")

if __name__ == "__main__":
    main()