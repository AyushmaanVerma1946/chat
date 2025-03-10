import streamlit as st

# 🔹 Fix: `set_page_config()` must be the first Streamlit command
st.set_page_config(page_title="Medical Chatbot", page_icon="💬", layout="wide")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# =========================================================
# 🚀 Load the fine-tuned model and cache it to reduce loading time
# =========================================================
@st.cache_resource
def load_model():
    """
    Loads the fine-tuned Phi-2 model and tokenizer.
    Uses caching to prevent reloading on every user interaction.
    """
    MODEL_PATH = "../models/phi2_finetuned_model"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, 
        torch_dtype=torch.float16,  
        device_map="auto"  
    )
    model = torch.compile(model)  
    return model, tokenizer

# Load model and tokenizer once
model, tokenizer = load_model()

# =========================================================
# ✨ Response Cleaning & Formatting Function
# =========================================================
def clean_response(response, user_input):
    """
    Cleans and refines the AI's response:
    - Limits response to the first 3 sentences.
    - Ensures a professional medical tone.
    - Adds a disclaimer only for medical-related responses.
    """
    response = response.split(".")[0:3]  
    response = ". ".join(response).strip()

    # Greeting keywords (to avoid adding medical disclaimer)
    greeting_keywords = ["hello", "hi", "hey", "good morning", "good evening", "good afternoon"]

    # If user input is not a greeting, add a medical disclaimer
    if not any(word in user_input.lower() for word in greeting_keywords):
        if "doctor" not in response.lower() and "medical" not in response.lower():
            response += " If symptoms persist, please consult a licensed medical professional."

    return response

# =========================================================
# 🛑 Filter Inappropriate or Non-Medical Topics
# =========================================================
def is_medical_question(user_input):
    """
    Checks if the query is related to health or greetings.
    - Allows medical-related and greeting messages.
    - Rejects violent or inappropriate topics.
    """
    medical_keywords = ["fever", "cough", "cold", "pain", "vomit", "medicine", "treatment", "infection", "allergy", "flu", "injury", "vitamins", "supplements"]
    greeting_keywords = ["hello", "hi", "hey", "good morning", "good evening", "good afternoon"]
    non_medical_keywords = ["violence", "harm", "kill", "suicide", "attack", "fight"]

    # Convert input to lowercase
    user_input = user_input.lower()

    if any(word in user_input for word in non_medical_keywords):
        return "I'm sorry, but I cannot provide assistance on this topic."

    if any(word in user_input for word in greeting_keywords):
        return None  # Safe to answer

    if any(word in user_input for word in medical_keywords):
        return None  # Safe to answer

    return "I'm designed to provide medical guidance. Please ask about health-related topics or say hello!"

# =========================================================
# 💬 Generate AI Response Function
# =========================================================
def generate_response(user_input):
    """
    Generates an AI response while handling greetings, farewells, and thanks separately.
    """
    st.session_state.chat_history = st.session_state.chat_history[-5:]

    # Define keyword categories
    greeting_keywords = ["hello", "hi", "hey", "good morning", "good evening", "good afternoon"]
    farewell_keywords = ["bye", "goodbye", "see you", "take care"]
    thanks_keywords = ["thank you", "thanks", "appreciate it"]

    # Convert user input to lowercase for better matching
    user_input_lower = user_input.lower()

    # Handle greetings
    if any(word in user_input_lower for word in greeting_keywords):
        response = "Hello! How can I assist you today?"  

    # Handle farewells
    elif any(word in user_input_lower for word in farewell_keywords):
        response = "Goodbye! Take care and stay healthy. See you next time!"

    # Handle thank-you messages
    elif any(word in user_input_lower for word in thanks_keywords):
        response = "You're very welcome! Let me know if you need any more help."

    else:
        # Check if question is medical or restricted
        filter_message = is_medical_question(user_input)
        if filter_message:
            return filter_message

        # Format input prompt for model
        prompt = f"<start>\nUser: {user_input}\nAI:"

        # Tokenize input text
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}  

        # Generate AI response
        with torch.no_grad():
            output = model.generate(
                **inputs, 
                max_new_tokens=80,  
                temperature=0.5,  
                top_p=0.7,  
                do_sample=True,
                eos_token_id=tokenizer.eos_token_id  
            )

        # Decode & refine AI response
        response = tokenizer.decode(output[0], skip_special_tokens=True)
        response = response.split("AI:")[-1].strip()
        response = clean_response(response, user_input)  

    # Store conversation history
    st.session_state.chat_history.append(("user", user_input))
    st.session_state.chat_history.append(("ai", response))

    return response


# =========================================================
# 🌐 Streamlit UI for Chatbot (Enhanced Version)
# =========================================================

# Sidebar with app info
with st.sidebar:
    st.title("🩺 Medical Chatbot")
    st.write("💡 **AI-powered chatbot for medical guidance.**")
    st.write("🔹 **Ask about health conditions & symptoms.**")
    st.write("⚠️ **Disclaimer:** This chatbot does not replace professional medical advice.")

# Main UI
st.title("💬 Medical Chatbot")
st.write("🤖 **Ask any medical-related questions, and I'll do my best to assist you!**")

# Initialize chat history in session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display chat history with improved UI
st.markdown("<hr>", unsafe_allow_html=True)
chat_container = st.container()

with chat_container:
    for role, text in st.session_state.chat_history:
        if role == "user":
            st.markdown(f'''
                            <div style="
                                text-align: right; 
                                background-color: #006400;  
                                color: white;
                                padding: 10px;
                                border-radius: 10px;
                                margin: 5px;
                                display: inline-block; 
                                max-width: 80%; 
                                word-wrap: break-word;
                            ">
                                <b>You:</b> {text}
                            </div>
                        ''', unsafe_allow_html=True)
        else:
            st.markdown(f'''
                            <div style="
                                text-align: left; 
                                background-color: #333333;  
                                color: white;
                                padding: 10px;
                                border-radius: 10px;
                                margin: 5px;
                                display: inline-block; 
                                max-width: 80%; 
                                word-wrap: break-word;
                            ">
                                <b>Chatbot:</b> {text}
                            </div>
                        ''', unsafe_allow_html=True)

# =========================================================
# 📌 User Input & Send Button
# =========================================================

# Initialize user input in session state
if "user_input" not in st.session_state:
    st.session_state.user_input = ""

# User input box with larger text area
user_input = st.text_area("Your Question:", value=st.session_state.user_input, height=100)

# Custom styled send button
send_button = st.button("🟢 Send", use_container_width=True)

# Generate response on button click
if send_button and user_input.strip():
    response = generate_response(user_input)
    st.session_state.user_input = ""  # Clears text area
    st.rerun()  # Refreshes UI
