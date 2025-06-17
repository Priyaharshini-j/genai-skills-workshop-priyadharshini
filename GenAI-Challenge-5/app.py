# app.py
import streamlit as st
from rag_pipeline import generate_bot_response, is_safe_input

st.set_page_config(page_title="Alaska FAQ Chatbot", page_icon="❄️")
st.title("🤖 Alaska FAQ Chatbot")
st.markdown("Ask anything about Alaska Department of Snow services.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

prompt = st.chat_input("Type your question here...")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            if is_safe_input(prompt) != "YES":
                response_text = "🚫 Rejected: Your question contains sensitive information."
            else:
                response = generate_bot_response(prompt)
                if response.candidates and response.candidates[0].finish_reason == "SAFETY":
                    response_text = "⚠️ Gemini blocked this response due to safety policies."
                elif is_safe_input(response.text.strip()) != "YES":
                    response_text = "🚫 Sorry. The response contains sensitive information and cannot be shown."
                else:
                    response_text = response.text.strip()

            st.markdown(response_text)

        except Exception as e:
            st.markdown(f"❗ Error: {str(e)}")

        st.session_state.messages.append({"role": "assistant", "content": response_text})
