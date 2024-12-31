import streamlit as st
import google.generativeai as genai
import os

def chatbot():
    # Configure page settings
    st.title("🤖 EduResume AI Assistant")
    
    # Configure Google Gemini API
    try:
        GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
        genai.configure(api_key=GOOGLE_API_KEY)
        model = genai.GenerativeModel('gemini-pro')
    except Exception as e:
        st.error("Please configure the Google API Key in your secrets.toml file")
        return

    # Initialize chat session
    if "chat_session" not in st.session_state:
        st.session_state.chat_session = model.start_chat(history=[])

    # Initialize message history
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []

    # Display chat messages
    for message in st.session_state.chat_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask me anything about resumes, jobs, or interviews!"):
        # Display user message
        st.chat_message("user").markdown(prompt)
        
        # Add user message to chat history
        st.session_state.chat_messages.append({"role": "user", "content": prompt})
        
        try:
            # Get Gemini response
            response = st.session_state.chat_session.send_message(prompt)
            
            # Display assistant response
            with st.chat_message("assistant"):
                st.markdown(response.text)
                
            # Add assistant response to chat history
            st.session_state.chat_messages.append({"role": "assistant", "content": response.text})
                
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

    # Add a clear chat button
    if st.button("Clear Chat"):
        st.session_state.chat_messages = []
        st.session_state.chat_session = model.start_chat(history=[])
        st.rerun()

# This line is important - it allows the function to be imported
__all__ = ['chatbot']