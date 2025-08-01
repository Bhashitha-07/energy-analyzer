import openai
import streamlit as st

# Use your Streamlit secret manager or API key directly for local testing
openai.api_key = st.secrets["OPENAI_API_KEY"]

def get_ai_response(user_message):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful energy assistant."},
                {"role": "user", "content": user_message}
            ]
        )
        reply = response['choices'][0]['message']['content']
        return reply.strip()
    except Exception as e:
        return f"⚠️ Error: {str(e)}"


