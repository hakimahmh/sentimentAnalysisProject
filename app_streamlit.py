import streamlit as st
import requests

st.title("Sentiment Analysis Dashboard")

text = st.text_area("Enter Text")
if st.button("Analyze"):
    response = requests.post("http://127.0.0.1:5000/analyze", json={"text": text})
    result = response.json()

    st.write(f"Sentiment: {result['sentiment']} (Score: {result['sentiment_score']:.2f})")
    st.write(f"Emotion: {result['emotion']} (Score: {result['emotion_score']:.2f})")
