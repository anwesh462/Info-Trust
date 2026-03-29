import streamlit as st
import joblib

# Load model
model = joblib.load("fake_news_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# UI
st.set_page_config(page_title="InfoTrust", layout="centered")

st.title("📰 InfoTrust - News Credibility Analyzer")

st.markdown("### 🔍 Enter news content below")

# Input
news_text = st.text_area("News Text")

# Prediction
if st.button("Analyze"):
    if news_text.strip() != "":
        vec = vectorizer.transform([news_text])
        prediction = model.predict(vec)[0]
        prob = model.predict_proba(vec)[0]

        if prediction == 1:
            st.success("✅ This news is likely REAL")
            st.write(f"Confidence: {round(prob[1]*100,2)}%")
        else:
            st.error("❌ This news is likely FAKE")
            st.write(f"Confidence: {round(prob[0]*100,2)}%")
    else:
        st.warning("Please enter text")