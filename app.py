import streamlit as st
import joblib

# Load model
model = joblib.load("fake_news_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Page config
st.set_page_config(page_title="InfoTrust", layout="wide")

# Custom CSS (IMPORTANT)
st.markdown("""
    <style>
    body {
        background-color: #f5f7fa;
    }

    .title {
        text-align: center;
        font-size: 40px;
        font-weight: bold;
        color: white;
        padding: 20px;
        background-color: #2c3e50;
    }

    .navbar {
        display: flex;
        justify-content: center;
        background-color: #34495e;
        padding: 10px;
    }

    .nav-item {
        margin: 0 20px;
        color: white;
        font-weight: bold;
    }

    .card {
        background-color: white;
        padding: 30px;
        border-radius: 10px;
        width: 60%;
        margin: auto;
        margin-top: 50px;
        box-shadow: 0px 4px 10px rgba(0,0,0,0.1);
    }

    textarea {
        width: 100%;
        height: 150px;
    }

    </style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="title">InfoTrust News Credibility Analysis</div>', unsafe_allow_html=True)

# Navbar
st.markdown("""
<div class="navbar">
    <div class="nav-item">Home</div>
    <div class="nav-item">Summarize</div>
    <div class="nav-item">QA</div>
    <div class="nav-item">URL Analysis</div>
    <div class="nav-item">Twitter/X Analysis</div>
</div>
""", unsafe_allow_html=True)

# Main Card
st.markdown('<div class="card">', unsafe_allow_html=True)

st.subheader("News Text Analysis")

news_text = st.text_area("Enter News Text")

if st.button("Analyze News"):
    if news_text:
        vec = vectorizer.transform([news_text])
        prediction = model.predict(vec)[0]
        prob = model.predict_proba(vec)[0]

        if prediction == 1:
            st.success(f"✅ Real News (Confidence: {round(prob[1]*100,2)}%)")
        else:
            st.error(f"❌ Fake News (Confidence: {round(prob[0]*100,2)}%)")
    else:
        st.warning("Please enter text")

st.markdown('</div>', unsafe_allow_html=True)
