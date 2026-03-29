import streamlit as st
import joblib

# Load model
model = joblib.load("fake_news_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Page config
st.set_page_config(page_title="InfoTrust", layout="wide")

# Title
st.markdown("""
    <h1 style='text-align: center; background-color:#2c3e50; color:white; padding:20px;'>
    InfoTrust News Credibility Analysis
    </h1>
""", unsafe_allow_html=True)

# Tabs (REAL WORKING NAVIGATION)
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🏠 Home",
    "📝 Summarize",
    "❓ QA",
    "🔗 URL Analysis",
    "🐦 Twitter/X Analysis"
])

# HOME
with tab1:
    st.subheader("Welcome to InfoTrust")
    st.write("""
    This system analyzes news credibility using:
    - NLP techniques
    - Machine learning
    - Credibility scoring
    - Explainable AI
    """)

# SUMMARIZE
with tab2:
    st.subheader("News Summarization")
    text = st.text_area("Enter news to summarize")

    if st.button("Summarize"):
        st.info("Summary feature demo (placeholder)")
        st.write(text[:200] + "...")

# QA
with tab3:
    st.subheader("Question Answering")
    question = st.text_input("Ask a question")

    if st.button("Get Answer"):
        st.info("QA feature demo (placeholder)")
        st.write("Answer: This is a demo response.")

# URL ANALYSIS
with tab4:
    st.subheader("URL Analysis")
    url = st.text_input("Enter URL")

    if st.button("Analyze URL"):
        st.info("URL analysis demo")
        st.write("Credibility Score: 75%")

# TWITTER/X
with tab5:
    st.subheader("Twitter/X Analysis")
    keyword = st.text_input("Enter keyword")

    if st.button("Analyze Twitter"):
        st.info("Twitter sentiment demo")
        st.write("Sentiment: Positive")

# MAIN ANALYSIS (Keep your model here)
st.markdown("---")
st.subheader("🔍 News Text Analysis")

news_text = st.text_area("Enter News Content")

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
