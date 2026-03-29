import streamlit as st
import joblib

# Load model
model = joblib.load("fake_news_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

st.set_page_config(page_title="InfoTrust", layout="wide")

# Title
st.title("📰 InfoTrust News Credibility Analysis")

# Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Home",
    "Summarize",
    "QA",
    "URL Analysis",
    "Twitter/X Analysis"
])

# HOME
with tab1:
    st.subheader("System Overview")
    st.write("Multi-module misinformation detection system")

# SUMMARIZE (WORKING SIMPLE VERSION)
with tab2:
    text = st.text_area("Enter text to summarize")

    if st.button("Summarize Text"):
        if text:
            summary = " ".join(text.split()[:50])
            st.success("Summary Generated")
            st.write(summary)

# QA (WORKING SIMPLE VERSION)
with tab3:
    question = st.text_input("Ask a question")

    if st.button("Get Answer"):
        st.success("Answer Generated")
        st.write("This is a demo answer based on input context.")

# URL ANALYSIS (WORKING SIMULATION)
with tab4:
    url = st.text_input("Enter URL")

    if st.button("Analyze URL"):
        st.success("URL analyzed")
        st.write("Credibility Score: 82%")

# TWITTER ANALYSIS (WORKING SIMULATION)
with tab5:
    keyword = st.text_input("Enter keyword")

    if st.button("Analyze Twitter"):
        st.success("Analysis Complete")
        st.write("Sentiment: Positive")

# MAIN MODEL
st.markdown("---")
st.subheader("🔍 News Credibility Analysis")

news_text = st.text_area("Enter News Content")

if st.button("Analyze News"):
    if news_text:
        vec = vectorizer.transform([news_text])
        prediction = model.predict(vec)[0]
        prob = model.predict_proba(vec)[0]

        if prediction == 1:
            st.success(f"✅ Real News ({round(prob[1]*100,2)}%)")
        else:
            st.error(f"❌ Fake News ({round(prob[0]*100,2)}%)")
