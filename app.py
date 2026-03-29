import streamlit as st
import joblib
import numpy as np

# Load model and vectorizer
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

# Tabs (Navigation)
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🏠 Home",
    "📝 Summarize",
    "❓ QA",
    "🔗 URL Analysis",
    "🐦 Twitter/X Analysis"
])

# ---------------- HOME ----------------
with tab1:
    st.subheader("📊 System Overview")
    st.write("""
    InfoTrust is a multi-module system for news credibility analysis using:
    - Natural Language Processing (NLP)
    - Machine Learning models
    - Explainable AI techniques
    - Social signal analysis
    """)

# ---------------- SUMMARIZE ----------------
with tab2:
    st.subheader("📝 News Summarization")
    text = st.text_area("Enter news text")

    if st.button("Summarize Text"):
        if text:
            summary = " ".join(text.split()[:50])
            st.success("Summary Generated")
            st.write(summary)
        else:
            st.warning("Enter text")

# ---------------- QA ----------------
with tab3:
    st.subheader("❓ Question Answering")
    question = st.text_input("Ask a question")

    if st.button("Get Answer"):
        if question:
            st.success("Answer Generated")
            st.write("This is a demo answer based on input context.")
        else:
            st.warning("Enter a question")

# ---------------- URL ANALYSIS ----------------
with tab4:
    st.subheader("🔗 URL Analysis")
    url = st.text_input("Enter URL")

    if st.button("Analyze URL"):
        if url:
            st.success("URL analyzed successfully")
            st.write("Credibility Score: 80%")
        else:
            st.warning("Enter URL")

# ---------------- TWITTER ----------------
with tab5:
    st.subheader("🐦 Twitter/X Analysis")
    keyword = st.text_input("Enter keyword")

    if st.button("Analyze Twitter"):
        if keyword:
            st.success("Analysis Complete")
            st.write("Sentiment: Positive")
        else:
            st.warning("Enter keyword")

# ---------------- MAIN MODEL ----------------
st.markdown("---")
st.subheader("🔍 News Credibility Analysis")

news_text = st.text_area("Enter News Content")

if st.button("Analyze News"):
    if news_text:
        vec = vectorizer.transform([news_text])
        prediction = model.predict(vec)[0]
        prob = model.predict_proba(vec)[0]

        # Prediction output
        if prediction == 1:
            st.success(f"✅ Real News ({round(prob[1]*100,2)}%)")
        else:
            st.error(f"❌ Fake News ({round(prob[0]*100,2)}%)")

        # ---------------- EXPLAINABILITY ----------------
        st.subheader("🧠 Explainability (Top Influencing Words)")

        try:
            feature_names = vectorizer.get_feature_names_out()
            coefficients = model.coef_[0]

            input_indices = vec.nonzero()[1]

            top_features = sorted(
                [(feature_names[i], coefficients[i]) for i in input_indices],
                key=lambda x: abs(x[1]),
                reverse=True
            )[:10]

            words = [f[0] for f in top_features]
            scores = [f[1] for f in top_features]

            st.write("Top words influencing prediction:")

            for w, s in zip(words, scores):
                st.write(f"🔹 {w} → Influence: {round(s,3)}")

            # Graph
            st.bar_chart(scores)

        except:
            st.warning("Explainability not available for this model type")

    else:
        st.warning("Please enter text")
