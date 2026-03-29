import streamlit as st
import joblib
import json

model = joblib.load("fake_news_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# -------- API MODE --------
query_params = st.query_params

if "api" in query_params:
    text = query_params.get("text", "")

    if text:
        vec = vectorizer.transform([text])
        prediction = model.predict(vec)[0]
        prob = model.predict_proba(vec)[0]

        result = {
            "prediction": "REAL" if prediction == 1 else "FAKE",
            "confidence": float(max(prob))
        }

        st.write(result)

    st.stop()
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
    - NLP techniques
    - Machine Learning models
    - Explainable AI
    """)

# ---------------- SUMMARIZE ----------------
with tab2:
    st.subheader("📝 News Summarization")
    text = st.text_area("Enter text to summarize")

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
            st.write("This is a demo answer.")
        else:
            st.warning("Enter a question")

# ---------------- URL ANALYSIS ----------------
with tab4:
    st.subheader("🔗 URL Analysis")
    url = st.text_input("Enter URL")

    if st.button("Analyze URL"):
        if url:
            st.success("URL analyzed")
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

        # Prediction result
        if prediction == 1:
            st.success(f"✅ Real News ({round(prob[1]*100,2)}%)")
        else:
            st.error(f"❌ Fake News ({round(prob[0]*100,2)}%)")

        # -------- SIMPLE EXPLAINABILITY --------
        st.subheader("🧠 Important Words")

        try:
            feature_names = vectorizer.get_feature_names_out()
            coefficients = model.coef_[0]

            input_indices = vec.nonzero()[1]

            # Get top words
            word_scores = [
                (feature_names[i], coefficients[i]) for i in input_indices
            ]

            top_words = sorted(
                word_scores,
                key=lambda x: abs(x[1]),
                reverse=True
            )[:8]

            words_only = [w[0] for w in top_words]

            st.write(", ".join(words_only))

        except:
            st.warning("Explainability not available")

    else:
        st.warning("Please enter text")
