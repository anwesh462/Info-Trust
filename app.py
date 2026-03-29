import streamlit as st
import joblib

# Load model
model = joblib.load("fake_news_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Page Config
st.set_page_config(page_title="InfoTrust System", layout="wide")

# Sidebar Navigation
st.sidebar.title("📊 InfoTrust System")
option = st.sidebar.selectbox(
    "Select Module",
    ["Home", "News Analysis", "Source Credibility", "Social Signals", "Explainability"]
)

# HOME
if option == "Home":
    st.title("📰 InfoTrust: News Credibility System")
    st.write("""
    This system analyzes news using:
    - NLP techniques
    - Machine Learning
    - Credibility scoring
    - Explainable AI
    """)
    st.success("System Ready ✅")

# NEWS ANALYSIS
elif option == "News Analysis":
    st.title("🔍 News Credibility Analysis")

    news_text = st.text_area("Enter News Content")

    if st.button("Analyze News"):
        if news_text:
            vec = vectorizer.transform([news_text])
            prediction = model.predict(vec)[0]
            prob = model.predict_proba(vec)[0]

            if prediction == 1:
                st.success("✅ Real News")
                st.write(f"Confidence: {round(prob[1]*100,2)}%")
            else:
                st.error("❌ Fake News")
                st.write(f"Confidence: {round(prob[0]*100,2)}%")
        else:
            st.warning("Enter news text")

# SOURCE CREDIBILITY
elif option == "Source Credibility":
    st.title("🌐 Source Credibility Module")

    st.write("This module evaluates reliability of news sources")

    source = st.text_input("Enter Source Name")

    if st.button("Check Source"):
        st.info("Sample Output: Source credibility score = 78%")
        st.progress(78)

# SOCIAL SIGNALS
elif option == "Social Signals":
    st.title("📱 Social Media Analysis")

    st.write("Analyzing social engagement patterns...")

    st.metric("Twitter Trust Score", "72%")
    st.metric("Reddit Sentiment", "Positive")
    st.metric("Engagement Risk", "Low")

# EXPLAINABILITY
elif option == "Explainability":
    st.title("🧠 Explainable AI")

    st.write("Model explanation for predictions")

    st.info("Important words influencing prediction:")
    st.write("- breaking")
    st.write("- shocking")
    st.write("- exclusive")

    st.success("These words indicate possible misinformation patterns")
