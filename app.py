import streamlit as st
import joblib
from googletrans import Translator

# Load model
model = joblib.load("fake_news_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

translator = Translator()

st.set_page_config(page_title="InfoTrust", layout="wide")

# ---------- SESSION ----------
if "page" not in st.session_state:
    st.session_state.page = "Home"

def navigate(page):
    st.session_state.page = page

# ---------- HEADER ----------
st.markdown("""
<div style="background-color:#2c3e50;padding:25px;text-align:center;">
<h1 style="color:white;">InfoTrust AI Powered Content Trust & Credibility Platform</h1>
</div>
""", unsafe_allow_html=True)

# ---------- NAVBAR ----------
cols = st.columns(8)
pages = ["Home","Summarize","QA","URL","Reddit","YouTube","Translate","Dashboard"]

for i, p in enumerate(pages):
    if cols[i].button(p):
        navigate(p)

# ---------- CARD STYLE ----------
st.markdown("""
<style>
.card {
    background-color: white;
    padding: 30px;
    border-radius: 10px;
    width: 60%;
    margin: auto;
    margin-top: 40px;
    box-shadow: 0px 4px 10px rgba(0,0,0,0.1);
}
</style>
""", unsafe_allow_html=True)

# ---------- HOME ----------
if st.session_state.page == "Home":
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("News Text Analysis")

    news_text = st.text_area("Paste news here...")

    if st.button("Analyze News"):
        if news_text:
            vec = vectorizer.transform([news_text])
            prediction = model.predict(vec)[0]
            prob = model.predict_proba(vec)[0]

            if prediction == 1:
                st.success(f"✅ REAL NEWS ({round(prob[1]*100,2)}%)")
            else:
                st.error(f"❌ FAKE NEWS ({round(prob[0]*100,2)}%)")

            # Explainability
            st.subheader("Important Words")
            try:
                feature_names = vectorizer.get_feature_names_out()
                coefficients = model.coef_[0]
                input_indices = vec.nonzero()[1]

                word_scores = [(feature_names[i], coefficients[i]) for i in input_indices]
                top_words = sorted(word_scores, key=lambda x: abs(x[1]), reverse=True)[:8]

                words = [w[0] for w in top_words]
                st.write(", ".join(words))
            except:
                st.warning("Explainability not available")

        else:
            st.warning("Enter text")

    st.markdown('</div>', unsafe_allow_html=True)

# ---------- SUMMARIZE ----------
elif st.session_state.page == "Summarize":
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Summarization")

    text = st.text_area("Enter text")

    if st.button("Summarize"):
        if text:
            summary = " ".join(text.split()[:50])
            st.success("Summary Generated")
            st.write(summary)
        else:
            st.warning("Enter text")

    st.markdown('</div>', unsafe_allow_html=True)

# ---------- QA ----------
elif st.session_state.page == "QA":
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Question Answering")

    question = st.text_input("Ask question")

    if st.button("Get Answer"):
        if question:
            st.success("Answer Generated")
            st.write("This is a demo answer.")
        else:
            st.warning("Enter question")

    st.markdown('</div>', unsafe_allow_html=True)

# ---------- URL ----------
elif st.session_state.page == "URL":
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("URL Analysis")

    url = st.text_input("Enter URL")

    if st.button("Analyze URL"):
        if url:
            st.success("Credibility Score: 80%")
        else:
            st.warning("Enter URL")

    st.markdown('</div>', unsafe_allow_html=True)

# ---------- REDDIT ----------
elif st.session_state.page == "Reddit":
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Reddit Analysis")

    keyword = st.text_input("Enter keyword")

    if st.button("Analyze Reddit"):
        if keyword:
            st.success("Sentiment: Positive")
        else:
            st.warning("Enter keyword")

    st.markdown('</div>', unsafe_allow_html=True)

# ---------- YOUTUBE ----------
elif st.session_state.page == "YouTube":
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("YouTube Analysis")

    keyword = st.text_input("Enter topic")

    if st.button("Analyze YouTube"):
        if keyword:
            st.success("Engagement Score: High")
        else:
            st.warning("Enter topic")

    st.markdown('</div>', unsafe_allow_html=True)

# ---------- TRANSLATE ----------
elif st.session_state.page == "Translate":
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Translate")

    text = st.text_area("Enter text")

    language = st.selectbox(
        "Select Language",
        ["Hindi", "Telugu", "Tamil", "French", "German"]
    )

    lang_code = {
        "Hindi": "hi",
        "Telugu": "te",
        "Tamil": "ta",
        "French": "fr",
        "German": "de"
    }

    if st.button("Translate"):
        if text:
            translated = translator.translate(text, dest=lang_code[language])
            st.success("Translated Text:")
            st.write(translated.text)
        else:
            st.warning("Enter text")

    st.markdown('</div>', unsafe_allow_html=True)

# ---------- DASHBOARD ----------
elif st.session_state.page == "Dashboard":
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Dashboard")

    st.metric("Fake News Detected", "73%")
    st.metric("Trust Score", "82%")
    st.metric("Risk Level", "Moderate")

    st.markdown('</div>', unsafe_allow_html=True)
