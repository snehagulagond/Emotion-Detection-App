import streamlit as st
import joblib
import re
from sklearn.metrics import accuracy_score

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Emotion Detection App",
    page_icon="🧠",
    layout="centered"
)

# ---------------- LOAD MODEL ----------------
model = joblib.load("emotion_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# ---------------- CLEAN TEXT FUNCTION ----------------
def clean_text(text):
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    return text

# ---------------- EMOTION LABELS ----------------
emotion_labels = {
    0: "sadness",
    1: "anger",
    2: "love",
    3: "surprise",
    4: "fear",
    5: "joy"
}

# ---------------- EMOTION EMOJIS ----------------
emotion_emoji = {
    "joy": "😄",
    "sadness": "😢",
    "anger": "😠",
    "fear": "😨",
    "love": "❤️",
    "surprise": "😲",
    "happy": "😊",
    "neutral": "😐"
}

# ---------------- TITLE ----------------
st.title("🧠 Emotion Detection App")
st.markdown("### Detect emotions from text using Machine Learning")

st.write("---")

# ---------------- USER INPUT ----------------
user_input = st.text_area(
    "✍️ Enter your text here:",
    height=150,
    placeholder="Type something like: I am very happy today!"
)

# ---------------- BUTTON ----------------
if st.button("🔍 Detect Emotion"):

    if user_input.strip() == "":
        st.warning("⚠️ Please enter some text.")

    else:
        # Clean text
        cleaned_text = clean_text(user_input)

        # Convert text into vectors
        vectorized_text = vectorizer.transform([cleaned_text])

        # Predict emotion
        prediction = model.predict(vectorized_text)[0]

        # Convert numeric label to emotion name
        emotion = emotion_labels.get(prediction, "Unknown")

        # Get emoji
        emoji = emotion_emoji.get(emotion, "🙂")

        # Display result
        st.success(f"### Predicted Emotion: {emotion.upper()} {emoji}")

        # Confidence score
        if hasattr(model, "predict_proba"):
            confidence = model.predict_proba(vectorized_text).max() * 100
            st.info(f"Confidence Score: {confidence:.2f}%")

# ---------------- SIDEBAR ----------------
st.sidebar.title("📌 About")

st.sidebar.info("""
This Emotion Detection App uses:

- Machine Learning
- NLP (Natural Language Processing)
- TF-IDF Vectorization
- Logistic Regression
- Streamlit

Built using Python ❤️
""")

# ---------------- FOOTER ----------------
st.write("---")
st.caption("Made with ❤️ using Streamlit")