import streamlit as st
import pickle
import time
import re

# 1. Page Configuration
st.set_page_config(
    page_title="Email Spam Detector",
    page_icon="📧",
    layout="wide"
)

# 2. Custom UI Styling (CSS)
st.markdown("""
<style>
.main {
    background-color:#f5f7fa;
}
.title {
    font-size:45px;
    font-weight:bold;
    text-align:center;
    color:#1f4e79;
}
.subtitle {
    font-size:20px;
    text-align:center;
    color:gray;
}
.stTextArea textarea {
    border-radius:15px;
    border:2px solid #1f77b4;
    font-size:17px;
}
.result {
    font-size:30px;
    text-align:center;
    font-weight:bold;
    padding:15px;
    border-radius:15px;
}
.footer {
    text-align:center;
    color:gray;
    font-size:14px;
}
</style>
""", unsafe_allow_html=True)


# 3. Safe Model and Vectorizer Loader
@st.cache_resource
def load_ml_assets():
    # Loads the pre-fitted vectorizer and model saved during your Colab phase
    with open("vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    with open("EmailSpamDetector.pkl", "rb") as f:
        model = pickle.load(f)
    return vectorizer, model

try:
    vectorizer, model = load_ml_assets()
except FileNotFoundError as e:
    st.error("⚠️ Error: 'vectorizer.pkl' or 'EmailSpamDetector.pkl' not found. Please place them in the same folder as this script.")
    st.stop()


# 4. Header UI
st.markdown("<div class='title'>📧 Email Spam Detection</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='subtitle'>Detect whether an Email is Spam or Not using Machine Learning</div>",
    unsafe_allow_html=True
)
st.write("")


# 5. Main Content Layout
col1, col2 = st.columns([2, 1])

with col1:
    # Capturing input inside unique variable to prevent duplicates
    email_text = st.text_area(
        "✉️ Enter Email Message",
        height=250,
        placeholder="Paste your email content here..."
    )

with col2:
    st.info("""
### 📌 Model Details

**Algorithm:** Multinomial Naive Bayes

**Features**
- Spam Detection
- Fast Prediction
- User Friendly
- Machine Learning Based
""")


# 6. Prediction Execution
if st.button("🚀 Detect Email", use_container_width=True):
    if not email_text.strip():
        st.warning("Please enter some text to analyze.")
    else:
        with st.spinner("Analyzing Email..."):
            time.sleep(0.5)  # Soft delay for UI experience
            
            # Simple text baseline cleaning to align with standard NLP steps
            cleaned_text = email_text.lower().strip()
            
            # Use the loaded, pre-fitted vectorizer to transform text
            transformed_data = vectorizer.transform([cleaned_text])
            
            # Run prediction
            prediction = model.predict(transformed_data)[0]

        st.balloons()

        # Render Results HTML
        if prediction == 1:
            st.markdown(
                """
                <div class='result' style='background:#ffebee;color:#c62828;'>
                🚨 SPAM EMAIL Detected!
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                """
                <div class='result' style='background:#e8f5e9;color:#2e7d32;'>
                ✅ SAFE EMAIL (Not Spam)
                </div>
                """,
                unsafe_allow_html=True
            )


# 7. Sidebar Information
st.sidebar.title("📊 Dashboard")
st.sidebar.success("Machine Learning Project")
st.sidebar.markdown("---")
st.sidebar.write("### Technologies Used")
st.sidebar.write("""
- Python
- Scikit-Learn
- Streamlit
- Pandas
- NumPy
""")
st.sidebar.markdown("---")
st.sidebar.write("### Project Info")
st.sidebar.write(
    """
    This app reads inbound text strings and maps them across the pre-calculated weights of a Naive Bayes model to determine safety metrics.
    """
)

st.markdown("---")
st.markdown("<div class='footer'>Made with ❤️ using Streamlit</div>", unsafe_allow_html=True)