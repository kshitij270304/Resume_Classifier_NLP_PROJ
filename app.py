import streamlit as st
import pickle
import docx
import PyPDF2
import re
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer


# ---------------- LOAD MODELS ----------------
svc_model = pickle.load(open('clf.pkl', 'rb'))
tfidf = pickle.load(open('tfidf.pkl', 'rb'))
le = pickle.load(open('encoder.pkl', 'rb'))


# ---------------- TEXT CLEANING ----------------
def cleanResume(txt):
    cleanText = re.sub('http\S+\s', ' ', txt)
    cleanText = re.sub('RT|cc', ' ', cleanText)
    cleanText = re.sub('#\S+\s', ' ', cleanText)
    cleanText = re.sub('@\S+', ' ', cleanText)
    cleanText = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', cleanText)
    cleanText = re.sub(r'[^\x00-\x7f]', ' ', cleanText)
    cleanText = re.sub('\s+', ' ', cleanText)
    return cleanText.strip()


# ---------------- FILE EXTRACTION ----------------
def extract_text_from_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ''
    for page in pdf_reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text
    return text


def extract_text_from_docx(file):
    doc = docx.Document(file)
    return '\n'.join([para.text for para in doc.paragraphs])


def extract_text_from_txt(file):
    try:
        return file.read().decode('utf-8')
    except UnicodeDecodeError:
        return file.read().decode('latin-1')


def handle_file_upload(uploaded_file):
    ext = uploaded_file.name.split('.')[-1].lower()
    if ext == 'pdf':
        return extract_text_from_pdf(uploaded_file)
    elif ext == 'docx':
        return extract_text_from_docx(uploaded_file)
    elif ext == 'txt':
        return extract_text_from_txt(uploaded_file)
    else:
        raise ValueError("Unsupported file format")


# ---------------- RESUME CLASSIFIER ----------------
def pred(input_resume):
    cleaned_text = cleanResume(input_resume)
    vectorized_text = tfidf.transform([cleaned_text]).toarray()

    # Predict category
    prediction = svc_model.predict(vectorized_text)
    category = le.inverse_transform(prediction)[0]

    # Determine confidence (model-based ATS heuristic)
    score = None
    try:
        if hasattr(svc_model, "predict_proba"):
            probs = svc_model.predict_proba(vectorized_text)
            score = float(probs.max() * 100)
        elif hasattr(svc_model, "decision_function"):
            df = svc_model.decision_function(vectorized_text)
            df = np.atleast_2d(df)
            exps = np.exp(df - np.max(df, axis=1, keepdims=True))
            probs = exps / exps.sum(axis=1, keepdims=True)
            score = float(probs.max() * 100)
    except Exception:
        score = None

    score = round(score, 2) if score is not None else None
    return category, score


# ---------------- ATS SCORE ----------------
def calculate_ats_score(resume_text, jd_text):
    resume_clean = cleanResume(resume_text)
    jd_clean = cleanResume(jd_text)

    documents = [resume_clean, jd_clean]

    ats_vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    vectors = ats_vectorizer.fit_transform(documents)

    similarity = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
    return round(similarity * 100, 2)


# ---------------- STREAMLIT APP ----------------
def main():
    st.set_page_config(
        page_title="Resume ATS & Category Predictor",
        page_icon="📄",
        layout="wide"
    )

    st.title("Resume Category Prediction & ATS Scoring")
    st.markdown(
        "Upload a resume and paste a job description to get **job category** and **ATS match score**."
    )

    # Job Description input
    jd_text = st.text_area(
        "Job Description",
        placeholder="Paste the job description here...",
        height=200
    )

    # Resume upload
    uploaded_file = st.file_uploader(
        "Upload Resume (PDF / DOCX / TXT)",
        type=["pdf", "docx", "txt"]
    )

    if uploaded_file is not None:
        try:
            resume_text = handle_file_upload(uploaded_file)
            st.success("Resume text extracted successfully!")

            if st.checkbox("Show extracted resume text"):
                st.text_area(
                    "Extracted Resume Text",
                    resume_text,
                    height=300
                )

            # Category prediction
            st.subheader("Predicted Job Category")
            category, model_score = pred(resume_text)
            st.write(f"**{category}**")
            # if model_score is not None:
            #     st.metric("Model Confidence (heuristic ATS)", f"{model_score}%")

            # ATS Score
            st.subheader("ATS Match Score")

            if jd_text.strip():
                ats_score = calculate_ats_score(resume_text, jd_text)

                st.metric("Resume–JD Match", f"{ats_score}%")

                if ats_score >= 75:
                    st.success("Excellent match! Resume is highly relevant.")
                elif ats_score >= 50:
                    st.warning("Good match, but improvements are possible.")
                else:
                    st.error("Low match. Resume needs optimization.")
            else:
                st.info("Paste a job description to calculate ATS score. Model confidence is shown above.")

        except Exception as e:
            st.error(f"Error processing the file: {str(e)}")


if __name__ == "__main__":
    main()
