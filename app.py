import streamlit as st
import pickle
import docx
import PyPDF2
import re
import numpy as np
import plotly.graph_objects as go

from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer


# ==================================================
# PAGE CONFIG
# ==================================================

st.set_page_config(
    page_title="Resume ATS & Category Predictor",
    page_icon="📄",
    layout="wide"
)


# ==================================================
# LOAD MODELS
# ==================================================

@st.cache_resource
def load_models():

    svc_model = pickle.load(
        open('clf.pkl', 'rb')
    )

    le = pickle.load(
        open('encoder.pkl', 'rb')
    )

    embedding_model = SentenceTransformer(
        'all-MiniLM-L6-v2'
    )

    return svc_model, le, embedding_model


svc_model, le, embedding_model = load_models()


# ==================================================
# TEXT CLEANING
# ==================================================

def cleanResume(txt):

    cleanText = re.sub(
        r'http\S+\s*',
        ' ',
        txt
    )

    cleanText = re.sub(
        r'RT|cc',
        ' ',
        cleanText
    )

    cleanText = re.sub(
        r'#\S+',
        ' ',
        cleanText
    )

    cleanText = re.sub(
        r'@\S+',
        ' ',
        cleanText
    )

    cleanText = re.sub(
        r'[%s]' % re.escape(
            """!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""
        ),
        ' ',
        cleanText
    )

    cleanText = re.sub(
        r'[^\x00-\x7f]',
        ' ',
        cleanText
    )

    cleanText = re.sub(
        r'\s+',
        ' ',
        cleanText
    )

    return cleanText.strip().lower()


# ==================================================
# FILE EXTRACTION
# ==================================================

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

    return '\n'.join(
        [para.text for para in doc.paragraphs]
    )


def extract_text_from_txt(file):

    try:

        return file.read().decode('utf-8')

    except UnicodeDecodeError:

        return file.read().decode('latin-1')


def handle_file_upload(uploaded_file):

    ext = uploaded_file.name.split('.')[-1].lower()

    if ext == 'pdf':

        return extract_text_from_pdf(
            uploaded_file
        )

    elif ext == 'docx':

        return extract_text_from_docx(
            uploaded_file
        )

    elif ext == 'txt':

        return extract_text_from_txt(
            uploaded_file
        )

    else:

        raise ValueError(
            "Unsupported file format"
        )


# ==================================================
# CHARTS
# ==================================================

def ats_gauge_chart(score):

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        title={'text': "ATS Match Score"},
        gauge={
            'axis': {'range': [0, 100]},
            'steps': [
                {
                    'range': [0, 40],
                    'color': "#ff4b4b"
                },
                {
                    'range': [40, 70],
                    'color': "#ffa500"
                },
                {
                    'range': [70, 100],
                    'color': "#2ecc71"
                }
            ],
            'bar': {'color': "darkblue"}
        }
    ))

    fig.update_layout(height=300)

    return fig


def category_chart(probs, labels):

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=labels,
        y=probs,
        text=[
            f"{p:.2f}%"
            for p in probs
        ],
        textposition="auto"
    ))

    fig.update_layout(
        title="Category Prediction Confidence",
        xaxis_title="Job Category",
        yaxis_title="Probability %",
        height=400
    )

    return fig


# ==================================================
# EMBEDDINGS
# ==================================================

def get_embedding(text):

    cleaned_text = cleanResume(text)

    embedding = embedding_model.encode(
        [cleaned_text],
        normalize_embeddings=True
    )

    return embedding


# ==================================================
# RESUME CLASSIFIER
# ==================================================

def pred(input_resume):

    vectorized_text = get_embedding(
        input_resume
    )

    prediction = svc_model.predict(
        vectorized_text
    )

    category = le.inverse_transform(
        prediction
    )[0]

    score = None
    probabilities = None

    try:

        if hasattr(
            svc_model,
            "predict_proba"
        ):

            probabilities = (
                svc_model.predict_proba(
                    vectorized_text
                )[0] * 100
            )

            score = float(
                probabilities.max()
            )

    except Exception:

        score = None
        probabilities = None

    if score is not None:

        score = round(score, 2)

    return category, score, probabilities


# ==================================================
# ATS SCORE
# ==================================================

def calculate_ats_score(
    resume_text,
    jd_text
):

    resume_embedding = get_embedding(
        resume_text
    )

    jd_embedding = get_embedding(
        jd_text
    )

    similarity = cosine_similarity(
        resume_embedding,
        jd_embedding
    )[0][0]

    return round(
        similarity * 100,
        2
    )


# ==================================================
# STREAMLIT APP
# ==================================================

def main():

    st.title(
        "📄 Resume ATS & Category Predictor"
    )

    st.markdown(
        """
        Upload a resume and paste a job
        description to get:

        ✅ Predicted Job Category  
        ✅ Semantic ATS Match Score
        """
    )

    # ==================================================
    # JOB DESCRIPTION
    # ==================================================

    jd_text = st.text_area(
        "Job Description",
        placeholder=(
            "Paste the job description here..."
        ),
        height=200
    )

    # ==================================================
    # FILE UPLOAD
    # ==================================================

    uploaded_file = st.file_uploader(
        "📎 Upload Resume",
        type=["pdf", "docx", "txt"]
    )

    if uploaded_file is not None:

        try:

            resume_text = handle_file_upload(
                uploaded_file
            )

            st.success(
                "Resume extracted successfully!"
            )

            # ==================================================
            # SHOW TEXT
            # ==================================================

            if st.checkbox(
                "Show extracted resume text"
            ):

                st.text_area(
                    "Extracted Resume Text",
                    resume_text,
                    height=300
                )

            # ==================================================
            # CATEGORY PREDICTION
            # ==================================================

            st.subheader(
                "🔍 Predicted Job Category"
            )

            category, score, probabilities = pred(
                resume_text
            )

            st.write(f"### {category}")

            if score is not None:

                st.metric(
                    "Prediction Confidence",
                    f"{score}%"
                )

            if probabilities is not None:

                labels = le.classes_

                chart = category_chart(
                    probabilities,
                    labels
                )

                st.plotly_chart(
                    chart,
                    use_container_width=True
                )

            # ==================================================
            # ATS SCORE
            # ==================================================

            st.subheader(
                "📊 ATS Match Score"
            )

            if jd_text.strip():

                ats_score = calculate_ats_score(
                    resume_text,
                    jd_text
                )

                st.plotly_chart(
                    ats_gauge_chart(
                        ats_score
                    ),
                    use_container_width=True
                )

                if ats_score >= 75:

                    st.success(
                        "Excellent semantic match!"
                    )

                elif ats_score >= 50:

                    st.warning(
                        "Moderate semantic similarity."
                    )

                else:

                    st.error(
                        "Low semantic similarity."
                    )

            else:

                st.info(
                    """
                    Paste a job description
                    to calculate ATS score.
                    """
                )

        except Exception as e:

            st.error(
                f"Error: {str(e)}"
            )


# ==================================================
# RUN APP
# ==================================================

if __name__ == "__main__":

    main()