from sentence_transformers import SentenceTransformer

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC

import pandas as pd
import pickle


# ==========================================
# LOAD DATASET
# ==========================================

df = pd.read_csv("UpdatedResumeDataSet.csv")

texts = df["Resume"]

labels = df["Category"]


# ==========================================
# EMBEDDING MODEL
# ==========================================

embedding_model = SentenceTransformer(
    'all-MiniLM-L6-v2'
)


# ==========================================
# GENERATE EMBEDDINGS
# ==========================================

X = embedding_model.encode(
    texts.tolist(),
    show_progress_bar=True
)

# X shape becomes:
# (num_samples, 384)


# ==========================================
# ENCODE LABELS
# ==========================================

le = LabelEncoder()

y = le.fit_transform(labels)


# ==========================================
# TRAIN TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# ==========================================
# TRAIN MODEL
# ==========================================

model = SVC(
    kernel='linear',
    probability=True
)

model.fit(X_train, y_train)


# ==========================================
# SAVE MODEL
# ==========================================

pickle.dump(
    model,
    open('clf.pkl', 'wb')
)

pickle.dump(
    le,
    open('encoder.pkl', 'wb')
)


print("Model trained successfully!")