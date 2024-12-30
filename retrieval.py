# retrieval.py
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from google.cloud import storage
import streamlit as st
import json
from google.oauth2 import service_account

###############################################################################
#  GCS Configuration
###############################################################################
# ตั้งค่าตัวแปร GOOGLE_APPLICATION_CREDENTIALS

# Read your service account JSON from secrets
service_account_json = st.secrets["general"]["GOOGLE_APPLICATION_CREDENTIALS_JSON"]

# Parse the string into a Python dict
service_account_info = json.loads(service_account_json)

# Create credentials object
credentials = service_account.Credentials.from_service_account_info(service_account_info)

# Then initialize your Cloud Storage client with these credentials
client = storage.Client(credentials=credentials, project=service_account_info["project_id"])

# ฟังก์ชันสำหรับดาวน์โหลดไฟล์จาก GCS
def download_from_gcs(bucket_name, source_blob_name, destination_file_name):
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)
    print(f"Downloaded {source_blob_name} to {destination_file_name}.")


###############################################################################
#  Load embeddings from GCS
###############################################################################
bucket_name = "streamlit-disc-candidate-bucket"

# ระบุไฟล์ embeddings ที่จะดาวน์โหลดจาก GCS
vectorizer_path = "local_disc_tfidf_vectorizer.pkl"
chunks_path = "local_disc_tfidf_chunks.pkl"

download_from_gcs(bucket_name, "embeddings/disc_tfidf_vectorizer.pkl", vectorizer_path)
download_from_gcs(bucket_name, "embeddings/disc_tfidf_chunks.pkl", chunks_path)

###############################################################################
#  Optimized Loading for Vectorizer and Chunks
###############################################################################
@st.cache_resource
def load_model():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

@st.cache_resource
def load_vectorizer(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)

@st.cache_resource
def load_chunks(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)

model = load_model()
vectorizer = load_vectorizer(vectorizer_path)
chunks = load_chunks(chunks_path)
vocab = vectorizer.vocabulary_

###############################################################################
# Negation Words & Switch Logic
###############################################################################
NEGATIVE_WORDS = {
    "not", "no", "nor", "dont", "didnt", "cant", "wont", "shouldnt",
    "wouldnt", "cannot", "never"
}

def contains_negation(user_text: str) -> bool:
    tokens = user_text.lower().split()
    return any(token in tokens for token in NEGATIVE_WORDS)

# Set 1 => Q1, Q2, Q5: (I ↔ S), (D ↔ C)
def switch_type_set_1(original_type: str) -> str:
    if original_type == "I":
        return "S"
    elif original_type == "S":
        return "I"
    elif original_type == "D":
        return "C"
    elif original_type == "C":
        return "D"
    return original_type

# Set 2 => Q3, Q4, Q6: (I ↔ D), (S ↔ C)
def switch_type_set_2(original_type: str) -> str:
    if original_type == "I":
        return "D"
    elif original_type == "D":
        return "I"
    elif original_type == "S":
        return "C"
    elif original_type == "C":
        return "S"
    return original_type

###############################################################################
# Building Hybrid Vectors for new user answer
###############################################################################
def preprocess_text_for_retrieval(text: str) -> str:
    """Optionally replicate the same cleanup you did in utils.py"""
    return text.lower()  # minimal for now—adjust as needed

def get_token_embedding(token):
    return model.encode(token)

def build_hybrid_vector(user_text: str):
    clean_text = preprocess_text_for_retrieval(user_text)
    user_tfidf = vectorizer.transform([clean_text])
    tokens = clean_text.split()
    
    weighted_vecs = []
    for token in tokens:
        token_index = vocab.get(token)
        if token_index is not None:
            weight = user_tfidf[0, token_index]
            if weight > 0:
                emb = get_token_embedding(token)
                weighted_vecs.append(weight * emb)
    if len(weighted_vecs) > 0:
        vec_sum = np.sum(weighted_vecs, axis=0)
        norm = np.linalg.norm(vec_sum)
        if norm > 0:
            vec_sum = vec_sum / norm
        return vec_sum
    else:
        # If no recognized tokens or zero length
        return np.zeros(model.get_sentence_embedding_dimension())


###############################################################################
# retrieve_top_n with negation switch
###############################################################################
def retrieve_top_n(user_answer: str, question_id: str, n=1):
    user_vec = build_hybrid_vector(user_answer)

    # Filter chunks by question_id
    relevant = [c for c in chunks if c["question_id"] == question_id]
    
    scored = []
    for c in relevant:
        chunk_vec = c["hybrid_vector"]  # from your pickled data
        dot = np.dot(user_vec, chunk_vec)
        denom = np.linalg.norm(user_vec) * np.linalg.norm(chunk_vec)
        sim = dot / denom if denom != 0 else 0
        scored.append((sim, c))
    
    scored.sort(key=lambda x: x[0], reverse=True)
    
    if not scored:
        return []
    
    best_sim, best_chunk = scored[0]
    
    if contains_negation(user_answer):
        original_type = best_chunk["type"]
        # Switch sets
        if question_id in ["Q1", "Q2", "Q5"]:
            new_type = switch_type_set_1(original_type)
        elif question_id in ["Q3", "Q4", "Q6"]:
            new_type = switch_type_set_2(original_type)
        else:
            new_type = original_type
        
        # Overwrite type for display
        best_chunk["type"] = new_type
        scored[0] = (best_sim, best_chunk)
    
    return scored[:n]

###############################################################################
# A helper to check max similarity (for relevance filter)
###############################################################################
def get_max_similarity(user_answer: str, question_id: str) -> float:
    """
    Returns the highest similarity (0..1) among all chunks for that question_id,
    given the user's answer.
    """
    user_vec = build_hybrid_vector(user_answer)
    relevant = [c for c in chunks if c["question_id"] == question_id]
    if not relevant:
        return 0.0
    
    scores = []
    for c in relevant:
        chunk_vec = c["hybrid_vector"]
        dot = np.dot(user_vec, chunk_vec)
        denom = np.linalg.norm(user_vec) * np.linalg.norm(chunk_vec)
        sim = dot / denom if denom != 0 else 0
        scores.append(sim)
    
    return max(scores) if scores else 0.0
