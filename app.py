import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel
import docx
from PyPDF2 import PdfReader
import joblib

# Constants
MODEL_DIR = "./peft_lora_legalbert_model/model_files"
BASE_MODEL = "nlpaueb/legal-bert-base-uncased"

# Load the label encoder
LABEL_ENCODER_PATH = "./dataset_processed/label_encoder.joblib"   # adjust path if needed
le = joblib.load(LABEL_ENCODER_PATH)

# Extract labels
LABELS = list(le.classes_)   # dynamically retrieved


THRESHOLD = 0.75  # confidence threshold for human review

# Load model and tokenizer
@st.cache_resource
def load_model_and_tokenizer():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    num_labels = 41
    base_model = AutoModelForSequenceClassification.from_pretrained(BASE_MODEL, num_labels=num_labels)
    model = PeftModel.from_pretrained(base_model, MODEL_DIR)
    model.to(device)
    model.eval()
    return tokenizer, model, device

def predict_topn_clauses(text, tokenizer, model, device, label_encoder, top_n=3):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256).to(device)
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=-1).cpu().numpy().flatten()
    top_indices = probs.argsort()[::-1][:top_n]
    labels = label_encoder.inverse_transform(top_indices)
    confidences = probs[top_indices]
    return list(zip(labels, confidences))


# Utility to extract text from uploaded files
def extract_text_from_file(uploaded_file):
    if uploaded_file.type == "text/plain":
        return uploaded_file.read().decode("utf-8")
    elif uploaded_file.type == "application/pdf":
        pdf = PdfReader(uploaded_file)
        return " ".join([page.extract_text() for page in pdf.pages if page.extract_text()])
    elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        doc = docx.Document(uploaded_file)
        return " ".join([para.text for para in doc.paragraphs])
    else:
        return None

# Streamlit App
def main():
    st.markdown("""
        <style>
        body {
            background-color: #f8f9fa;
            color: #212529;
        }
        .stApp {
            background-color: #129956;
            border: 1px solid #dee2e6;
            border-radius: 10px;
            padding: 15px;
        }
        h1, h2, h3 {
            color: #003366;
        }
        </style>
    """, unsafe_allow_html=True)

    st.title("Legal Clause Classification Prototype")

    tokenizer, model, device = load_model_and_tokenizer()

    st.subheader("Input Options")
    text_input = st.text_area("Enter text of the legal clause:", height=150)

    uploaded_file = st.file_uploader("Or upload a file (txt, pdf, docx)", type=["txt", "pdf", "docx"])
    if uploaded_file:
        extracted_text = extract_text_from_file(uploaded_file)
        if extracted_text:
            text_input = extracted_text[:2000]  # limit for safety
            st.info("Extracted text loaded from file.")

    if st.button("Predict Clause"):
        if not text_input.strip():
            st.error("Please enter clause text for prediction.")
        else:
            results = predict_topn_clauses(text_input, tokenizer, model, device, le, top_n=3)
            st.write("### Top Predictions")
            for label, conf in results:
                st.write(f"- **{label}** (confidence: {conf:.2f})")
            if results[0][1]<THRESHOLD:
                st.write(f"This is flagged for human review as the confidence is less than the threshold")

    # Disclaimer
    st.markdown("---")
    st.markdown("### Disclaimer")
    st.write("""
    ⚖️ This system is an **assistive tool** for contract clause classification.  
    - It does **not** replace professional legal advice.  
    - Predictions are probabilistic and may contain errors.  
    - Users are responsible for reviewing all outputs, and low-confidence results should be referred to legal experts.  
    """)

if __name__ == "__main__":
    main()
