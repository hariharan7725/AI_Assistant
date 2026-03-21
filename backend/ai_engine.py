from backend.subject_detector import detect_subject
import os

def get_answer(question):

    subject = detect_subject(question)

    if subject is None:
        return "❌ Question not related to available subjects"

    # TODO: load FAISS + chunks + LLM here
    # (your original code)

    # For now dummy:
    return f"Answer generated for subject: {subject}"