import os
import re

MEMORY_FOLDER = "memory"

def load_keywords():
    subjects = {}

    for subject in os.listdir(MEMORY_FOLDER):
        path = os.path.join(MEMORY_FOLDER, subject, "values.txt")

        if os.path.exists(path):
            with open(path, "r") as f:
                content = f.read()
                keywords = re.findall(r'"(.*?)"', content)
                subjects[subject] = keywords

    return subjects

subject_keywords = load_keywords()

def detect_subject(question):
    question = question.lower()

    for subject, keywords in subject_keywords.items():
        for word in keywords:
            if word.lower() in question:
                return subject

    return None