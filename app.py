import os
import streamlit as st
import joblib
import pandas as pd
from pdfminer.high_level import extract_text
import fitz
import spacy
from nltk.corpus import stopwords
import nltk
import re

# Load the saved models
svm_model = joblib.load('svm_model.pkl')
logistic_regression_model = joblib.load('logistic_regression_model.pkl')
tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Load Spacy's English language model for lemmatization
nlp = spacy.load("en_core_web_sm")

# Download NLTK stopwords if not already downloaded
nltk.download('stopwords')

# Define a list of possible educational qualifications
education_keywords = ['bachelor', 'master', 'phd', 'degree', 'bsc', 'msc']

# Preprocess text function
def preprocess_text(text):
    text = text.lower()  # Lowercase text
    doc = nlp(text)
    lemmatized_tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
    return " ".join(lemmatized_tokens)

# Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    # Open the provided PDF file
    doc = fitz.open(pdf_path)
    
    # Extract text from each page
    text = ""
    for page_num in range(doc.page_count):
        page = doc.load_page(page_num)  # Page is 0-indexed
        text += page.get_text()

    # Clean up the text (optional, for better processing)
    text = text.strip()  # Remove leading/trailing whitespace
    text = ' '.join(text.split())  # Remove extra spaces between words

    return text

# Function to screen for skills in the resume
def screen_resume_for_skills(resume_text, keywords):
    resume_text = resume_text.lower()
    matched_skills = [keyword for keyword in keywords if keyword.lower() in resume_text]
    return matched_skills

# Function to screen for education qualifications in the resume
def screen_resume_for_education(resume_text, education_keywords):
    resume_text = resume_text.lower()
    matched_education = [keyword for keyword in education_keywords if keyword.lower() in resume_text]
    return matched_education

# Function to extract experience (years) and job titles
def extract_experience_and_titles(resume_text):
    doc = nlp(resume_text)
    experience = []
    titles = []
    experience_pattern = r'(\d{1,2}[\+\-]?\s?(year|yr|yrs|experience|exp))'
    experience_matches = re.findall(experience_pattern, resume_text.lower())
    
    if experience_matches:
        for match in experience_matches:
            experience.append(match[0].strip())

    for token in doc:
        if token.dep_ == 'nsubj' and token.pos_ in ['NOUN', 'PROPN']:
            if token.text.lower() not in ['the', 'a', 'an', 'and', 'or']:
                titles.append(token.text)

    return experience, titles

# Function to combine all screening and extraction
def screen_resume(resume_text, skills_keywords):
    matched_skills = screen_resume_for_skills(resume_text, skills_keywords)
    matched_education = screen_resume_for_education(resume_text, education_keywords)
    experience, job_titles = extract_experience_and_titles(resume_text)
    
    return matched_skills, matched_education, experience, job_titles

# Set up folder for uploaded resumes
UPLOAD_FOLDER = "uploaded_resumes"
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Set title and style
st.set_page_config(page_title="Resume Categorization and Screening App", page_icon=":guardsman:", layout="centered")
st.markdown("""
    <style>
    .stMarkdown .title {
        font-size: 50px !important; 
        font-weight: bold;
        color: #2C3E50;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .stMarkdown .section-header {
        font-size: 25px !important; 
        font-weight: bold;
        color: #1ABC9C;
        text-shadow: 2px 2px 5px rgba(0,0,0,0.15);
    }
    .stTextArea textarea { height:100px; background-color: #ECF0F1; padding: 12px; border-radius: 8px; }
    .best-resume {
        background-color: #FFD700 !important;  /* Golden background for the best resume */
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# Set title
st.markdown('<p class="title">Resume Categorization and Screening App</p>', unsafe_allow_html=True)

# Model selection section
st.markdown("<p class='section-header'>Step 1: Select the Model</p>", unsafe_allow_html=True)
model_choice = st.selectbox("Select the Model to Use", ["Logistic Regression", "SVM"])

# Skills input section
st.markdown("<p class='section-header'>Step 2: Enter Custom Skills to Screen for</p>", unsafe_allow_html=True)
user_skills_input = st.text_area(
    "Enter skills to screen for (e.g., Python, SQL, Machine Learning):", 
    height=200,
    help="Enter a list of skills separated by commas (e.g., Python, SQL, Machine Learning)"
)

if user_skills_input:
    user_skills = [skill.strip() for skill in user_skills_input.split(",")]
else:
    user_skills = ['python', 'sql', 'machine learning', 'java', 'c++', 'project management', 'data science']


# Multiple resume upload section
st.markdown("<p class='section-header'>Step 3: Upload Your Resumes (PDF)</p>", unsafe_allow_html=True)
uploaded_files = st.file_uploader("Upload your resumes (PDF)", type="pdf", accept_multiple_files=True)

if uploaded_files:
    results = []
    
    with st.spinner("Processing resumes... Please wait!"):
        for uploaded_file in uploaded_files:
            file_path = os.path.join(UPLOAD_FOLDER, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            resume_text = extract_text_from_pdf(file_path)
            cleaned_resume_text = preprocess_text(resume_text)
            resume_features = tfidf_vectorizer.transform([cleaned_resume_text]).toarray()
            
            # Choose model based on user input
            if model_choice == "Logistic Regression":
                category_prediction = logistic_regression_model.predict(resume_features)
            else:
                category_prediction = svm_model.predict(resume_features)

            matched_skills, matched_education, experience, job_titles = screen_resume(resume_text, user_skills)
            
            # Weighted score calculation
            skill_weight = 1
            education_weight = 1
            experience_weight = 1  # 1 point per year of experience
            
            # Calculate total score for this resume
            experience_years = sum([int(exp.split()[0]) for exp in experience if exp.split()[0].isdigit()])  # Sum up years
            score = (len(matched_skills) * skill_weight) + (len(matched_education) * education_weight) + (experience_years * experience_weight)

            results.append({
                "Resume Name": uploaded_file.name,
                "Predicted Category": category_prediction[0],
                "Matched Skills": matched_skills,
                "Matched Education": matched_education,
                "Experience Mentioned": experience,
                "Score": score
            })
        
        # Sort the results based on score (descending)
        results.sort(key=lambda x: x["Score"], reverse=True)
        
        # Display results in a table with highlight for the best-ranked resume
        st.write("### Resume Ranking")
        st.write(f"Total Resumes Processed: {len(uploaded_files)}")
        
        # Create DataFrame for better visualization
        df = pd.DataFrame(results)

        # Find the index of the highest score
        best_resume_index = df['Score'].idxmax()

        # Apply CSS to highlight the top-ranked resume
        df_html = df.to_html(classes="table table-striped", escape=False)
        df_html = df_html.replace('<tr>', f'<tr class="best-resume" id="best-resume">', 1)  # Highlight the first row (best resume)
        
        # Render table with highlights
        st.markdown(df_html, unsafe_allow_html=True)

        # Export results as CSV
        st.download_button(
            label="Download Resume Screening Results (CSV)",
            data=df.to_csv(index=False),
            file_name="resume_screening_results.csv",
            mime="text/csv"
        )
