import streamlit as st
import PyPDF2
import io
import os
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch
import re
from collections import Counter

# Custom CSS styling for dotted drag area (no icon, no cloud)
st.markdown("""
<style>
    .stApp {
        max-width: 800px;
        margin: 0 auto;
        padding: 2rem;
        background-color: #0e1117;
        color: #c0c0c0;
    }
    .drag-drop-area {
        border: 2.5px dotted #00cccc;
        border-radius: 16px;
        background: rgba(0, 128, 128, 0.06);
        padding: 32px 16px 16px 16px;
        margin-bottom: 1.5rem;
        text-align: center;
        transition: border-color 0.2s, background 0.2s;
        position: relative;
    }
    .drag-drop-area:hover, .drag-drop-area:focus-within {
        border-color: #008080;
        background: rgba(0, 128, 128, 0.12);
    }
    .file-uploader {
        margin: 1rem 0;
    }
    .job-role-input {
        margin: 1rem 0;
    }
    .analyze-button {
        background-color: #008080;
        color: white;
        font-weight: bold;
        padding: 0.5rem 2rem;
        border-radius: 0.5rem;
        border: none;
        transition: background-color 0.2s;
    }
    .analyze-button:hover {
        background-color: #006666;
    }
    .progress-bar {
        background-color: #1a1a1a;
        border-radius: 0.5rem;
        overflow: hidden;
        margin: 1rem 0;
    }
    .progress-bar .stProgress {
        background-color: #008080;
        height: 1rem;
        border-radius: 0.5rem;
    }
    .feedback-container {
        background-color: #1a1a1a;
        padding: 1.5rem;
        border-radius: 0.8rem;
        margin: 1.5rem 0;
        border: 1px solid #404040;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
    }
    .feedback-item {
        margin: 1rem 0;
        padding: 0.8rem;
        border-left: 4px solid #008080;
        border-radius: 0 0.5rem 0.5rem 0;
        background-color: #1a1a1a;
    }
    .feedback-item.success {
        border-left-color: #008080;
    }
    .feedback-item.warning {
        border-left-color: #ff4b4b;
    }
    .feedback-item.info {
        border-left-color: #0066ff;
    }
    h1, h2, h3 {
        color: #008080;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    p {
        color: #c0c0c0;
        margin-bottom: 0.5rem;
    }
    .stMarkdown {
        color: #c0c0c0;
    }
    .stProgress {
        background-color: #008080;
    }
    .stButton>button {
        background-color: #008080;
        color: white;
        border: none;
        border-radius: 0.5rem;
        padding: 0.5rem 2rem;
        font-weight: bold;
        transition: background-color 0.2s;
    }
    .stButton>button:hover {
        background-color: #006666;
    }
    .stTextInput>div>div>input {
        background-color: #1a1a1a;
        color: white;
        border: 1px solid #404040;
    }
    .stTextInput>div>div>input:focus {
        border-color: #008080;
    }
    .stFileUploader>div>div>input {
        background-color: #1a1a1a;
        color: white;
        border: 1px solid #404040;
    }
    .stFileUploader>div>div>input:focus {
        border-color: #008080;
    }
    .stSpinner {
        color: #008080;
    }
    .stError {
        color: #ff4b4b;
    }
    .stSuccess {
        color: #008080;
    }
    .keyword-tag {
        display: inline-block;
        background-color: rgba(0, 128, 128, 0.2);
        color: #00cccc;
        padding: 2px 8px;
        margin: 2px;
        border-radius: 12px;
        font-size: 0.85em;
    }
    .metrics-container {
        display: flex;
        justify-content: space-between;
        flex-wrap: wrap;
        gap: 10px;
        margin: 1rem 0;
    }
    .metric-card {
        background-color: #1a1a1a;
        padding: 1rem;
        border-radius: 0.5rem;
        flex: 1;
        min-width: 120px;
        text-align: center;
        border: 1px solid #2a2a2a;
    }
    .metric-value {
        font-size: 1.5rem;
        font-weight: bold;
        color: #008080;
    }
    .metric-label {
        font-size: 0.8rem;
        color: #a0a0a0;
    }
</style>
""", unsafe_allow_html=True)

# Main header
st.markdown("""
<div style='text-align: center; margin-bottom: 2rem; color: #008080;'>
    <h1>AI Resume Suggestor</h1>
    <p style='color: #c0c0c0;'>Upload your resume and let the AI suggest targeted improvements.</p>
</div>
""", unsafe_allow_html=True)

# Columns for layout
left_column, right_column = st.columns([2, 1])

with left_column:
    st.markdown("""
    <div class="drag-drop-area">
        <div style="color:#00cccc; font-weight:600; margin-bottom:12px;">
            Drag and drop your resume here
        </div>
    """, unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Upload your resume",
        type=["pdf", "txt"],
        help="Upload your resume in PDF or TXT format",
        key="file_uploader"
    )
    st.markdown("</div>", unsafe_allow_html=True)

    # Acknowledge file receipt
    if uploaded_file is not None:
        st.success(f"✅ File received: {uploaded_file.name}")


with right_column:
    job_role = st.text_input(
        "Target Job Role",
        placeholder="E.g., Senior Software Engineer",
        help="Specify the job role you're applying for"
    )

analyze = st.button("Analyze Resume", type="primary")

# --- Model and analysis logic unchanged ---
@st.cache_resource
def load_sentiment_model():
    model_name = "distilbert-base-uncased-finetuned-sst-2-english"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    return model, tokenizer

@st.cache_resource
def load_text_classifier():
    return pipeline("text-classification", model="distilbert-base-uncased")

@st.cache_data
def get_job_role_keywords():
    return {
        "software engineer": ["python", "java", "javascript", "react", "angular", "node.js", 
                             "algorithms", "data structures", "api", "rest", "git", "agile", 
                             "cloud", "aws", "azure", "docker", "kubernetes", "ci/cd", "testing"],
        "data scientist": ["python", "r", "sql", "pandas", "numpy", "machine learning", "deep learning", 
                         "tensorflow", "pytorch", "scikit-learn", "statistics", "data visualization", 
                         "data analysis", "big data", "hadoop", "spark", "nlp"],
        "product manager": ["product strategy", "roadmap", "agile", "scrum", "user stories", 
                          "market research", "customer feedback", "stakeholder", "kpi", "metrics", 
                          "a/b testing", "product lifecycle", "prioritization", "user experience"],
        "marketing": ["social media", "content marketing", "seo", "analytics", "campaign management", 
                    "market research", "branding", "digital marketing", "email marketing", 
                    "customer acquisition", "conversion rates", "marketing strategy", "crm"],
        "designer": ["ui", "ux", "user experience", "figma", "sketch", "adobe", "photoshop", 
                   "illustrator", "wireframing", "prototyping", "visual design", "typography", 
                   "responsive design", "design systems"]
    }

def extract_text_from_pdf(pdf_file):
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() + "\n"
    return text

def extract_text_from_file(uploaded_file):
    if uploaded_file.type == "application/pdf":
        return extract_text_from_pdf(io.BytesIO(uploaded_file.read()))
    return uploaded_file.read().decode("utf-8")

def count_action_verbs(text):
    action_verbs = [
        "achieved", "improved", "trained", "managed", "created", "resolved", "developed", 
        "implemented", "designed", "launched", "increased", "decreased", "negotiated", 
        "coordinated", "led", "organized", "planned", "produced", "reduced", "researched",
        "supervised", "built", "delivered", "executed", "generated", "initiated", "performed",
        "streamlined", "transformed", "analyzed", "established"
    ]
    text_lower = text.lower()
    count = 0
    for verb in action_verbs:
        count += len(re.findall(r'\b' + verb + r'\b', text_lower))
    return count

def detect_quantifiable_achievements(text):
    percentage_pattern = r'\b\d+(\.\d+)?%'
    number_pattern = r'\b\d+(\.\d+)?\s+(million|thousand|percent|users|customers|clients|increase|decrease|reduction)'
    percentages = re.findall(percentage_pattern, text)
    numbers = re.findall(number_pattern, text, re.IGNORECASE)
    return len(percentages) + len(numbers)

def analyze_skills_for_role(text, job_role):
    job_keywords = get_job_role_keywords()
    job_role_lower = job_role.lower()
    matched_role = None
    for role in job_keywords:
        if role in job_role_lower:
            matched_role = role
            break
    if not matched_role:
        matched_role = "software engineer"
        for role in job_keywords:
            role_words = set(role.split())
            job_words = set(job_role_lower.split())
            if len(role_words.intersection(job_words)) > 0:
                matched_role = role
                break
    relevant_keywords = job_keywords[matched_role]
    text_lower = text.lower()
    found_keywords = []
    for keyword in relevant_keywords:
        if keyword in text_lower:
            found_keywords.append(keyword)
    match_rate = len(found_keywords) / len(relevant_keywords) if relevant_keywords else 0
    return found_keywords, match_rate, matched_role

def detect_education(text):
    education_keywords = ["degree", "bachelor", "master", "phd", "mba", "certificate", 
                         "certification", "diploma", "university", "college", "graduate"]
    text_lower = text.lower()
    education_count = 0
    for keyword in education_keywords:
        if keyword in text_lower:
            education_count += 1
    return education_count >= 2

def count_words(text):
    return len(re.findall(r'\b\w+\b', text))

def detect_contact_info(text):
    phone_pattern = r'(\+\d{1,3}[-\.\s]??)?\(?\d{3}\)?[-\.\s]?\d{3}[-\.\s]?\d{4}'
    email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
    linkedin_pattern = r'linkedin\.com\/in\/[a-zA-Z0-9_-]+'
    has_phone = bool(re.search(phone_pattern, text))
    has_email = bool(re.search(email_pattern, text))
    has_linkedin = bool(re.search(linkedin_pattern, text))
    return has_phone, has_email, has_linkedin

if analyze and uploaded_file:
    try:
        file_content = extract_text_from_file(uploaded_file)
        if not file_content.strip():
            st.error("The uploaded file is empty or could not be read properly.")
            st.stop()

        with st.spinner("Analyzing your resume..."):
            progress_bar = st.progress(0)
            # Step 1: Basic content analysis
            progress_bar.progress(20)
            word_count = count_words(file_content)
            action_verb_count = count_action_verbs(file_content)
            quantifiable_count = detect_quantifiable_achievements(file_content)
            has_phone, has_email, has_linkedin = detect_contact_info(file_content)
            has_education = detect_education(file_content)
            # Step 2: Job role matching
            progress_bar.progress(40)
            if job_role:
                found_keywords, match_rate, matched_role = analyze_skills_for_role(file_content, job_role)
            else:
                found_keywords, match_rate, matched_role = [], 0, None
            # Step 3: Sentiment analysis with transformer model
            progress_bar.progress(60)
            try:
                model, tokenizer = load_sentiment_model()
                max_length = 512
                chunks = [file_content[i:i+max_length] for i in range(0, len(file_content), max_length)]
                inputs = tokenizer(chunks[0], return_tensors="pt", truncation=True, max_length=max_length)
                with torch.no_grad():
                    outputs = model(**inputs)
                    scores = torch.nn.functional.softmax(outputs.logits, dim=1)
                    sentiment_score = scores[0][1].item()
            except Exception as e:
                st.warning("Could not perform advanced sentiment analysis. Using simpler analysis methods.")
                sentiment_score = min(0.5 + (action_verb_count / max(20, word_count/50)) + (quantifiable_count / 10), 0.95)
            # Step 4: Overall resume quality assessment
            progress_bar.progress(80)
            contact_score = (has_phone + has_email + has_linkedin) / 3
            content_score = min((action_verb_count / max(15, word_count/100)) + (quantifiable_count / 8), 1)
            structure_score = 0.7 if has_education else 0.4
            if job_role:
                relevance_score = match_rate
            else:
                relevance_score = 0.5
            overall_score = (sentiment_score * 0.3) + (contact_score * 0.1) + (content_score * 0.3) + (structure_score * 0.1) + (relevance_score * 0.2)
            overall_score = max(0.1, min(overall_score, 0.98))
            progress_bar.progress(100)

        st.balloons()
        st.toast("Analysis complete!", icon="🎉")

        st.markdown("### Resume Analysis Results")
        st.markdown("""
        <div class="metrics-container">
            <div class="metric-card">
                <div class="metric-value">{:.0f}</div>
                <div class="metric-label">WORDS</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{:.0f}</div>
                <div class="metric-label">ACTION VERBS</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{:.0f}</div>
                <div class="metric-label">QUANTIFIED RESULTS</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{:.0%}</div>
                <div class="metric-label">JOB MATCH</div>
            </div>
        </div>
        """.format(word_count, action_verb_count, quantifiable_count, match_rate if job_role else 0), unsafe_allow_html=True)

        st.markdown("""
        <div class="feedback-container">
            <h3>Resume Strength Score</h3>
            <div class="progress-bar">
                <div class="stProgress" style="width: {}%"></div>
            </div>
            <p style='color: #c0c0c0;'>{:.1%} overall strength score</p>
        </div>
        """.format(int(overall_score * 100), overall_score), unsafe_allow_html=True)

        feedback = []
        if overall_score > 0.8:
            feedback.append("<div class='feedback-item success'>✨ <strong>Excellent Resume</strong>: Your resume appears professional and well-structured with strong content.</div>")
        elif overall_score > 0.6:
            feedback.append("<div class='feedback-item success'>✓ <strong>Good Resume</strong>: Your resume has several strengths but could benefit from some improvements.</div>")
        elif overall_score > 0.4:
            feedback.append("<div class='feedback-item'>📝 <strong>Average Resume</strong>: Your resume needs work in multiple areas to make a stronger impact.</div>")
        else:
            feedback.append("<div class='feedback-item warning'>⚠️ <strong>Needs Improvement</strong>: Your resume requires significant revisions to be competitive.</div>")
        content_feedback = ""
        if action_verb_count < 5:
            content_feedback += "<li>Add more action verbs (like 'achieved', 'developed', 'implemented') to highlight your accomplishments</li>"
        if quantifiable_count < 3:
            content_feedback += "<li>Include more quantifiable achievements (numbers, percentages, metrics) to demonstrate impact</li>"
        if content_feedback:
            feedback.append(f"""<div class='feedback-item info'>
            <strong>Content Improvements:</strong>
            <ul>{content_feedback}</ul>
            </div>""")
        if not all([has_phone, has_email]):
            feedback.append("<div class='feedback-item warning'><strong>Missing Contact Information:</strong> Ensure your resume includes complete contact details including phone and email.</div>")
        if job_role and matched_role:
            keyword_html = " ".join([f'<span class="keyword-tag">{kw}</span>' for kw in found_keywords])
            keyword_feedback = f"""<div class='feedback-item'>
            <strong>🎯 {job_role} Relevant Keywords Found:</strong><br>
            {keyword_html}<br><br>
            """
            if match_rate < 0.7:
                missing_keywords = get_job_role_keywords()[matched_role]
                missing_keywords = [k for k in missing_keywords if k not in found_keywords][:5]
                if missing_keywords:
                    keyword_feedback += "<strong>Consider adding these relevant keywords:</strong><br>"
                    keyword_feedback += " ".join([f'<span class="keyword-tag">{kw}</span>' for kw in missing_keywords])
            keyword_feedback += "</div>"
            feedback.append(keyword_feedback)
            feedback.append(f"""<div class='feedback-item info'>
            <strong>Tips for {job_role} Applications:</strong>
            <ul>
                <li>Tailor your experience section to highlight skills most relevant to {job_role}</li>
                <li>Include specific projects or achievements that demonstrate your capabilities</li>
                <li>Match your terminology to what's commonly used in {matched_role} job descriptions</li>
            </ul>
            </div>""")
        feedback.append("""<div class='feedback-item info'>
        <strong>💡 General Enhancement Tips:</strong>
        <ul>
            <li>Keep your resume concise and focused (1-2 pages maximum)</li>
            <li>Use a clean, professional layout with consistent formatting</li>
            <li>Proofread carefully for spelling and grammatical errors</li>
            <li>Customize for each application rather than using a generic version</li>
            <li>Include a brief professional summary that highlights your strengths</li>
        </ul>
        </div>""")
        st.markdown("""
        <div class="feedback-container">
            <h3>Detailed Resume Feedback</h3>
            {}
        </div>
        """.format("\n".join(feedback)), unsafe_allow_html=True)
        st.markdown("""
        <div class="feedback-container">
            <h3>Next Steps</h3>
            <p>Based on this analysis, consider making the suggested improvements to strengthen your resume for your target role. After revisions, upload again to see if your score improves!</p>
        </div>
        """, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"An error occurred during analysis: {str(e)}")
        st.error("Please try uploading a different file or try again later.")
