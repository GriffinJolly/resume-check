AI Resume Suggestor
A modern, AI-powered application that analyzes resumes and provides personalized feedback and improvement suggestions to help job seekers optimize their resumes for specific roles.

Features

Instant Resume Analysis: Upload your resume in PDF or TXT format and get immediate feedback
Job-Specific Targeting: Tailor your resume analysis to specific job roles
Comprehensive Metrics: Evaluate word count, action verbs, quantifiable achievements, and more
Keyword Optimization: Identify relevant keywords for your target role
Detailed Feedback: Get specific, actionable suggestions to improve your resume
Visual Dashboard: View your resume's strength score and key metrics at a glance
Modern UI: Clean, intuitive interface with drag-and-drop functionality

Installation
Prerequisites

Python 3.7+
pip

Setup Instructions

Clone the repository:
bashgit clone https://github.com/GriffinJolly/ai-resume-suggestor.git
cd ai-resume-suggestor

Create and activate a virtual environment:
bashpython -m venv venv

# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate

Install the required packages:
bashpip install -r requirements.txt

Run the application:
bashstreamlit run app.py

Open your browser and navigate to http://localhost:8501

Dependencies

streamlit
PyPDF2
transformers
torch
re
collections

How It Works

Upload Your Resume: Drag and drop your resume file (PDF or TXT) into the upload area
Specify Target Role: Enter the job position you're targeting
Analyze: Click the "Analyze Resume" button to start the AI analysis
Review Results: Get a comprehensive breakdown of your resume's strengths and areas for improvement
Implement Suggestions: Make the recommended changes to improve your resume's effectiveness
Reanalyze: Upload your revised resume to see if your score improves

Technical Details
Analysis Components

Content Analysis: Evaluates the use of action verbs and quantifiable achievements
Structure Analysis: Checks for proper organization and inclusion of key sections
Keyword Matching: Identifies industry and role-specific keywords
Sentiment Analysis: Measures the overall tone and impact of your resume
Contact Information Check: Ensures all necessary contact details are included

AI Models
The application uses transformer-based language models to analyze resumes:

DistilBERT for sentiment analysis and text classification
Custom keyword extraction algorithms

Future Enhancements

Resume comparison with industry standards
ATS (Applicant Tracking System) compatibility scoring
Custom formatting suggestions based on industry
Export functionality for feedback reports
Integration with job posting analysis
Interview preparation suggestions
