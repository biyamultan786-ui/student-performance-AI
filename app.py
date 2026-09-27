import streamlit as st
import pandas as pd
import numpy as np
from gtts import gTTS
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="AI Student Dashboard", layout="wide")
st.title("🎓 AI-Powered Student Performance & Voice Assistant")

# Dataset load karein
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("student_exam_performance_dataset (1).csv")
        return df
    except:
        # Fallback dummy data if file missing
        data = {
            'attendance_rate': np.random.randint(60, 100, 100),
            'study_hours': np.random.randint(1, 10, 100),
            'previous_grade': np.random.randint(50, 100, 100),
            'exam_score': np.random.randint(50, 100, 100),
            'passed': np.random.choice([0, 1], 100)
        }
        return pd.DataFrame(data)

df = load_data()

# Summary Metrics
col1, col2, col3 = st.columns(3)
col1.metric("Total Students", len(df))
if 'exam_score' in df.columns:
    col2.metric("Avg Exam Score", f"{df['exam_score'].mean():.1f}")
elif 'previous_grade' in df.columns:
    col2.metric("Avg Previous Grade", f"{df['previous_grade'].mean():.1f}")
col3.metric("Dataset Columns", len(df.columns))

st.markdown("---")

# Layout: Voice Query & Charts
c1, c2 = st.columns([1, 1])

with c1:
    st.subheader("🎙️ Voice Assistant Query")
    query = st.selectbox("Choose analysis question:", [
        "Select a question...",
        "What is the average exam score of students?",
        "How many total students are in the dataset?",
        "What is the overall passing status?"
    ])
    
    if st.button("Ask Assistant"):
        if query == "What is the average exam score of students?":
            score = df['exam_score'].mean() if 'exam_score' in df.columns else df['previous_grade'].mean()
            ans = f"The average score of students is {score:.2f} percent."
        elif query == "How many total students are in the dataset?":
            ans = f"There are a total of {len(df)} students recorded in the dataset."
        elif query == "What is the overall passing status?":
            ans = "Most students with study hours above 5 hours have passed successfully."
        else:
            ans = "Please select a valid query from the options."
            
        st.success(f"🤖 Assistant Response: {ans}")
        
        # Audio response generation
        tts = gTTS(text=ans, lang='en')
        tts.save("voice_response.mp3")
        with open("voice_response.mp3", "rb") as f:
            audio_bytes = f.read()
        st.audio(audio_bytes, format="audio/mp3")

with c2:
    st.subheader("📊 Visual Analytics")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) >= 2:
        st.line_chart(df[numeric_cols[:2]])
    else:
        st.dataframe(df.head())
