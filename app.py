import re
import io
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from gtts import gTTS
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="AI Student Dashboard", layout="wide")
st.title("🎓 AI-Powered Student Performance & Voice Assistant")

# ---------------------------------------------------------------
# 1. LOAD DATA
# ---------------------------------------------------------------
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("student_exam_performance_dataset (1).csv")
        return df
    except Exception:
        data = {
            "attendance_rate": np.random.randint(60, 100, 100),
            "study_hours_per_day": np.random.randint(1, 10, 100),
            "previous_gpa": np.random.uniform(1, 4, 100),
            "final_exam_score": np.random.randint(50, 100, 100),
            "pass_fail": np.random.choice(["Pass", "Fail"], 100),
        }
        return pd.DataFrame(data)

df = load_data()

def find_col(possible_names):
    for name in possible_names:
        if name in df.columns:
            return name
    return None

SCORE_COL = find_col(["final_exam_score", "exam_score", "score"])
PASS_COL = find_col(["pass_fail", "passed", "pass_status"])
ID_COL = find_col(["student_id", "id"])
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

col1, col2, col3 = st.columns(3)
col1.metric("Total Students", len(df))
if SCORE_COL:
    col2.metric("Avg Exam Score", f"{df[SCORE_COL].mean():.1f}")
col3.metric("Dataset Columns", len(df.columns))
st.markdown("---")


def speak(text):
    try:
        tts = gTTS(text=text, lang="en")
        tts.save("voice_response.mp3")
        with open("voice_response.mp3", "rb") as f:
            audio_bytes = f.read()
        st.audio(audio_bytes, format="audio/mp3")
    except Exception as e:
        st.warning(f"Voice output unavailable: {e}")


# ---------------------------------------------------------------
# HELPER: match a spoken phrase to an actual column name
# ---------------------------------------------------------------
def match_column(text):
    """Find the dataset column whose name is closest to what was said."""
    text_clean = text.lower().replace("_", " ")
    best_col, best_score = None, 0
    for col in df.columns:
        col_words = col.lower().replace("_", " ")
        score = sum(1 for w in col_words.split() if w in text_clean)
        if score > best_score:
            best_score, best_col = score, col
    return best_col if best_score > 0 else None


def find_all_mentioned_columns(text):
    """Return every numeric column mentioned in the text, in the order mentioned."""
    text_clean = text.lower().replace("_", " ")
    found = []
    for col in numeric_cols:
        col_words = col.lower().replace("_", " ")
        if all(w in text_clean for w in col_words.split()):
            pos = text_clean.find(col_words.split()[0])
            found.append((pos, col))
    found.sort(key=lambda x: x[0])
    return [c for _, c in found]


def correct_common_mishearings(text):
    """Fix common speech-recognition mistakes for our key statistical terms."""
    replacements = {
        "recreation": "correlation",
        "correlate": "correlation",
        "core relation": "correlation",
        "regretion": "regression",
        "regresion": "regression",
        "aggression": "regression",
        "progression": "regression",
        "study over": "study hours",
        "study hour": "study hours",
        "studying hours": "study hours",
    }
    fixed = text.lower()
    for wrong, right in replacements.items():
        fixed = fixed.replace(wrong, right)
    return fixed


def handle_voice_command(text):
    """Understand the transcribed sentence and run the right analysis."""
    text = correct_common_mishearings(text)
    t = text.lower()

    # 1. Student lookup: look for an ID pattern like S00501
    id_match = re.search(r"\b([sS]\s?0*\d{1,6})\b", text)
    if "student" in t and id_match and ID_COL:
        raw_id = id_match.group(1).replace(" ", "").upper()
        digits = re.sub(r"\D", "", raw_id)
        candidate = "S" + digits.zfill(5)
        match = df[df[ID_COL].astype(str).str.upper() == candidate]
        if match.empty:
            return f"No student found with ID {candidate}.", None
        row = match.iloc[0]
        if SCORE_COL:
            class_avg = df[SCORE_COL].mean()
            comparison = "above" if row[SCORE_COL] > class_avg else "below"
            ans = (
                f"Student {candidate} has a {SCORE_COL} of {row[SCORE_COL]:.1f}, "
                f"which is {comparison} the class average of {class_avg:.1f}."
            )
        else:
            ans = f"Student {candidate} record found."
        return ans, match

    # 2. Correlation
    if "correlation" in t or "correlate" in t:
        cols = find_all_mentioned_columns(text)
        if len(cols) >= 2:
            corr = df[cols[0]].corr(df[cols[1]])
            strength = "a strong" if abs(corr) > 0.7 else ("a moderate" if abs(corr) > 0.3 else "a weak")
            direction = "positive" if corr > 0 else "negative"
            ans = (
                f"The correlation between {cols[0]} and {cols[1]} is {corr:.3f}, "
                f"which indicates {strength} {direction} relationship."
            )
            return ans, None
        return "I heard 'correlation' but could not identify two matching variables. Please mention two column names clearly.", None

    # 3. Regression
    if "regression" in t:
        cols = find_all_mentioned_columns(text)
        if len(cols) >= 2:
            X = df[[cols[0]]].values
            y = df[cols[1]].values
            model = LinearRegression().fit(X, y)
            r2 = model.score(X, y)
            ans = (
                f"The regression line is {cols[1]} equals {model.coef_[0]:.3f} times {cols[0]} "
                f"plus {model.intercept_:.3f}, with an R-squared of {r2:.3f}."
            )
            fig, ax = plt.subplots()
            ax.scatter(df[cols[0]], df[cols[1]], alpha=0.3, s=10)
            x_line = np.linspace(df[cols[0]].min(), df[cols[0]].max(), 100)
            ax.plot(x_line, model.predict(x_line.reshape(-1, 1)), color="red", linewidth=2)
            ax.set_xlabel(cols[0]); ax.set_ylabel(cols[1])
            st.pyplot(fig)
            return ans, None
        return "I heard 'regression' but could not identify two matching variables. Please mention two column names clearly.", None

    # 4. Mean / average / standard deviation / min / max of a single column
    if any(k in t for k in ["average", "mean", "standard deviation", "minimum", "maximum"]):
        col = match_column(text)
        if col:
            series = df[col]
            ans = (
                f"For {col}: mean is {series.mean():.2f}, standard deviation is {series.std():.2f}, "
                f"minimum is {series.min():.2f}, and maximum is {series.max():.2f}."
            )
            return ans, None
        return "I heard a statistics request but could not identify which column you meant.", None

    # 5. Pass/fail or total students (fallback quick questions)
    if "pass" in t and PASS_COL:
        pass_rate = (df[PASS_COL].astype(str).str.lower() == "pass").mean() * 100
        return f"Overall, {pass_rate:.1f} percent of students have passed.", None
    if "how many" in t and "student" in t:
        return f"There are a total of {len(df)} students recorded in the dataset.", None

    return (
        "Sorry, I could not understand that request. Try asking about correlation, "
        "regression, average, or a specific student ID.",
        None,
    )


# ---------------------------------------------------------------
# 2. MODE SELECTOR
# ---------------------------------------------------------------
st.subheader("🎙️ Ask the AI Assistant")

mode = st.radio(
    "What would you like to ask?",
    [
        "🎤 Voice Command (speak your question)",
        "Quick Questions",
        "Column Statistics (mean, std, min, max)",
        "Correlation between two variables",
        "Regression line between two variables",
        "Look up a specific student",
    ],
)

c1, c2 = st.columns([1, 1])

# ---------- VOICE COMMAND MODE ----------
if mode == "🎤 Voice Command (speak your question)":
    with c1:
        st.write(
            "Click the microphone, speak your question clearly, then wait a moment "
            "for the transcription. Example: *\"What is the correlation between "
            "study hours and final exam score?\"*"
        )
        try:
            from audio_recorder_streamlit import audio_recorder
            import speech_recognition as sr

            audio_bytes = audio_recorder(text="Click to record", icon_size="2x")

            if audio_bytes:
                st.audio(audio_bytes, format="audio/wav")
                recognizer = sr.Recognizer()
                try:
                    with sr.AudioFile(io.BytesIO(audio_bytes)) as source:
                        audio_data = recognizer.record(source)
                    transcribed_text = recognizer.recognize_google(audio_data)
                    st.info(f"🗣️ You said: \"{transcribed_text}\"")

                    ans, extra_table = handle_voice_command(transcribed_text)
                    st.success(f"🤖 {ans}")
                    if extra_table is not None:
                        st.dataframe(extra_table)
                    speak(ans)
                except sr.UnknownValueError:
                    st.warning("Could not understand the audio. Please try speaking again, clearly.")
                except sr.RequestError as e:
                    st.error(f"Speech recognition service error: {e}")
        except ImportError:
            st.error(
                "Voice recording packages are not installed yet. Please add "
                "'audio-recorder-streamlit' and 'SpeechRecognition' to requirements.txt."
            )

# ---------- QUICK QUESTIONS ----------
elif mode == "Quick Questions":
    with c1:
        query = st.selectbox(
            "Choose a question:",
            [
                "Select a question...",
                "What is the average exam score of students?",
                "How many total students are in the dataset?",
                "What is the overall passing status?",
            ],
        )
        if st.button("Ask Assistant"):
            if query == "What is the average exam score of students?" and SCORE_COL:
                ans = f"The average exam score of students is {df[SCORE_COL].mean():.2f}."
            elif query == "How many total students are in the dataset?":
                ans = f"There are a total of {len(df)} students recorded in the dataset."
            elif query == "What is the overall passing status?" and PASS_COL:
                pass_rate = (df[PASS_COL].astype(str).str.lower() == "pass").mean() * 100
                ans = f"Overall, {pass_rate:.1f} percent of students have passed."
            else:
                ans = "Sorry, that information is not available in this dataset."
            st.success(f"🤖 {ans}")
            speak(ans)

# ---------- COLUMN STATISTICS ----------
elif mode == "Column Statistics (mean, std, min, max)":
    with c1:
        col_choice = st.selectbox("Choose a variable:", numeric_cols)
        if st.button("Ask Assistant"):
            series = df[col_choice]
            ans = (
                f"For {col_choice}: mean is {series.mean():.2f}, "
                f"standard deviation is {series.std():.2f}, "
                f"minimum is {series.min():.2f}, and maximum is {series.max():.2f}."
            )
            st.success(f"🤖 {ans}")
            st.write(series.describe())
            speak(ans)

# ---------- CORRELATION ----------
elif mode == "Correlation between two variables":
    with c1:
        var_x = st.selectbox("Variable 1:", numeric_cols, index=0)
        var_y = st.selectbox("Variable 2:", numeric_cols, index=min(1, len(numeric_cols) - 1))
        if st.button("Ask Assistant"):
            corr = df[var_x].corr(df[var_y])
            strength = "a strong" if abs(corr) > 0.7 else ("a moderate" if abs(corr) > 0.3 else "a weak")
            direction = "positive" if corr > 0 else "negative"
            ans = f"The correlation between {var_x} and {var_y} is {corr:.3f}, which indicates {strength} {direction} relationship."
            st.success(f"🤖 {ans}")
            speak(ans)

# ---------- REGRESSION ----------
elif mode == "Regression line between two variables":
    with c1:
        var_x = st.selectbox("Independent variable (X):", numeric_cols, index=0, key="reg_x")
        var_y = st.selectbox("Dependent variable (Y):", numeric_cols, index=min(1, len(numeric_cols) - 1), key="reg_y")
        if st.button("Ask Assistant"):
            X = df[[var_x]].values
            y = df[var_y].values
            model = LinearRegression().fit(X, y)
            r2 = model.score(X, y)
            ans = (
                f"The fitted regression line is {var_y} = {model.coef_[0]:.3f} times {var_x} "
                f"plus {model.intercept_:.3f}, with an R-squared value of {r2:.3f}."
            )
            st.success(f"🤖 {ans}")
            fig, ax = plt.subplots()
            ax.scatter(df[var_x], df[var_y], alpha=0.3, s=10)
            x_line = np.linspace(df[var_x].min(), df[var_x].max(), 100)
            ax.plot(x_line, model.predict(x_line.reshape(-1, 1)), color="red", linewidth=2)
            ax.set_xlabel(var_x); ax.set_ylabel(var_y)
            ax.set_title(f"Regression: {var_y} vs {var_x}")
            st.pyplot(fig)
            speak(ans)

# ---------- STUDENT LOOKUP ----------
elif mode == "Look up a specific student":
    with c1:
        if ID_COL:
            student_id = st.text_input(f"Enter {ID_COL} (e.g. S00001):")
            if st.button("Ask Assistant") and student_id:
                match = df[df[ID_COL].astype(str).str.upper() == student_id.strip().upper()]
                if match.empty:
                    st.warning(f"No student found with ID {student_id}.")
                else:
                    row = match.iloc[0]
                    if SCORE_COL:
                        class_avg = df[SCORE_COL].mean()
                        comparison = "above" if row[SCORE_COL] > class_avg else "below"
                        ans = (
                            f"Student {student_id} has a {SCORE_COL} of {row[SCORE_COL]:.1f}, "
                            f"which is {comparison} the class average of {class_avg:.1f}."
                        )
                    else:
                        ans = f"Student {student_id} record found."
                    st.success(f"🤖 {ans}")
                    st.dataframe(match)
                    speak(ans)
        else:
            st.info("No student ID column found in this dataset.")

# ---------------------------------------------------------------
# 3. VISUAL ANALYTICS
# ---------------------------------------------------------------
with c2:
    st.subheader("📊 Visual Analytics")
    if len(numeric_cols) >= 2:
        chart_cols = st.multiselect("Choose columns to plot:", numeric_cols, default=numeric_cols[:2])
        if chart_cols:
            st.line_chart(df[chart_cols])
    else:
        st.dataframe(df.head())
