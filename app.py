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

# Detect key columns automatically so the app works even if column
# names differ slightly between dataset versions.
def find_col(possible_names):
    for name in possible_names:
        if name in df.columns:
            return name
    return None

SCORE_COL = find_col(["final_exam_score", "exam_score", "score"])
PASS_COL = find_col(["pass_fail", "passed", "pass_status"])
ID_COL = find_col(["student_id", "id"])

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

# ---------------------------------------------------------------
# 2. TOP SUMMARY METRICS
# ---------------------------------------------------------------
col1, col2, col3 = st.columns(3)
col1.metric("Total Students", len(df))
if SCORE_COL:
    col2.metric("Avg Exam Score", f"{df[SCORE_COL].mean():.1f}")
col3.metric("Dataset Columns", len(df.columns))

st.markdown("---")


def speak(text):
    """Generate and play a voice response for the given text."""
    try:
        tts = gTTS(text=text, lang="en")
        tts.save("voice_response.mp3")
        with open("voice_response.mp3", "rb") as f:
            audio_bytes = f.read()
        st.audio(audio_bytes, format="audio/mp3")
    except Exception as e:
        st.warning(f"Voice output unavailable: {e}")


# ---------------------------------------------------------------
# 3. MODE SELECTOR (this is what makes the assistant "advanced")
# ---------------------------------------------------------------
st.subheader("🎙️ Ask the AI Assistant")

mode = st.radio(
    "What would you like to ask?",
    [
        "Quick Questions",
        "Column Statistics (mean, std, min, max)",
        "Correlation between two variables",
        "Regression line between two variables",
        "Look up a specific student",
    ],
    horizontal=False,
)

c1, c2 = st.columns([1, 1])

# ---------- 3a. QUICK QUESTIONS ----------
if mode == "Quick Questions":
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

# ---------- 3b. COLUMN STATISTICS ----------
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

# ---------- 3c. CORRELATION ----------
elif mode == "Correlation between two variables":
    with c1:
        var_x = st.selectbox("Variable 1:", numeric_cols, index=0)
        var_y = st.selectbox("Variable 2:", numeric_cols, index=min(1, len(numeric_cols) - 1))
        if st.button("Ask Assistant"):
            corr = df[var_x].corr(df[var_y])
            if abs(corr) > 0.7:
                strength = "a strong"
            elif abs(corr) > 0.3:
                strength = "a moderate"
            else:
                strength = "a weak"
            direction = "positive" if corr > 0 else "negative"
            ans = (
                f"The correlation between {var_x} and {var_y} is {corr:.3f}, "
                f"which indicates {strength} {direction} relationship."
            )
            st.success(f"🤖 {ans}")
            speak(ans)

# ---------- 3d. REGRESSION LINE ----------
elif mode == "Regression line between two variables":
    with c1:
        var_x = st.selectbox("Independent variable (X):", numeric_cols, index=0, key="reg_x")
        var_y = st.selectbox(
            "Dependent variable (Y):", numeric_cols,
            index=min(1, len(numeric_cols) - 1), key="reg_y",
        )
        if st.button("Ask Assistant"):
            X = df[[var_x]].values
            y = df[var_y].values
            model = LinearRegression().fit(X, y)
            slope = model.coef_[0]
            intercept = model.intercept_
            r2 = model.score(X, y)

            ans = (
                f"The fitted regression line is {var_y} = {slope:.3f} times {var_x} "
                f"plus {intercept:.3f}, with an R-squared value of {r2:.3f}."
            )
            st.success(f"🤖 {ans}")

            fig, ax = plt.subplots()
            ax.scatter(df[var_x], df[var_y], alpha=0.3, s=10)
            x_line = np.linspace(df[var_x].min(), df[var_x].max(), 100)
            y_line = model.predict(x_line.reshape(-1, 1))
            ax.plot(x_line, y_line, color="red", linewidth=2)
            ax.set_xlabel(var_x)
            ax.set_ylabel(var_y)
            ax.set_title(f"Regression: {var_y} vs {var_x}")
            st.pyplot(fig)

            speak(ans)

# ---------- 3e. STUDENT LOOKUP ----------
elif mode == "Look up a specific student":
    with c1:
        if ID_COL:
            student_id = st.text_input(f"Enter {ID_COL} (e.g. S00001):")
            if st.button("Ask Assistant") and student_id:
                match = df[df[ID_COL].astype(str).str.upper() == student_id.strip().upper()]
                if match.empty:
                    ans = f"No student found with ID {student_id}."
                    st.warning(ans)
                else:
                    row = match.iloc[0]
                    details = ", ".join(
                        f"{c}: {row[c]}" for c in df.columns if c != ID_COL
                    )
                    if SCORE_COL:
                        class_avg = df[SCORE_COL].mean()
                        comparison = (
                            "above" if row[SCORE_COL] > class_avg else "below"
                        )
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
# 4. VISUAL ANALYTICS (right column, always visible)
# ---------------------------------------------------------------
with c2:
    st.subheader("📊 Visual Analytics")
    if len(numeric_cols) >= 2:
        chart_cols = st.multiselect(
            "Choose columns to plot:", numeric_cols, default=numeric_cols[:2]
        )
        if chart_cols:
            st.line_chart(df[chart_cols])
    else:
        st.dataframe(df.head())
