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

# Columns that make sense as "categories" (text columns, or numbers with few unique values)
category_cols = [
    c for c in df.columns
    if c != ID_COL and (c not in numeric_cols or df[c].nunique() <= 15)
]
# Text-only columns (used for grouping)
text_cols = [c for c in df.columns if c not in numeric_cols and c != ID_COL]

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
# CHART HELPERS (all return a matplotlib figure)
# ---------------------------------------------------------------
def prepare_category(col):
    """Turn a column into categories. Numeric columns with many values are binned."""
    s = df[col]
    if col in numeric_cols and s.nunique() > 15:
        return pd.cut(s, bins=10)
    return s


def draw_bar(cat_col, metric_col=None):
    cats = prepare_category(cat_col)
    if metric_col and metric_col != "Count":
        values = df.groupby(cats, observed=True)[metric_col].mean()
        ylabel = f"Average {metric_col}"
    else:
        values = cats.value_counts().sort_index()
        ylabel = "Number of students"
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(range(len(values)), values.values, color="#4C78A8", edgecolor="black")
    ax.set_xticks(range(len(values)))
    ax.set_xticklabels([str(i) for i in values.index], rotation=45, ha="right")
    ax.set_xlabel(cat_col)
    ax.set_ylabel(ylabel)
    ax.set_title(f"Bar Chart: {ylabel} by {cat_col}")
    fig.tight_layout()
    return fig


def draw_hist(col, bins=20):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(df[col].dropna(), bins=bins, color="#54A24B", edgecolor="black")
    ax.set_xlabel(col)
    ax.set_ylabel("Number of students")
    ax.set_title(f"Histogram of {col}")
    fig.tight_layout()
    return fig


def draw_scatter(x_col, y_col, with_line=True, color_by=None):
    fig, ax = plt.subplots(figsize=(7, 4))
    if color_by:
        for name, g in df.groupby(color_by):
            ax.scatter(g[x_col], g[y_col], alpha=0.4, s=12, label=str(name))
        ax.legend(title=color_by)
    else:
        ax.scatter(df[x_col], df[y_col], alpha=0.3, s=10, color="#4C78A8")
    if with_line:
        clean = df[[x_col, y_col]].dropna()
        model = LinearRegression().fit(clean[[x_col]].values, clean[y_col].values)
        x_line = np.linspace(clean[x_col].min(), clean[x_col].max(), 100)
        ax.plot(x_line, model.predict(x_line.reshape(-1, 1)), color="red", linewidth=2)
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title(f"Scatter Plot: {y_col} vs {x_col}")
    fig.tight_layout()
    return fig


def draw_box(col, group_col=None):
    fig, ax = plt.subplots(figsize=(7, 4))
    if group_col:
        groups, names = [], []
        for name, g in df.groupby(group_col):
            groups.append(g[col].dropna())
            names.append(str(name))
        ax.boxplot(groups)
        ax.set_xticklabels(names, rotation=45, ha="right")
        ax.set_xlabel(group_col)
        ax.set_title(f"Box Plot of {col} by {group_col}")
    else:
        ax.boxplot(df[col].dropna())
        ax.set_xticklabels([col])
        ax.set_title(f"Box Plot of {col}")
    ax.set_ylabel(col)
    fig.tight_layout()
    return fig


def draw_pie(col):
    counts = prepare_category(col).value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.pie(counts.values, labels=[str(i) for i in counts.index], autopct="%1.1f%%", startangle=90)
    ax.set_title(f"Pie Chart of {col}")
    fig.tight_layout()
    return fig


def draw_line(x_col, y_col):
    if df[x_col].nunique() > 30:
        grouped = df.groupby(pd.cut(df[x_col], 20), observed=True)[y_col].mean()
        xs = [i.mid for i in grouped.index]
    else:
        grouped = df.groupby(x_col)[y_col].mean()
        xs = grouped.index
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(xs, grouped.values, marker="o", color="#E45756")
    ax.set_xlabel(x_col)
    ax.set_ylabel(f"Average {y_col}")
    ax.set_title(f"Line Chart: Average {y_col} by {x_col}")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def draw_heatmap():
    corr = df[numeric_cols].corr()
    n = len(numeric_cols)
    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(numeric_cols, rotation=90, fontsize=8)
    ax.set_yticklabels(numeric_cols, fontsize=8)
    if n <= 12:
        for i in range(n):
            for j in range(n):
                ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax)
    ax.set_title("Correlation Heatmap")
    fig.tight_layout()
    return fig


def show_fig(fig):
    st.pyplot(fig)
    plt.close(fig)


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


def find_all_mentioned_any(text):
    """Same as above but for ALL columns (numeric and text), used for charts."""
    text_clean = text.lower().replace("_", " ")
    found = []
    for col in df.columns:
        if col == ID_COL:
            continue
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
        "scatterplot": "scatter plot",
        "boxplot": "box plot",
        "heat map": "heatmap",
        "bargraph": "bar graph",
        "history gram": "histogram",
        "histogram": "histogram",
    }
    fixed = text.lower()
    for wrong, right in replacements.items():
        fixed = fixed.replace(wrong, right)
    return fixed


def detect_chart_type(t):
    """Return the chart type mentioned in the sentence, or None."""
    if "heatmap" in t:
        return "heatmap"
    if "histogram" in t or "distribution" in t:
        return "histogram"
    if "scatter" in t:
        return "scatter"
    if "box" in t and any(k in t for k in ["plot", "graph", "chart"]):
        return "box"
    if "pie" in t:
        return "pie"
    if "line" in t and any(k in t for k in ["graph", "chart", "plot"]):
        return "line"
    if "bar" in t and any(k in t for k in ["graph", "chart", "plot"]):
        return "bar"
    return None


def handle_chart_command(text, chart_type):
    """Draw the requested chart from a spoken sentence."""
    cols = find_all_mentioned_any(text)
    if not cols:
        fallback = match_column(text)
        cols = [fallback] if fallback and fallback != ID_COL else []
    nums = [c for c in cols if c in numeric_cols]

    if chart_type == "heatmap":
        show_fig(draw_heatmap())
        return "Here is the correlation heatmap of all numeric variables.", None

    if chart_type == "histogram":
        if nums:
            show_fig(draw_hist(nums[0]))
            return f"Here is the histogram of {nums[0]}.", None
        return "Please tell me which numeric column you want the histogram for.", None

    if chart_type == "scatter":
        if len(nums) >= 2:
            show_fig(draw_scatter(nums[0], nums[1], with_line=False))
            return f"Here is the scatter plot of {nums[1]} against {nums[0]}.", None
        return "For a scatter plot, please mention two numeric column names.", None

    if chart_type == "line":
        if len(nums) >= 2:
            show_fig(draw_line(nums[0], nums[1]))
            return f"Here is the line chart of average {nums[1]} by {nums[0]}.", None
        return "For a line chart, please mention two numeric column names.", None

    if chart_type == "box":
        if nums:
            group = next((c for c in cols if c != nums[0] and c in category_cols), None)
            show_fig(draw_box(nums[0], group))
            return f"Here is the box plot of {nums[0]}.", None
        return "Please tell me which numeric column you want the box plot for.", None

    if chart_type == "pie":
        if cols:
            show_fig(draw_pie(cols[0]))
            return f"Here is the pie chart of {cols[0]}.", None
        return "Please tell me which column you want the pie chart for.", None

    if chart_type == "bar":
        if cols:
            metric = next((c for c in nums if c != cols[0]), None)
            show_fig(draw_bar(cols[0], metric))
            if metric:
                return f"Here is the bar chart of average {metric} by {cols[0]}.", None
            return f"Here is the bar chart of {cols[0]}.", None
        return "Please tell me which column you want the bar chart for.", None

    return "Sorry, I could not create that chart.", None


def handle_voice_command(text):
    """Understand the transcribed sentence and run the right analysis."""
    text = correct_common_mishearings(text)
    t = text.lower()

    # 0. Charts (bar graph, histogram, scatter plot, box plot, pie chart, line chart, heatmap)
    chart_type = detect_chart_type(t)
    if chart_type and "regression" not in t:
        return handle_chart_command(text, chart_type)

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
        "regression, average, a chart like 'bar graph of study hours', or a specific student ID.",
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
        "📈 Charts & Graphs (bar, histogram, scatter...)",
        "Look up a specific student",
    ],
)

c1, c2 = st.columns([1, 1])

# ---------- VOICE COMMAND MODE ----------
if mode == "🎤 Voice Command (speak your question)":
    with c1:
        st.write(
            "Click the microphone, speak your question clearly, then wait a moment "
            "for the transcription. Examples: *\"What is the correlation between "
            "study hours and final exam score?\"*, *\"Bar graph of pass fail\"*, "
            "*\"Histogram of final exam score\"*, *\"Scatter plot of study hours and final exam score\"*"
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

# ---------- CHARTS & GRAPHS ----------
elif mode == "📈 Charts & Graphs (bar, histogram, scatter...)":
    with c1:
        chart_type = st.selectbox(
            "Chart type:",
            [
                "Bar Chart",
                "Histogram",
                "Scatter Plot",
                "Box Plot",
                "Pie Chart",
                "Line Chart",
                "Correlation Heatmap",
            ],
        )

        fig = None

        if chart_type == "Bar Chart":
            cat = st.selectbox("Category (X axis):", category_cols or list(df.columns))
            metric = st.selectbox("Show:", ["Count"] + numeric_cols)
            if st.button("Draw Chart"):
                fig = draw_bar(cat, metric)

        elif chart_type == "Histogram":
            col = st.selectbox("Variable:", numeric_cols)
            bins = st.slider("Number of bins:", 5, 60, 20)
            if st.button("Draw Chart"):
                fig = draw_hist(col, bins)

        elif chart_type == "Scatter Plot":
            sx = st.selectbox("X variable:", numeric_cols, index=0, key="sc_x")
            sy = st.selectbox("Y variable:", numeric_cols, index=min(1, len(numeric_cols) - 1), key="sc_y")
            line = st.checkbox("Show regression line", value=True)
            color_by = st.selectbox("Colour points by (optional):", ["None"] + category_cols)
            if st.button("Draw Chart"):
                fig = draw_scatter(sx, sy, line, None if color_by == "None" else color_by)

        elif chart_type == "Box Plot":
            col = st.selectbox("Variable:", numeric_cols, key="box_col")
            group = st.selectbox("Group by (optional):", ["None"] + category_cols)
            if st.button("Draw Chart"):
                fig = draw_box(col, None if group == "None" else group)

        elif chart_type == "Pie Chart":
            col = st.selectbox("Category:", category_cols or list(df.columns), key="pie_col")
            if st.button("Draw Chart"):
                fig = draw_pie(col)

        elif chart_type == "Line Chart":
            lx = st.selectbox("X variable:", numeric_cols, index=0, key="ln_x")
            ly = st.selectbox("Y variable (average shown):", numeric_cols, index=min(1, len(numeric_cols) - 1), key="ln_y")
            if st.button("Draw Chart"):
                fig = draw_line(lx, ly)

        elif chart_type == "Correlation Heatmap":
            if st.button("Draw Chart"):
                fig = draw_heatmap()

        if fig is not None:
            show_fig(fig)

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
