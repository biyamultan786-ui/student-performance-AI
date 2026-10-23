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


def strength_word(r):
    a = abs(r)
    if a >= 0.7:
        return "strong"
    if a >= 0.3:
        return "moderate"
    if a >= 0.1:
        return "weak"
    return "very weak"


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

    top, bottom = values.idxmax(), values.idxmin()
    if metric_col and metric_col != "Count":
        fig.explanation = (
            f"Figure: Bar chart of the average {metric_col} for each category of {cat_col}. "
            f"The chart compares {len(values)} groups. The highest average {metric_col} is observed "
            f"for '{top}' ({values.max():.2f}), while the lowest is observed for '{bottom}' "
            f"({values.min():.2f}). The difference between these two groups is "
            f"{values.max() - values.min():.2f}."
        )
    else:
        total = values.sum()
        fig.explanation = (
            f"Figure: Bar chart showing the number of students in each category of {cat_col}. "
            f"The dataset contains {int(total)} students across {len(values)} categories. "
            f"The largest group is '{top}' with {int(values.max())} students "
            f"({values.max() / total * 100:.1f}%), whereas the smallest group is '{bottom}' with "
            f"{int(values.min())} students ({values.min() / total * 100:.1f}%)."
        )
    return fig


def draw_hist(col, bins=20):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(df[col].dropna(), bins=bins, color="#54A24B", edgecolor="black")
    ax.set_xlabel(col)
    ax.set_ylabel("Number of students")
    ax.set_title(f"Histogram of {col}")
    fig.tight_layout()

    s = df[col].dropna()
    counts, edges = np.histogram(s, bins=bins)
    k = int(counts.argmax())
    skew = s.skew()
    if abs(skew) < 0.5:
        shape = "approximately symmetric (bell-shaped)"
    elif skew > 0:
        shape = "positively skewed (right-skewed), with a longer tail towards higher values"
    else:
        shape = "negatively skewed (left-skewed), with a longer tail towards lower values"
    fig.explanation = (
        f"Figure: Histogram of {col}. This graph shows how {col} is distributed across "
        f"{len(s)} students. The mean is {s.mean():.2f}, the median is {s.median():.2f} and the "
        f"standard deviation is {s.std():.2f}; values range from {s.min():.2f} to {s.max():.2f}. "
        f"The distribution is {shape} (skewness = {skew:.2f}). The most frequent interval is "
        f"{edges[k]:.2f} to {edges[k + 1]:.2f}, which contains {int(counts[k])} students "
        f"({counts[k] / len(s) * 100:.1f}%)."
    )
    return fig


def draw_scatter(x_col, y_col, with_line=True, color_by=None):
    fig, ax = plt.subplots(figsize=(7, 4))
    if color_by:
        for name, g in df.groupby(color_by):
            ax.scatter(g[x_col], g[y_col], alpha=0.4, s=12, label=str(name))
        ax.legend(title=color_by)
    else:
        ax.scatter(df[x_col], df[y_col], alpha=0.3, s=10, color="#4C78A8")
    clean = df[[x_col, y_col]].dropna()
    slope = None
    if with_line:
        model = LinearRegression().fit(clean[[x_col]].values, clean[y_col].values)
        slope = model.coef_[0]
        x_line = np.linspace(clean[x_col].min(), clean[x_col].max(), 100)
        ax.plot(x_line, model.predict(x_line.reshape(-1, 1)), color="red", linewidth=2)
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title(f"Scatter Plot: {y_col} vs {x_col}")
    fig.tight_layout()

    r = clean[x_col].corr(clean[y_col])
    if abs(r) < 0.1:
        rel = f"there is almost no linear relationship between {x_col} and {y_col}"
    else:
        direction = "positive" if r > 0 else "negative"
        rel = (
            f"there is a {strength_word(r)} {direction} linear relationship between "
            f"{x_col} and {y_col}"
        )
    text = (
        f"Figure: Scatter plot of {y_col} against {x_col}. Each point represents one student "
        f"({len(clean)} observations in total). The Pearson correlation coefficient is "
        f"r = {r:.3f}, which indicates that {rel}. The coefficient of determination is "
        f"R-squared = {r ** 2:.3f}, meaning that about {r ** 2 * 100:.1f}% of the variation in "
        f"{y_col} is linearly associated with {x_col}."
    )
    if slope is not None:
        text += (
            f" The red regression line has a slope of {slope:.3f}, meaning that on average "
            f"{y_col} changes by {slope:.3f} units for every one-unit increase in {x_col}."
        )
    text += " Note that correlation does not necessarily imply causation."
    fig.explanation = text
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

    s = df[col].dropna()
    q1, med, q3 = s.quantile([0.25, 0.5, 0.75])
    iqr = q3 - q1
    outliers = int(((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).sum())
    text = (
        f"Figure: Box plot of {col}"
        + (f" grouped by {group_col}" if group_col else "")
        + ". The box shows the interquartile range (the middle 50% of students), the line inside "
        "the box is the median, and points beyond the whiskers are potential outliers. "
    )
    if group_col:
        medians = df.groupby(group_col)[col].median()
        text += (
            f"The group with the highest median {col} is '{medians.idxmax()}' "
            f"({medians.max():.2f}), and the group with the lowest median is "
            f"'{medians.idxmin()}' ({medians.min():.2f}). "
        )
    text += (
        f"Overall, the median is {med:.2f}, Q1 is {q1:.2f}, Q3 is {q3:.2f} (IQR = {iqr:.2f}), "
        f"and {outliers} potential outliers were identified ({outliers / len(s) * 100:.1f}% "
        f"of the data)."
    )
    fig.explanation = text
    return fig


def draw_pie(col):
    counts = prepare_category(col).value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.pie(counts.values, labels=[str(i) for i in counts.index], autopct="%1.1f%%", startangle=90)
    ax.set_title(f"Pie Chart of {col}")
    fig.tight_layout()

    total = counts.sum()
    fig.explanation = (
        f"Figure: Pie chart showing the proportion of students in each category of {col}. "
        f"There are {len(counts)} categories in total. The largest share belongs to "
        f"'{counts.idxmax()}' ({counts.max() / total * 100:.1f}%), while the smallest share "
        f"belongs to '{counts.idxmin()}' ({counts.min() / total * 100:.1f}%)."
    )
    return fig


def draw_line(x_col, y_col):
    if df[x_col].nunique() > 30:
        grouped = df.groupby(pd.cut(df[x_col], 20), observed=True)[y_col].mean()
        xs = [i.mid for i in grouped.index]
    else:
        grouped = df.groupby(x_col)[y_col].mean()
        xs = list(grouped.index)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(xs, grouped.values, marker="o", color="#E45756")
    ax.set_xlabel(x_col)
    ax.set_ylabel(f"Average {y_col}")
    ax.set_title(f"Line Chart: Average {y_col} by {x_col}")
    ax.grid(alpha=0.3)
    fig.tight_layout()

    vals = np.array(grouped.values, dtype=float)
    r = np.corrcoef(np.array(xs, dtype=float), vals)[0, 1]
    if np.isnan(r) or abs(r) < 0.5:
        trend = f"does not show a clear consistent upward or downward trend as {x_col} increases"
    elif r > 0:
        trend = f"generally increases as {x_col} increases"
    else:
        trend = f"generally decreases as {x_col} increases"
    i_max, i_min = int(np.nanargmax(vals)), int(np.nanargmin(vals))
    fig.explanation = (
        f"Figure: Line chart of the average {y_col} across different values of {x_col}. "
        f"The average {y_col} {trend}. It ranges from {np.nanmin(vals):.2f} "
        f"(at {x_col} = {float(xs[i_min]):.2f}) to {np.nanmax(vals):.2f} "
        f"(at {x_col} = {float(xs[i_max]):.2f})."
    )
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

    pairs = [
        (corr.iloc[i, j], numeric_cols[i], numeric_cols[j])
        for i in range(n)
        for j in range(i + 1, n)
        if not np.isnan(corr.iloc[i, j])
    ]
    text = (
        f"Figure: Correlation heatmap of {n} numeric variables. Red cells indicate positive "
        f"correlations, blue cells indicate negative correlations, and cells close to white "
        f"indicate little or no linear relationship. "
    )
    if pairs:
        pos, neg = max(pairs), min(pairs)
        text += (
            f"The strongest positive correlation is between {pos[1]} and {pos[2]} "
            f"(r = {pos[0]:.2f}). "
        )
        if neg[0] < -0.05:
            text += (
                f"The strongest negative correlation is between {neg[1]} and {neg[2]} "
                f"(r = {neg[0]:.2f}). "
            )
        else:
            text += "No notable negative correlations were observed. "
    if SCORE_COL in numeric_cols:
        top = corr[SCORE_COL].drop(SCORE_COL).abs().sort_values(ascending=False).head(3)
        listing = ", ".join(f"{c} (r = {corr.loc[c, SCORE_COL]:.2f})" for c in top.index)
        text += f"The variables most strongly correlated with {SCORE_COL} are {listing}."
    fig.explanation = text
    return fig


_fig_counter = [0]


def show_fig(fig):
    st.pyplot(fig)
    explanation = getattr(fig, "explanation", None)
    if explanation:
        st.markdown("**📝 Explanation (you can copy this into your thesis):**")
        st.info(explanation)
    # High-quality (300 DPI) PNG download, suitable for a thesis
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
    title = fig.axes[0].get_title() if fig.axes else "graph"
    safe_name = re.sub(r"[^A-Za-z0-9]+", "_", title).strip("_") or "graph"
    _fig_counter[0] += 1
    st.download_button(
        "⬇️ Download graph (PNG)",
        data=buf.getvalue(),
        file_name=f"{safe_name}.png",
        mime="image/png",
        key=f"dl_fig_{_fig_counter[0]}",
    )
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
