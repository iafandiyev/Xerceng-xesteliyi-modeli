# app.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Cancer AI Detection",
    page_icon="🩺",
    layout="wide"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>

.main {
    background: linear-gradient(135deg, #0f172a, #111827, #1e293b);
}

.title {
    font-size: 55px;
    font-weight: bold;
    text-align: center;
    color: white;
    animation: glow 2s infinite alternate;
}

@keyframes glow {
    from {
        text-shadow: 0 0 10px #00f2ff;
    }
    to {
        text-shadow: 0 0 30px #00f2ff;
    }
}

.card {
    background: rgba(255,255,255,0.08);
    padding: 25px;
    border-radius: 20px;
    backdrop-filter: blur(10px);
    box-shadow: 0 0 20px rgba(0,255,255,0.2);
    animation: fadeIn 1s ease-in-out;
}

@keyframes fadeIn {
    from {
        opacity: 0;
        transform: translateY(30px);
    }
    to {
        opacity: 1;
        transform: translateY(0px);
    }
}

.result-good {
    background: #16a34a;
    padding: 20px;
    border-radius: 15px;
    color: white;
    font-size: 25px;
    text-align: center;
    font-weight: bold;
}

.result-bad {
    background: #dc2626;
    padding: 20px;
    border-radius: 15px;
    color: white;
    font-size: 25px;
    text-align: center;
    font-weight: bold;
}

.metric-box {
    background: rgba(255,255,255,0.08);
    padding: 20px;
    border-radius: 15px;
    text-align: center;
    color: white;
}

</style>
""", unsafe_allow_html=True)

# ---------------- LOAD DATA ----------------
@st.cache_data
def load_data():
    df = pd.read_csv("Cancer_Data.csv")

    df.drop("id", axis=1, inplace=True)
    df.drop("Unnamed: 32", axis=1, inplace=True)

    df["diagnosis"] = df["diagnosis"].map({
        "B": 0,
        "M": 1
    })

    df.drop(
        columns=[
            "fractal_dimension_se",
            "texture_se",
            "symmetry_se",
            "concavity_se",
            "compactness_se",
            "fractal_dimension_worst",
            "symmetry_mean",
            "smoothness_mean"
        ],
        inplace=True
    )

    return df


# ---------------- MODEL ----------------
@st.cache_resource
def train_model():
    df = load_data()

    x = df.drop("diagnosis", axis=1)
    y = df["diagnosis"]

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42
    )

    model = SVC(probability=True)
    model.fit(x_train, y_train)

    predictions = model.predict(x_test)
    acc = accuracy_score(y_test, predictions)

    return model, acc, x.columns


model, accuracy, columns = train_model()


# ---------------- TITLE ----------------
st.markdown('<div class="title">🩺 Cancer AI Detection System</div>', unsafe_allow_html=True)

st.write("")
st.write("")


# ---------------- INFO SECTION ----------------
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(f'''
    <div class="metric-box">
        <h2>🎯 Accuracy</h2>
        <h1>{accuracy*100:.2f}%</h1>
    </div>
    ''', unsafe_allow_html=True)

with col2:
    st.markdown('''
    <div class="metric-box">
        <h2>🤖 Model</h2>
        <h1>SVM AI</h1>
    </div>
    ''', unsafe_allow_html=True)

with col3:
    st.markdown('''
    <div class="metric-box">
        <h2>📊 Dataset</h2>
        <h1>Breast Cancer</h1>
    </div>
    ''', unsafe_allow_html=True)


st.write("")
st.write("")


# ---------------- INPUT AREA ----------------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("🧬 Enter Medical Values")

input_data = []

cols = st.columns(3)

for i, column in enumerate(columns):
    with cols[i % 3]:
        value = st.number_input(
            column,
            value=0.0,
            format="%.5f"
        )
        input_data.append(value)

st.markdown('</div>', unsafe_allow_html=True)


# ---------------- PREDICTION ----------------
if st.button("🔍 Analyze Cancer Risk", use_container_width=True):

    input_array = np.array(input_data).reshape(1, -1)

    prediction = model.predict(input_array)[0]
    probability = model.predict_proba(input_array)[0]

    st.write("")

    if prediction == 0:
        st.markdown(f'''
        <div class="result-good">
            ✅ Benign (Non-Cancerous)<br><br>
            Confidence: {max(probability)*100:.2f}%
        </div>
        ''', unsafe_allow_html=True)

        st.balloons()

    else:
        st.markdown(f'''
        <div class="result-bad">
            ⚠️ Malignant (Cancer Risk Detected)<br><br>
            Confidence: {max(probability)*100:.2f}%
        </div>
        ''', unsafe_allow_html=True)


# ---------------- DATA PREVIEW ----------------
st.write("")
st.write("")

with st.expander("📁 Show Dataset"):
    st.dataframe(load_data())


# ---------------- FOOTER ----------------
st.write("")
st.markdown("""
<div style='text-align:center; color:gray;'>
Made with ❤️ using Streamlit + Scikit-Learn
</div>
""", unsafe_allow_html=True)
