import streamlit as st
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


# ------------------ Page Configuration ------------------
st.set_page_config(
    page_title="Decision Tree Classification App",
    page_icon="🌳",
    layout="centered"
)


# ------------------ Load CSS ------------------
def load_css():
    with open("style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()


# ------------------ Title ------------------
st.title("🌳 Decision Tree Classification")
st.write(
    "This application demonstrates **Decision Tree classification** "
    "using the **Breast Cancer dataset**."
)


# ------------------ Load Dataset ------------------
data = load_breast_cancer()
X = data.data
y = data.target

df = pd.DataFrame(X, columns=data.feature_names)
df["Target"] = y


# ------------------ Dataset Preview ------------------
st.subheader("📊 Dataset Preview")
st.dataframe(df.head())


# ------------------ Sidebar ------------------
st.sidebar.header("⚙️ Model Parameters")

max_depth = st.sidebar.slider(
    "Select Max Depth of Tree",
    min_value=1,
    max_value=10,
    value=3
)

test_size = st.sidebar.slider(
    "Select test size (%)",
    min_value=10,
    max_value=40,
    value=20
)


# ------------------ Train-Test Split ------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=test_size / 100,
    random_state=42
)


# ------------------ Decision Tree Model ------------------
model = DecisionTreeClassifier(
    criterion="gini",
    max_depth=max_depth,
    random_state=42
)

model.fit(X_train, y_train)


# ------------------ Prediction & Accuracy ------------------
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)


# ------------------ Model Performance ------------------
st.subheader("📈 Model Performance")
st.success(f"Accuracy: {accuracy:.2f}")


# ------------------ User Input ------------------
st.subheader("🧪 Predict Cancer Type")

# Take first 5 important features for simplicity
feature_names = data.feature_names[:5]
user_input = []

for feature in feature_names:
    value = st.number_input(f"{feature}", min_value=0.0, step=0.1)
    user_input.append(value)

# Fill remaining features with mean values
if st.button("🔍 Predict"):
    input_data = np.zeros((1, X.shape[1]))
    input_data[0, :5] = user_input
    input_data[0, 5:] = X[:, 5:].mean(axis=0)

    prediction = model.predict(input_data)
    result = "Malignant (Cancerous)" if prediction[0] == 0 else "Benign (Non-Cancerous)"

    st.success(f"🩺 Prediction Result: **{result}**")


# ------------------ Footer ------------------
st.markdown("---")
st.markdown(
    "<center>Developed using Streamlit & Scikit-Learn</center>",
    unsafe_allow_html=True
)
