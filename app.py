import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, r2_score
)

st.set_page_config(page_title="Employee Analyzer", layout="wide")

# Sidebar Navigation
page = st.sidebar.selectbox("📂 Select Page", ["Model View", "Visualizations"])

st.title("👩‍💼 Employee Performance & Retention Dashboard")

uploaded_file = st.file_uploader("📁 Upload Your HR CSV File", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("✅ File uploaded successfully!")

    df_numeric = df.select_dtypes(include=['int64', 'float64'])
    corr_matrix = df_numeric.corr()

    def get_high_corr_targets(corr_matrix):
        high_corr_pairs = {}
        for column in corr_matrix.columns:
            high_corr = corr_matrix[column][(corr_matrix[column] >= 0.4) & (corr_matrix[column] < 0.8)]
            if not high_corr.empty:
                high_corr_pairs[column] = high_corr.sort_values(ascending=False)
        return high_corr_pairs

    high_corr_dict = get_high_corr_targets(corr_matrix)

    if page == "Model View":
        if high_corr_dict:
            target = st.selectbox("🎯 Select Target Variable for Prediction", list(high_corr_dict.keys()))

            if target:
                st.markdown(f"#### 🔍 Top Correlated Features with `{target}`")
                top_corr_features = high_corr_dict[target]
                st.dataframe(top_corr_features)

                feature_list = top_corr_features.index.tolist()

                st.markdown("### ✍️ Input Feature Values")
                user_input = []
                for feature in feature_list:
                    val = st.number_input(f"{feature}", value=float(df[feature].mean()))
                    user_input.append(val)

                X = df[feature_list]
                y = df[target]

                # Determine if classification or regression
                if y.nunique() <= 10 and y.dtype in ['int64', 'object']:
                    problem_type = "classification"
                else:
                    problem_type = "regression"

                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                if problem_type == "classification":
                    model = RandomForestClassifier(random_state=42)
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)

                    accuracy = accuracy_score(y_test, y_pred)
                    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
                    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

                    input_array = np.array(user_input).reshape(1, -1)
                    prediction = model.predict(input_array)[0]

                    st.success(f"🔮 Predicted `{target}`: **{prediction}**")
                    st.markdown("### 📊 Evaluation Metrics (Classification)")
                    st.write(f"**Accuracy:** {accuracy:.2f}")
                    st.write(f"**Precision:** {precision:.2f}")
                    st.write(f"**Recall:** {recall:.2f}")
                    st.write(f"**F1-Score:** {f1:.2f}")

                else:
                    model = RandomForestRegressor(random_state=42)
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)

                    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                    r2 = r2_score(y_test, y_pred)

                    input_array = np.array(user_input).reshape(1, -1)
                    prediction = model.predict(input_array)[0]

                    st.success(f"📈 Predicted `{target}`: **{round(prediction, 2)}**")
                    st.markdown("### 📊 Evaluation Metrics (Regression)")
                    st.write(f"**RMSE:** {rmse:.2f}")
                    st.write(f"**R² Score:** {r2:.2f}")

                # Final binary prediction: Continue or Fired
                st.markdown("### 🧠 Overall Prediction: Employee Status")
                if problem_type == "regression":
                    status = "Will Continue" if prediction >= y.mean() else "Fired"
                else:
                    status = "Will Continue" if prediction == y.mode()[0] else "Fired"
                st.info(f"**Status:** {status}")

        else:
            st.warning("⚠️ No correlations between 0.4 and 0.8 found in numeric features.")

    elif page == "Visualizations":
        st.subheader("📈 Correlation Heatmap")
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5, ax=ax1)
        st.pyplot(fig1)

        st.subheader("📊 Pairplot (on top correlated features)")
        if high_corr_dict:
            top_features = list(high_corr_dict.keys())[:5]
            fig2 = sns.pairplot(df_numeric[top_features])
            st.pyplot(fig2)
        else:
            st.info("No high-correlation features to display pairplot.")
else:
    st.info("👈 Please upload a CSV file to begin.")
