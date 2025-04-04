import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

st.set_page_config(page_title="Employee Performance Predictor", layout="centered")
st.title("🏢 Employee Performance & Promotion Predictor")

uploaded_file = st.file_uploader("📂 Upload your HR dataset (.csv)", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.subheader("📊 Dataset Preview")
    st.dataframe(df.head())

    if 'PerformanceRating' not in df.columns:
        st.error("❌ Dataset must contain a column named 'PerformanceRating'.")
    else:
        # Drop rows with missing target
        df = df.dropna(subset=['PerformanceRating'])

        # Define features
        target = 'PerformanceRating'
        features = [col for col in df.columns if col != target]

        # Encode categorical features
        label_encoders = {}
        for col in df.select_dtypes(include='object').columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            label_encoders[col] = le

        X = df[features]
        y = df[target]

        # Class distribution chart
        st.subheader("📉 Class Distribution of PerformanceRating")
        st.bar_chart(y.value_counts().sort_index())

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Train model with class balancing
        model = RandomForestClassifier(random_state=42, class_weight='balanced')
        model.fit(X_train, y_train)

        # Evaluation
        st.subheader("📈 Model Evaluation")
        y_pred = model.predict(X_test)
        st.code(classification_report(y_test, y_pred))

        # Feature importance
        st.subheader("🔍 Feature Importance")
        importance_df = pd.DataFrame({
            'Feature': X.columns,
            'Importance': model.feature_importances_
        }).sort_values(by='Importance', ascending=False)
        st.dataframe(importance_df)

        # Manual Prediction
        st.subheader("🧪 Predict a New Employee's Performance")

        user_input = {}
        for col in X.columns:
            if col in label_encoders:
                options = label_encoders[col].classes_.tolist()
                choice = st.selectbox(f"{col}:", options)
                user_input[col] = label_encoders[col].transform([choice])[0]
            else:
                val = float(df[col].mean())
                user_input[col] = st.number_input(f"{col}:", value=val)

        input_df = pd.DataFrame([user_input])

        if st.button("🔮 Predict Performance Rating"):
            prediction = model.predict(input_df)[0]
            prediction_probs = model.predict_proba(input_df)[0]

            # Map numeric ratings to categories
            label_map = {
                1: "❌ Very Low Performer (Consider Exit or Retraining)",
                2: "⚠️ Low Performer (Needs Improvement)",
                3: "🙂 Average Performer (Stable)",
                4: "👍 Good Performer",
                5: "🏆 Excellent Performer (Eligible for Promotion/Increment)"
            }

            label_text = label_map.get(prediction, f"Rating: {prediction}")
            st.success(f"📌 Predicted Performance Rating: {prediction}")
            st.info(label_text)

            st.write("📊 Prediction Probabilities:")
            prob_df = pd.DataFrame({
                "PerformanceRating": model.classes_,
                "Probability": prediction_probs
            })
            st.dataframe(prob_df.set_index("PerformanceRating").T)
