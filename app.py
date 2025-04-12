import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from io import BytesIO, StringIO
from fpdf import FPDF

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, r2_score
)

st.set_page_config(page_title="Employee Analyzer", layout="wide")

page = st.sidebar.selectbox("📂 Select Page", ["Preprocessing", "Model View", "Visualizations"])

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

    if page == "Preprocessing":
        st.header("🔧 Data Preprocessing")

        col1, col2, col3 = st.columns(3)
        if col1.button("🔍 Head"):
            st.dataframe(df.head())
        if col2.button("🔎 Tail"):
            st.dataframe(df.tail())
        if col3.button("ℹ️ Info"):
            buffer = StringIO()
            df.info(buf=buffer)
            s = buffer.getvalue()
            st.text(s)

        col4, col5, col6 = st.columns(3)
        if col4.button("🧼 Missing Values"):
            st.dataframe(df.isnull().sum())
        if col5.button("📊 Describe"):
            st.dataframe(df.describe())
        if col6.button("🧽 Cleansing"):
            st.write("### Handling Missing Values")
            st.write(df.isnull().sum())

        col7, col8, col9 = st.columns(3)
        if col7.button("📉 Reduction"):
            st.write("### Dropping constant columns")
            constant_columns = [col for col in df.columns if df[col].nunique() == 1]
            st.write(constant_columns)
        if col8.button("🔄 Transformation"):
            st.write("### Encoding Categorical Variables")
            st.write(df.select_dtypes(include='object').head())
        if col9.button("✨ Enrichment"):
            st.write("### Derived Features Example")
            if 'experience' in df.columns and 'salary' in df.columns:
                df['salary_per_year'] = df['salary'] / (df['experience'] + 1)
                st.dataframe(df[['experience', 'salary', 'salary_per_year']].head())
            else:
                st.info("Required columns not found for enrichment example.")

        if st.button("✔️ Validate"):
            st.write("### Checking data types and ranges")
            st.dataframe(df.dtypes)

        if st.button("📄 Generate Preprocessing Report (PDF)"):
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", size=12)
            pdf.cell(200, 10, txt="Preprocessing Report", ln=True)
            pdf.multi_cell(0, 10, txt=f"Columns: {', '.join(df.columns)}")
            pdf.multi_cell(0, 10, txt="\nMissing Values:\n" + df.isnull().sum().to_string())
            pdf_bytes = pdf.output(dest='S').encode('latin-1')

            st.download_button(
                label="📥 Download Preprocessing Report",
                data=pdf_bytes,
                file_name="preprocessing_report.pdf",
                mime="application/pdf"
            )

    elif page == "Model View":
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

                st.markdown("### 🧠 Overall Prediction: Employee Status")
                if problem_type == "regression":
                    status = "Will Continue" if prediction >= y.mean() else "Fired"
                else:
                    status = "Will Continue" if prediction == y.mode()[0] else "Fired"
                st.info(f"**Status:** {status}")

        else:
            st.warning("⚠️ No correlations between 0.4 and 0.8 found in numeric features.")

    elif page == "Visualizations":
        st.header("📊 Enhanced Visualizations (Power BI Style)")

        st.subheader("🔢 Summary KPIs")
        num_col = st.selectbox("Select column for KPI summary", df_numeric.columns)

        col1, col2, col3 = st.columns(3)
        col1.metric("Mean", round(df[num_col].mean(), 2))
        col2.metric("Median", round(df[num_col].median(), 2))
        col3.metric("Std Dev", round(df[num_col].std(), 2))

        if 'graph_count' not in st.session_state:
            st.session_state.graph_count = 1

        if st.button("➕ Add Graph"):
            st.session_state.graph_count += 1

        for i in range(st.session_state.graph_count):
            st.markdown(f"### 📈 Graph {i+1}")
            plot_type = st.selectbox(
                f"Select plot type for Graph {i+1}",
                ["Histogram", "Boxplot", "Correlation Heatmap", "Bar Chart (Mean)", "Line Chart", "Donut Pie Chart", "Treemap"],
                key=f"plot_{i}"
            )
            column = st.selectbox(f"Select column for Graph {i+1}", df.columns, key=f"col_{i}")

            if plot_type == "Histogram":
                fig, ax = plt.subplots()
                sns.histplot(df[column], kde=True, ax=ax)
                st.pyplot(fig)

            elif plot_type == "Boxplot":
                fig, ax = plt.subplots()
                sns.boxplot(x=df[column], ax=ax)
                st.pyplot(fig)

            elif plot_type == "Correlation Heatmap":
                fig, ax = plt.subplots(figsize=(10, 8))
                sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
                st.pyplot(fig)

            elif plot_type == "Bar Chart (Mean)":
                mean_vals = df_numeric.mean().sort_values(ascending=False)
                fig, ax = plt.subplots()
                mean_vals.plot(kind='bar', ax=ax)
                st.pyplot(fig)

            elif plot_type == "Line Chart":
                fig, ax = plt.subplots()
                df[column].plot(ax=ax)
                st.pyplot(fig)

            elif plot_type == "Donut Pie Chart":
                if df[column].dtype == 'object':
                    category_counts = df[column].value_counts()
                    fig, ax = plt.subplots()
                    ax.pie(category_counts, labels=category_counts.index, autopct='%1.1f%%', startangle=90, wedgeprops=dict(width=0.4))
                    ax.axis('equal')
                    st.pyplot(fig)

            elif plot_type == "Treemap":
                try:
                    import plotly.express as px
                    if df[column].dtype == 'object':
                        numeric_for_treemap = df_numeric.columns[0]
                        treemap_df = df.groupby(column)[numeric_for_treemap].sum().reset_index()
                        fig = px.treemap(treemap_df, path=[column], values=numeric_for_treemap)
                        st.plotly_chart(fig)
                except:
                    st.warning("Install Plotly to use Treemap visual")

            if st.button(f"📝 Summary of Graph {i+1}", key=f"summary_btn_{i}"):
                st.info(f"Graph Type: {plot_type}\n\nColumn: {column}\n\nData Summary:\n{df[column].describe()}")

        if st.button("📄 Download Visualization Report"):
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", size=12)
            pdf.cell(200, 10, txt="Visualization Summary", ln=True)
            pdf.multi_cell(0, 10, txt=f"KPI for {num_col}:\nMean: {df[num_col].mean()}\nMedian: {df[num_col].median()}\nStd Dev: {df[num_col].std()}")
            pdf.multi_cell(0, 10, txt="Correlation matrix summary:\n" + corr_matrix.to_string())
            pdf_bytes = pdf.output(dest='S').encode('latin-1')
            st.download_button(
                label="📥 Download Visual Report",
                data=pdf_bytes,
                file_name="visualization_report.pdf",
                mime="application/pdf"
            )

else:
    st.info("👈 Please upload a CSV file to begin.")
