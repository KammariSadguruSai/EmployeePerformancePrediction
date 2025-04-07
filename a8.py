import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from io import BytesIO, StringIO
from fpdf import FPDF
import plotly.express as px
from PIL import Image
import tempfile
import os
import uuid

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

    if page == "Visualizations":
        st.header("📊 Enhanced Visualizations (Power BI Style)")

        st.subheader("🔢 Summary KPIs")
        num_col = st.selectbox("Select column for KPI summary", df_numeric.columns)

        col1, col2, col3 = st.columns(3)
        col1.metric("Mean", round(df[num_col].mean(), 2))
        col2.metric("Median", round(df[num_col].median(), 2))
        col3.metric("Std Dev", round(df[num_col].std(), 2))

        if 'graph_configs' not in st.session_state:
            st.session_state.graph_configs = []

        if st.button("➕ Add Graph"):
            st.session_state.graph_configs.append({"plot_type": "Histogram", "column": df.columns[0]})

        for i, config in enumerate(st.session_state.graph_configs):
            st.markdown(f"### 📈 Graph {i+1}")
            config['plot_type'] = st.selectbox(
                f"Select plot type for Graph {i+1}",
                ["Histogram", "Boxplot", "Correlation Heatmap", "Bar Chart (Mean)", "Line Chart", "Donut Pie Chart", "Treemap"],
                index=["Histogram", "Boxplot", "Correlation Heatmap", "Bar Chart (Mean)", "Line Chart", "Donut Pie Chart", "Treemap"].index(config['plot_type']) if config['plot_type'] in config else 0,
                key=f"plot_{i}"
            )
            config['column'] = st.selectbox(f"Select column for Graph {i+1}", df.columns, index=df.columns.get_loc(config['column']) if config['column'] in df.columns else 0, key=f"col_{i}")

            plot_type = config['plot_type']
            column = config['column']
            summary = None
            fig = None

            if plot_type == "Histogram":
                fig, ax = plt.subplots()
                sns.histplot(df[column], kde=True, ax=ax)

            elif plot_type == "Boxplot":
                fig, ax = plt.subplots()
                sns.boxplot(x=df[column], ax=ax)

            elif plot_type == "Correlation Heatmap":
                fig, ax = plt.subplots(figsize=(10, 8))
                sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)

            elif plot_type == "Bar Chart (Mean)":
                mean_vals = df_numeric.mean().sort_values(ascending=False)
                fig, ax = plt.subplots()
                mean_vals.plot(kind='bar', ax=ax)

            elif plot_type == "Line Chart":
                fig, ax = plt.subplots()
                df[column].plot(ax=ax)

            elif plot_type == "Donut Pie Chart":
                if df[column].dtype == 'object':
                    category_counts = df[column].value_counts()
                    fig, ax = plt.subplots()
                    ax.pie(category_counts, labels=category_counts.index, autopct='%1.1f%%', startangle=90, wedgeprops=dict(width=0.4))
                    ax.axis('equal')

            elif plot_type == "Treemap":
                if df[column].dtype == 'object':
                    numeric_for_treemap = df_numeric.columns[0]
                    treemap_df = df.groupby(column)[numeric_for_treemap].sum().reset_index()
                    fig = px.treemap(treemap_df, path=[column], values=numeric_for_treemap)

            if fig:
                if plot_type == "Treemap":
                    st.plotly_chart(fig)
                else:
                    st.pyplot(fig)

            if f"graph_summary_{i}" not in st.session_state:
                st.session_state[f"graph_summary_{i}"] = None

            if st.button(f"📝 Summary of Graph {i+1}", key=f"summary_btn_{i}"):
                summary = f"Graph Type: {plot_type}\n\nColumn: {column}\n\nData Summary:\n{df[column].describe()}"
                st.session_state[f"graph_summary_{i}"] = summary

            if st.session_state[f"graph_summary_{i}"]:
                st.info(st.session_state[f"graph_summary_{i}"])

        if st.button("📄 Download Visualization Report"):
            temp_dir = tempfile.mkdtemp()
            pdf = FPDF()
            pdf.set_auto_page_break(auto=True, margin=15)
            pdf.add_page()
            pdf.set_font("Arial", 'B', 16)
            pdf.cell(0, 10, f"Employee Visualization Report", ln=True, align='C')
            pdf.ln(10)
            pdf.set_font("Arial", size=12)
            pdf.multi_cell(0, 10, f"KPI Summary for {num_col}:\nMean = {df[num_col].mean():.2f}, Median = {df[num_col].median():.2f}, Std = {df[num_col].std():.2f}")
            pdf.ln(5)

            for i, config in enumerate(st.session_state.graph_configs):
                plot_type = config['plot_type']
                column = config['column']
                pdf.add_page()
                pdf.set_font("Arial", 'B', 14)
                pdf.cell(0, 10, f"Graph {i+1}: {plot_type} for {column}", ln=True)

                if plot_type == "Treemap":
                    numeric_for_treemap = df_numeric.columns[0]
                    treemap_df = df.groupby(column)[numeric_for_treemap].sum().reset_index()
                    fig = px.treemap(treemap_df, path=[column], values=numeric_for_treemap)
                    fig_path = os.path.join(temp_dir, f"treemap_{uuid.uuid4().hex}.png")
                    fig.write_image(fig_path)
                else:
                    fig, ax = plt.subplots()
                    if plot_type == "Histogram":
                        sns.histplot(df[column], kde=True, ax=ax)
                    elif plot_type == "Boxplot":
                        sns.boxplot(x=df[column], ax=ax)
                    elif plot_type == "Correlation Heatmap":
                        sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
                    elif plot_type == "Bar Chart (Mean)":
                        mean_vals = df_numeric.mean().sort_values(ascending=False)
                        mean_vals.plot(kind='bar', ax=ax)
                    elif plot_type == "Line Chart":
                        df[column].plot(ax=ax)
                    elif plot_type == "Donut Pie Chart":
                        if df[column].dtype == 'object':
                            category_counts = df[column].value_counts()
                            ax.pie(category_counts, labels=category_counts.index, autopct='%1.1f%%', startangle=90, wedgeprops=dict(width=0.4))
                            ax.axis('equal')
                    fig_path = os.path.join(temp_dir, f"graph_{uuid.uuid4().hex}.png")
                    plt.savefig(fig_path)
                    plt.close()

                pdf.image(fig_path, w=180)

                if st.session_state.get(f"graph_summary_{i}"):
                    pdf.ln(5)
                    pdf.set_font("Arial", size=11)
                    pdf.multi_cell(0, 10, f"Summary:\n{st.session_state[f'graph_summary_{i}']}")

            pdf_bytes = pdf.output(dest='S').encode('latin-1')
            st.download_button(
                label="📥 Download Visual Report",
                data=pdf_bytes,
                file_name="visualization_report.pdf",
                mime="application/pdf"
            )

else:
    st.info("👈 Please upload a CSV file to begin.")
