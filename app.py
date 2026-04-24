import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Professional EDA Dashboard", layout="wide")

st.title("📊 Professional Automated EDA Dashboard")

# Upload file
file = st.file_uploader("📁 Upload your CSV file", type=["csv"])

if file is not None:
    df = pd.read_csv(file)

    # ================= BASIC INFO =================
    st.subheader("📌 Dataset Overview")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Rows", df.shape[0])
    col2.metric("Columns", df.shape[1])
    col3.metric("Missing Values", df.isnull().sum().sum())
    col4.metric("Duplicates", df.duplicated().sum())

    st.divider()

    st.subheader("👀 Preview")
    st.dataframe(df.head())

    # ================= DATA TYPES =================
    st.subheader("📊 Data Types")
    st.write(df.dtypes)

    # ================= COLUMN CLASSIFICATION =================
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(include=["object"]).columns

    # ================= NUMERIC ANALYSIS =================
    if len(numeric_cols) > 0:
        st.subheader("📈 Numeric Analysis")

        selected_num = st.selectbox("Select Numeric Column", numeric_cols)

        fig, ax = plt.subplots()
        ax.hist(df[selected_num].dropna(), bins=20, color="skyblue")
        st.pyplot(fig)

        fig2, ax2 = plt.subplots()
        ax2.boxplot(df[selected_num].dropna())
        st.pyplot(fig2)

        # Correlation
        if len(numeric_cols) > 1:
            st.subheader("🔥 Correlation Heatmap")

            corr = df[numeric_cols].corr()

            fig3, ax3 = plt.subplots()
            cax = ax3.imshow(corr, cmap="coolwarm")

            plt.colorbar(cax)

            ax3.set_xticks(range(len(numeric_cols)))
            ax3.set_yticks(range(len(numeric_cols)))
            ax3.set_xticklabels(numeric_cols, rotation=90)
            ax3.set_yticklabels(numeric_cols)

            st.pyplot(fig3)

    # ================= CATEGORICAL ANALYSIS =================
    if len(categorical_cols) > 0:
        st.subheader("🥧 Categorical Analysis")

        selected_cat = st.selectbox("Select Categorical Column", categorical_cols)

        value_counts = df[selected_cat].value_counts()

        fig4, ax4 = plt.subplots()
        ax4.pie(value_counts, labels=value_counts.index, autopct="%1.1f%%")
        st.pyplot(fig4)

    # ================= OUTLIER DETECTION =================
    st.subheader("📦 Outlier Detection (IQR Method)")

    if len(numeric_cols) > 0:
        col = numeric_cols[0]

        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1

        outliers = df[(df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)]

        st.write(f"Outliers in {col}: {len(outliers)}")

    # ================= AUTO INSIGHTS =================
    st.subheader("🧠 Auto Insights")

    insights = []

    insights.append(f"Dataset has {df.shape[0]} rows and {df.shape[1]} columns.")

    if df.isnull().sum().sum() > 0:
        insights.append("Dataset contains missing values.")

    if df.duplicated().sum() > 0:
        insights.append("Dataset contains duplicate rows.")

    if len(categorical_cols) > 0:
        top_cat = categorical_cols[0]
        insights.append(f"Most frequent category in {top_cat}: {df[top_cat].mode()[0]}")

    for i in insights:
        st.write("✔", i)

    st.success("EDA Completed Successfully 🎉")

    