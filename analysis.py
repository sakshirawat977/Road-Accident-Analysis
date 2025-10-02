import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Page Configuration
st.set_page_config(page_title="US Traffic Accident Analysis", layout="wide")

# Title
st.title("US Traffic Accident Analysis Dashboard")

# File Upload
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Rename columns for easier access
    df.columns = df.columns.str.strip().str.replace(' ', '_').str.replace('%', 'Pct')

    # Sidebar Navigation
    analysis_type = st.sidebar.radio(
        "Choose analysis type:",
        [
            "State-level Comparative Analysis",
            "Correlation and Relationship Analysis",
            #"Insurance Insights",
            "Cluster Analysis (K-Means)",
            "Principal Component Analysis (PCA)",
            "Top 5 / Bottom 5 Rankings",
            "Additional EDA"
        ]
    )

    # 1. State-level Comparative Analysis
    if analysis_type == "State-level Comparative Analysis":
        st.header("State-level Comparative Analysis")
        selected_metric = st.selectbox("Select metric:", df.columns[1:])
        ranked_df = df[["State", selected_metric]].sort_values(by=selected_metric, ascending=False)
        st.dataframe(ranked_df)
        fig, ax = plt.subplots(figsize=(14, 6))
        sns.barplot(x='State', y=selected_metric, data=ranked_df, palette='viridis')
        plt.xticks(rotation=90)
        st.pyplot(fig)

    # 2. Correlation and Relationship Analysis
    elif analysis_type == "Correlation and Relationship Analysis":
        st.header("Correlation and Relationship Analysis")
        x_feature = st.selectbox("Select X-axis feature:", df.columns[1:])
        y_feature = st.selectbox("Select Y-axis feature:", df.columns[1:])
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(x=x_feature, y=y_feature, data=df, hue='State', palette='tab10', legend=False)
        plt.xlabel(x_feature)
        plt.ylabel(y_feature)
        plt.title(f"{x_feature} vs. {y_feature}")
        st.pyplot(fig)

        st.subheader("Correlation Heatmap")
        corr = df.select_dtypes(include='number').corr()
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.heatmap(corr, annot=True, cmap='coolwarm')
        st.pyplot(fig)

    # 3. Insurance Insights
    elif analysis_type == "Insurance Insights":
        st.header("Insurance Insights")
        if 'Insurance_premiums' in df.columns and 'Losses_incurred' in df.columns:
            fig, ax = plt.subplots(figsize=(14, 6))
            sns.boxplot(x='State', y='Insurance_premiums', data=df, palette='Set2')
            plt.xticks(rotation=90)
            plt.title("Insurance Premiums by State")
            st.pyplot(fig)

            fig, ax = plt.subplots(figsize=(14, 6))
            sns.barplot(x='State', y='Losses_incurred', data=df, palette='Set3')
            plt.xticks(rotation=90)
            plt.title("Losses Incurred by State")
            st.pyplot(fig)

            corr_val = df['Insurance_premiums'].corr(df['Losses_incurred'])
            st.write(f"Correlation between insurance premiums and losses: **{corr_val:.2f}**")
        else:
            st.warning("Insurance premiums or losses columns not found in your data.")

    # 4. Cluster Analysis (K-Means)
    elif analysis_type == "Cluster Analysis (K-Means)":
        st.header("Cluster Analysis (K-Means)")
        features = st.multiselect("Select features for clustering:", df.select_dtypes(include='number').columns.tolist())
        if len(features) >= 2:
            X = df[features].dropna()
            k = st.slider("Number of clusters:", 2, 10, 3)
            kmeans = KMeans(n_clusters=k, random_state=42).fit(X)
            df['Cluster'] = kmeans.labels_
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.scatterplot(x=features[0], y=features[1], hue='Cluster', data=df, palette='tab10')
            st.pyplot(fig)
            st.dataframe(df[['State'] + features + ['Cluster']])
        else:
            st.warning("Please select at least two features.")

    # 5. Principal Component Analysis (PCA)
    elif analysis_type == "Principal Component Analysis (PCA)":
        st.header("Principal Component Analysis (PCA)")
        features = st.multiselect("Select features for PCA:", df.select_dtypes(include='number').columns.tolist())
        if len(features) >= 2:
            X = df[features].dropna()
            pca = PCA(n_components=2)
            components = pca.fit_transform(X)
            df['PCA1'] = components[:, 0]
            df['PCA2'] = components[:, 1]
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.scatterplot(x='PCA1', y='PCA2', hue='State', data=df, palette='tab20')
            plt.xlabel('PCA Component 1')
            plt.ylabel('PCA Component 2')
            st.pyplot(fig)
            st.write("Explained Variance Ratio:", pca.explained_variance_ratio_)
        else:
            st.warning("Please select at least two features.")

    # 6. Top 5 / Bottom 5 Rankings
    elif analysis_type == "Top 5 / Bottom 5 Rankings":
        st.header("Top 5 / Bottom 5 Rankings")
        selected_feature = st.selectbox("Select feature:", df.columns[1:])
        top5 = df.sort_values(by=selected_feature, ascending=False).head(5)
        bottom5 = df.sort_values(by=selected_feature, ascending=True).head(5)
        st.write("Top 5 States:")
        st.dataframe(top5[['State', selected_feature]])
        st.write("Bottom 5 States:")
        st.dataframe(bottom5[['State', selected_feature]])

    # 7. Additional EDA
    elif analysis_type == "Additional EDA":
        st.header("Additional Exploratory Data Analysis")

        st.subheader("Boxplot")
        selected_box_feature = st.selectbox("Select feature for boxplot:", df.select_dtypes(include='number').columns.tolist())
        fig, ax = plt.subplots(figsize=(14, 6))
        sns.boxplot(x='State', y=selected_box_feature, data=df, palette='Set1')
        plt.xticks(rotation=90)
        st.pyplot(fig)

        st.subheader("Histogram")
        selected_hist_feature = st.selectbox("Select feature for histogram:", df.select_dtypes(include='number').columns.tolist())
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(df[selected_hist_feature], bins=20, kde=True, color='skyblue')
        plt.title(f"Distribution of {selected_hist_feature}")
        st.pyplot(fig)

        st.subheader("Summary Statistics")
        st.dataframe(df.describe())

        st.subheader("Correlation Heatmap")
        corr = df.select_dtypes(include='number').corr()
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.heatmap(corr, annot=True, cmap='coolwarm')
        st.pyplot(fig)

else:
    st.info("Please upload a CSV file to start the analysis.")
