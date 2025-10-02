# Road-Accident-Analysis
This is an interactive Streamlit dashboard to explore and analyze US traffic accident data. Users can upload a CSV file with predefined columns and perform various analyses including visualizations, clustering, dimensionality reduction (PCA), and rankings. The app is designed to help users quickly understand patterns, relationships, and trends in traffic accident data at a state level.

# Features : 
State-level Comparative Analysis: Rank and visualize selected metrics across states using bar plots.
Correlation and Relationship Analysis: Explore relationships between numeric features with scatter plots and correlation heatmaps.
Cluster Analysis (K-Means): Group states into clusters based on selected numeric features.
Principal Component Analysis (PCA): Reduce dimensionality of numeric features to identify patterns and visualize components.
Top 5 / Bottom 5 Rankings: Quickly identify top-performing and worst-performing states for any metric.
Additional Exploratory Data Analysis (EDA) :
1. Boxplots and histograms of numeric features
2. Summary statistics
3. Correlation heatmaps

# CSV Requirements
The app requires specific columns to function properly. Make sure your CSV includes:
State, Numeric columns for analysis (like Accidents, Fatalities, etc.)

# Tech Stack
Python
Streamlit (for the dashboard)
Pandas (for data handling)
Matplotlib & Seaborn (for visualization)
scikit-learn (KMeans & PCA)
