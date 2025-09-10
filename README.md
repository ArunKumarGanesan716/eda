# Capstone EDA & Modeling

This project performs Exploratory Data Analysis (EDA) and prepares data for modeling in the context of financial investment data. The workflow supports the following capabilities:

- **A. Risk Tolerance Prediction (Classification)**
- **B. Customer Segmentation (Clustering with KMeans)**
- **C. Investment Product Recommendation (Multi-class Classification)**
- **D. Anomaly Detection (Isolation Forest)**
- **E. Fairness & Bias Analysis (Race / Gender / City Tier)**
- **F. Sentiment & Satisfaction Analysis (Comment → sentiment; link to satisfaction & risk)**
- **G. Data Cleaning, Feature Engineering, and Export**

## Data Sources
- `Combined_Investment_Data_100k_full.csv`: Main dataset (from [Kaggle](https://www.kaggle.com/code/emremsr/finance-data/input))
- Additional columns: Education Level, Annual Income, Employment Status, Risk Tolerance, Race, City Tier, Investment Avenue, Comment

## Main Files
- `Capstone_EDA_Modeling.ipynb`: Main notebook for EDA, feature engineering, modeling, and analysis
- `cleaned_with_features.csv`: Output dataset with engineered features (clusters, anomaly flag, sentiment)
- `SampleGenerator.py`: (Optional) Script for generating or augmenting sample data

## Key Steps
1. **Data Loading & Cleaning**: Handles missing values, normalizes column names, and caps outliers.
2. **Feature Engineering**: Adds sentiment scores, clusters, and anomaly flags.
3. **Modeling**: Implements classification, clustering, and anomaly detection using scikit-learn.
4. **Fairness & Bias Analysis**: Evaluates model performance across sensitive groups.
5. **Visualization**: Plots distributions, cluster projections, and feature importances.
6. **Export**: Saves the cleaned and feature-enriched dataset for downstream tasks.

## Requirements
- Python 3.8+
- pandas, numpy, matplotlib, scikit-learn

Install dependencies with:
```bash
pip install pandas numpy matplotlib scikit-learn
```

## Usage
1. Open `Capstone_EDA_Modeling.ipynb` in Jupyter or VS Code.
2. Run cells sequentially to perform EDA, feature engineering, and modeling.
3. The cleaned dataset will be saved as `cleaned_with_features.csv`.

## Notes
- The notebook is modular; you can run only the sections you need.
- Sentiment analysis is basic and can be improved with NLP libraries.
- Fairness analysis is included for key sensitive attributes.

---

*For questions or improvements, please open an issue or submit a pull request.*
