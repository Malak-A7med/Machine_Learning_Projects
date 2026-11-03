# Customer Personality Analysis

Customer segmentation using unsupervised machine learning. The goal is to group customers by demographics, purchasing behavior, and campaign response, then turn those segments into actionable marketing recommendations.

## Dataset

[Customer Personality Analysis](https://www.kaggle.com/datasets/imakash3011/customer-personality-analysis) (`marketing_campaign.csv`): 2,240 customers, 29 features covering demographics, product spending, purchase channels, and campaign responses.

> The dataset is not included in this repo. Download it from Kaggle and place it next to the notebook.

## Workflow

1. **Data cleaning**: dropped constant columns, filled missing `Income` with the median, removed duplicates, merged rare `Marital_Status` / `Education` categories, and removed age and income outliers (2,240 → 2,047 customers).
2. **Feature engineering**: `Age`, `Family_Size`, `Total_Spending`, `Total_Purchases`, `Total_Accepted_Cmp`, `Customer_Tenure`.
3. **EDA**: distributions, correlations, and log transform of the skewed `Total_Spending`.
4. **Feature selection and scaling**: 6 clustering features, standardized with `StandardScaler`:
   `Age`, `Income`, `Total_Spending_log`, `Family_Size`, `NumDealsPurchases`, `NumWebVisitsMonth`.
   Other columns were kept for profiling only.
5. **Clustering and comparison**: KMeans, DBSCAN, Agglomerative, GMM, and Mean Shift.
6. **Visualization and profiling**: PCA projection and per-cluster profiles.

## Model Comparison

| Model         | Clusters | Silhouette | Davies-Bouldin | Calinski-Harabasz |
| ------------- | :------: | :--------: | :------------: | :---------------: |
| Mean Shift*   |    2     |   0.301    |     1.333      |      1021.8       |
| **KMeans**    |  **3**   |  **0.287** |   **1.353**    |     **877.0**     |
| GMM           |    3     |   0.255    |     1.518      |       765.8       |
| Agglomerative |    3     |   0.241    |     1.446      |       760.1       |
| DBSCAN        |    4     |   0.043    |     2.468      |       257.4       |

\*Mean Shift found only 2 real clusters and excluded 25 customers as outliers, so its scores are not directly comparable to the 3-cluster models.

**Selected model: KMeans (K=3).** Its metrics are close to the best, it covers all customers with balanced segments, and 3 segments are more useful for business profiling than 2.

## Customer Segments

| Segment                  | Customers | Avg Age | Avg Income | Avg Spending | Avg Purchases | Avg Accepted Campaigns |
| ------------------------ | :-------: | :-----: | :--------: | :----------: | :-----------: | :--------------------: |
| Budget Customers         |    793    |  40.8   |   $32.3K   |     $83      |      5.5      |          0.09          |
| High-Value Customers     |    721    |  46.9   |   $72.0K   |    $1,170    |     18.4      |          0.57          |
| Mid-Value Customers      |    533    |  49.1   |   $52.9K   |     $622     |     15.1      |          0.25          |

## Key Business Recommendations

- **Budget Customers**: entry-level bundles, coupons, and loyalty points. Avoid expensive premium campaigns.
- **High-Value Customers**: VIP/loyalty program, promote premium wine and meat, and prioritize them for new campaigns and retention.
- **Mid-Value Customers**: upsell with bundles and volume offers, and use online and family-oriented promotions to move them toward High-Value.

## Tech Stack

Python, Pandas, NumPy, Scikit-learn, SciPy, Matplotlib, Seaborn

## How to Run

```bash
pip install numpy pandas matplotlib seaborn scikit-learn scipy
jupyter notebook Customer_Personality_Analysis.ipynb
```
