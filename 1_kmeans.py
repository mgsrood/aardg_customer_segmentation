import pandas_gbq
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.cluster import KMeans
import numpy as np
from dotenv import load_dotenv

# Load the variables from the .env file
load_dotenv()

# Get the GCP keys
gc_keys = os.getenv("AARDG_GOOGLE_CREDENTIALS")
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = gc_keys

# Define full table id
kmeans_project_id = os.getenv("KMEANS_2_PROJECT_ID")
dataset_id = os.getenv("KMEANS_2_DATASET_ID")
table_id = os.getenv("KMEANS_2_TABLE_ID")
kmeans_full_table_id = f"{kmeans_project_id}.{dataset_id}.{table_id}"

# SQL-query
kmeans_query = f"""SELECT
  billing_email,
  ltv,
  total_order_count,
  avg_order_value,
  avg_time_between_orders,
  non_subscription_orders,
  subscription_orders
FROM `{kmeans_full_table_id}`
WHERE ltv != 0"""

# Use GBQ to get data from BigQuery
kmeans_table = pandas_gbq.read_gbq(kmeans_query, project_id=f'{kmeans_project_id}')

# Set month as index
kmeans_table.set_index("billing_email", inplace=True)

# Fill NA
kmeans_table = kmeans_table.fillna(0)

# Apply a logarithm transformation with a small constant added
def log_transform(x):
    return np.log(x + 1e-10)  # Adding a small constant (1e-10) to avoid log(0)

# Apply the logarithm transformation
kmeans_table_log = kmeans_table.apply(log_transform)

# Set up scaling
scaler = StandardScaler()
scaler.fit(kmeans_table_log)
scaled_table = scaler.transform(kmeans_table_log)
scaled_table_df = pd.DataFrame(scaled_table, index=kmeans_table.index, columns=kmeans_table.columns)

'''# Determine the optimal number of clusters
sse = {}

# Fit KMeans algorithm on k values between 1 and 11
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=333)
    kmeans.fit(scaled_table_df)
    sse[k] = kmeans.inertia_

# Add the title to the plot
plt.title('Elbow criterion method chart')

# Create and display a scatter plot
sns.pointplot(x=list(sse.keys()), y=list(sse.values()))

# Save the figure
plt.savefig('1_elbow_method.jpeg')

# Close the plot
plt.close()'''

# Initiate KMeans 
kmeans=KMeans(n_clusters=4, random_state=2)

# Fit the model on the pre-processed dataset
kmeans.fit(kmeans_table_log)

# Assign the generated labels to a new column
kmeans = kmeans_table.assign(segment = kmeans.labels_)

# Group by the segment label and calculate average column values
kmeans_averages = kmeans.groupby(['segment']).mean().round(0)

# Switch to numeric
kmeans_averages = kmeans_averages.astype(float)

# Set up the matplotlib figure
plt.figure(figsize=(10, 8))

# Create a heatmap on the average column values per each segment
sns.heatmap(kmeans_averages.T, cmap='YlOrRd', fmt='.2f', annot=True, linewidths=.5)

# Adjust the labels for readability
plt.yticks(rotation=0, fontsize=10)  
plt.xticks(rotation=45, fontsize=10)

# Save the figure
plt.savefig('1_heatmap_kmeans4.jpeg', bbox_inches='tight')

# Close the plot
plt.close()

# Count the number of customers in each segment
customer_counts_per_segment = kmeans['segment'].value_counts().sort_index()

# Now you can print or use the counts
print(customer_counts_per_segment)
