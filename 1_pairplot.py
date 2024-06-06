import pandas_gbq
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats
from dotenv import load_dotenv

# Load the variables from the .env file
load_dotenv()

# Get the GCP keys
gc_keys = os.getenv("AARDG_GOOGLE_CREDENTIALS")
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = gc_keys

# Define full table id
pairplot_project_id = os.getenv("KMEANS_2_PROJECT_ID")
dataset_id = os.getenv("KMEANS_2_DATASET_ID")
table_id = os.getenv("KMEANS_2_TABLE_ID")
pairplot_full_table_id = f"{pairplot_project_id}.{dataset_id}.{table_id}"

# SQL-query
pairplot_query = f"""SELECT
  billing_email,
  is_netherlands,
  is_randstad,
  ltv,
  total_order_count,
  avg_order_value,
  avg_time_between_orders,
  non_subscription_orders,
  subscription_orders,
  ever_subscription,
  active_subscription
FROM `{pairplot_full_table_id}`"""

# Use GBQ to get data from BigQuery
dataset = pandas_gbq.read_gbq(pairplot_query, project_id=f'{pairplot_project_id}')

# Set month as index
dataset.set_index("billing_email", inplace=True)

'''# Apply a box cox transformation
def boxcox_df(x):
    x_positive = x - x.min() + 1
    x_boxcox, _ = stats.boxcox(x_positive)
    return x_boxcox

dataset_boxcox = dataset.apply(boxcox_df, axis=0)'''

# Apply a log transformation
dataset_log = np.log(dataset)

# Plot the pairwise relationships between the variables
sns.pairplot(dataset_log, diag_kind='kde')

# Save the figure
plt.savefig('1_pairplot_log.jpeg')

# Close the plot
plt.close()

