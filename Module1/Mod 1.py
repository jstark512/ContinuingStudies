import pandas as pd
import numpy as np

# Step 1: Load the dataset
df = pd.read_csv("movies_metadata.csv", low_memory=False)

# Step 2: Fix and convert problematic columns
# Replace problematic 'budget' and 'revenue' values with NaN, then convert to numeric
df['budget'] = pd.to_numeric(df['budget'], errors='coerce')
df['revenue'] = pd.to_numeric(df['revenue'], errors='coerce')
df['runtime'] = pd.to_numeric(df['runtime'], errors='coerce')

# Optional: remove rows with extremely high/zero/negative values if needed
df = df[df['budget'] > 0]
df = df[df['revenue'] > 0]
df = df[df['runtime'] > 0]

# Step 3: Drop remaining rows with any NaNs in selected columns
df_clean = df.dropna(subset=['budget', 'revenue', 'runtime'])

# Step 4: Show data after cleaning
print("First 5 cleaned records:")
print(df_clean[['title', 'budget', 'revenue', 'runtime']].head())

# Step 5: Generate basic statistical summaries
print("\nStatistical Summary:")
print(df_clean[['budget', 'revenue', 'runtime']].describe())

# Additional summaries
print("\nAdditional Metrics:")
print(f"Average Runtime: {df_clean['runtime'].mean():.2f} minutes")
print(f"Median Budget: ${df_clean['budget'].median():,.2f}")
print(f"Maximum Revenue: ${df_clean['revenue'].max():,.2f}")