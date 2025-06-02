import pandas as pd
import numpy as np

# --- Step 1: Load the dataset ---
# Ensure these CSVs are in the same folder as your script
movies = pd.read_csv("movies.csv")
ratings = pd.read_csv("ratings.csv")

# --- Step 2: Inspect the data ---
print("First 5 movie records:")
print(movies.head())

print("\nFirst 5 rating records:")
print(ratings.head())

# --- Step 3: Check for missing values ---
print("\nMissing values in 'movies' dataset:")
print(movies.isnull().sum())

print("\nMissing values in 'ratings' dataset:")
print(ratings.isnull().sum())

# --- Step 4: Clean the data ---
# Drop any rows with missing values (if any)
movies_clean = movies.dropna()
ratings_clean = ratings.dropna()

# Remove duplicates if present
movies_clean = movies_clean.drop_duplicates()
ratings_clean = ratings_clean.drop_duplicates()

# --- Step 5: Merge datasets for analysis (optional) ---
# Combine ratings with movie titles
merged = pd.merge(ratings_clean, movies_clean, on="movieId")

# --- Step 6: Generate basic statistical summaries ---

print("\nRatings Summary:")
print(ratings_clean['rating'].describe())

print("\nTotal number of unique users:", ratings_clean['userId'].nunique())
print("Total number of unique movies rated:", ratings_clean['movieId'].nunique())

# Average rating per movie (top 5)
avg_ratings = merged.groupby('title')['rating'].mean().sort_values(ascending=False)
print("\nTop 5 movies by average rating:")
print(avg_ratings.head())

# Number of ratings per movie (top 5)
rating_counts = merged['title'].value_counts().head()
print("\nTop 5 most rated movies:")
print(rating_counts)

# --- Step 7: Use NumPy for additional insights ---
ratings_array = ratings_clean['rating'].to_numpy()
print("\nOverall average rating (NumPy):", np.mean(ratings_array))
print("Standard deviation of ratings (NumPy):", np.std(ratings_array))