import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix

# Load the datasets
movies_df = pd.read_csv('ml-32m/movies.csv')
ratings_df = pd.read_csv('ml-32m/ratings.csv')

# Create genre features
def create_genre_matrix(movies_df):
    # Split the genres and create dummy variables
    genres = movies_df['genres'].str.get_dummies('|')
    return genres

# Create initial user profile based on demographic information
def create_user_profile(age, gender, occupation, favorite_genres, preferred_year_range):
    # Create a dictionary to store user preferences
    user_profile = {
        'age': age,
        'gender': gender,
        'occupation': occupation,
        'favorite_genres': favorite_genres,
        'year_min': preferred_year_range[0],
        'year_max': preferred_year_range[1]
    }
    return user_profile

# Extract year from movie title
def extract_year(title):
    try:
        year = int(title.strip()[-5:-1])
    except:
        year = None
    return year

# Get movie recommendations based on user profile
def get_initial_recommendations(user_profile, movies_df, n_recommendations=10):
    # Extract years from titles
    movies_df['year'] = movies_df['title'].apply(extract_year)
    
    # Filter by year range
    year_filtered = movies_df[
        (movies_df['year'] >= user_profile['year_min']) & 
        (movies_df['year'] <= user_profile['year_max'])
    ]
    
    # Create genre matrix
    genre_matrix = create_genre_matrix(year_filtered)
    
    # Create user genre preferences vector
    user_genres = np.zeros(len(genre_matrix.columns))
    for genre in user_profile['favorite_genres']:
        if genre in genre_matrix.columns:
            user_genres[genre_matrix.columns.get_loc(genre)] = 1
    
    # Calculate similarity between user preferences and movies
    similarities = cosine_similarity([user_genres], genre_matrix)[0]
    
    # Get top N recommendations
    movie_indices = np.argsort(similarities)[-n_recommendations:][::-1]
    recommendations = year_filtered.iloc[movie_indices]
    
    return recommendations[['title', 'genres']]

def get_collaborative_recommendations(user_id, ratings_df, movies_df, n_recommendations=10):
    from scipy.sparse import csr_matrix
    from sklearn.neighbors import NearestNeighbors

    # Reduce dataset: filter active users and popular movies
    min_user_ratings = 200
    min_movie_ratings = 100

    active_users = ratings_df['userId'].value_counts()
    active_users = active_users[active_users >= min_user_ratings].index
    ratings_df = ratings_df[ratings_df['userId'].isin(active_users)]

    popular_movies = ratings_df['movieId'].value_counts()
    popular_movies = popular_movies[popular_movies >= min_movie_ratings].index
    ratings_df = ratings_df[ratings_df['movieId'].isin(popular_movies)]

    # Create sparse user-item matrix
    user_item_matrix = csr_matrix((
        ratings_df['rating'], 
        (ratings_df['userId'], ratings_df['movieId'])
    ))

    # Get user index mapping
    user_idx = ratings_df['userId'].drop_duplicates().reset_index(drop=True)
    if user_id not in user_idx.values:
        raise ValueError(f"User ID {user_id} not found in the dataset.")
    user_index = user_idx[user_idx == user_id].index[0]

    # Fit a NearestNeighbors model
    knn = NearestNeighbors(metric='cosine', algorithm='brute')
    knn.fit(user_item_matrix)

    # Find the nearest neighbors for the target user
    distances, indices = knn.kneighbors(user_item_matrix[user_index], n_neighbors=10)

    # Get recommendations from similar users
    recommendations = set()
    for similar_user in indices.flatten():
        if similar_user != user_index:  # Skip self
            user_movies = user_item_matrix[similar_user].indices
            recommendations.update(user_movies)

    # Filter and limit recommendations
    recommendations = list(recommendations)[:n_recommendations]
    return movies_df[movies_df['movieId'].isin(recommendations)][['title', 'genres']]


# Content-based recommendation system
def get_content_based_recommendations(movie_id, movies_df, n_recommendations=10):
    """
    Get movie recommendations using content-based filtering.
    Recommendations are based on the similarity of genres between movies.

    Parameters:
        movie_id (int): ID of the movie the user likes.
        movies_df (DataFrame): Movies dataset with columns ['movieId', 'title', 'genres'].
        n_recommendations (int): Number of recommendations to return.

    Returns:
        DataFrame: Recommended movies with title and genres.
    """
    # Create genre matrix
    genre_matrix = create_genre_matrix(movies_df)
    
    # Get the index of the target movie
    movie_idx = movies_df[movies_df['movieId'] == movie_id].index[0]
    
    # Calculate similarity between the target movie and all other movies
    similarities = cosine_similarity([genre_matrix.iloc[movie_idx]], genre_matrix)[0]
    
    # Get the indices of the most similar movies
    similar_movie_indices = np.argsort(similarities)[-n_recommendations-1:][::-1]  # Exclude the target movie itself
    similar_movie_indices = [idx for idx in similar_movie_indices if idx != movie_idx][:n_recommendations]
    
    # Fetch recommended movies
    recommendations = movies_df.iloc[similar_movie_indices]
    
    return recommendations[['title', 'genres']]

# Example usage
if __name__ == "__main__":
    # Create a sample user profile
    user_profile = create_user_profile(
        age=25,
        gender='M',
        occupation='student',
        favorite_genres=['Action', 'Sci-Fi', 'Adventure'],
        preferred_year_range=(2000, 2023)
    )
    
    # Get recommendations
    recommendations = get_initial_recommendations(user_profile, movies_df)
    print("Initial movie recommendations:")
    print(recommendations)