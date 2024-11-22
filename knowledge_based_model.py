import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

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
    """
    Get movie recommendations using collaborative filtering
    """
    # Create user-item matrix
    user_item_matrix = ratings_df.pivot(
        index='userId',
        columns='movieId',
        values='rating'
    ).fillna(0)
    
    # Calculate user similarity
    user_similarity = cosine_similarity(user_item_matrix)
    
    # Find similar users
    user_idx = user_item_matrix.index.get_loc(user_id)
    similar_users = np.argsort(user_similarity[user_idx])[-10:][::-1]
    
    # Get recommendations based on similar users
    recommendations = []
    for similar_user in similar_users:
        user_movies = user_item_matrix.iloc[similar_user]
        unseen_movies = user_movies[user_movies > 0].index
        recommendations.extend(unseen_movies)
    
    # Remove duplicates and get top N
    recommendations = list(dict.fromkeys(recommendations))[:n_recommendations]
    
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