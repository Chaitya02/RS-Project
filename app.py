from flask import Flask, jsonify, request
from functools import wraps
import jwt
from datetime import datetime, timedelta
import pandas as pd
from knowledge_based_model import get_initial_recommendations, get_collaborative_recommendations
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sqlalchemy import desc, func
from flask import abort

# Move these to the top of the file, right after the imports and before app initialization
movies_df = None
ratings_df = None

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///recommender.db'  # Use PostgreSQL in production
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# JWT Configuration
JWT_SECRET = 'your-secret-key'  # In production, use environment variable
JWT_ALGORITHM = 'HS256'

# User Model
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(200), nullable=False)
    age = db.Column(db.Integer)
    gender = db.Column(db.String(10))
    occupation = db.Column(db.String(50))
    favorite_genres = db.Column(db.String(200))  # Stored as comma-separated string
    preferred_year_min = db.Column(db.Integer)
    preferred_year_max = db.Column(db.Integer)
    is_admin = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def set_password(self, password):
        self.password_hash = generate_password_hash(password)
        
    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

class UserLike(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    movie_id = db.Column(db.Integer, nullable=False)
    rating = db.Column(db.Float, nullable=True)  # Optional rating
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    user = db.relationship('User', backref=db.backref('likes', lazy=True))

# Authentication decorator
def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.headers.get('Authorization')
        if not token:
            return jsonify({'error': 'Token is missing'}), 401
        try:
            token = token.split(' ')[1]  # Remove 'Bearer ' prefix
            data = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
            current_user = User.query.get(data['user_id'])
            if not current_user:
                raise Exception('User not found')
        except Exception as e:
            return jsonify({'error': 'Invalid token'}), 401
        return f(current_user, *args, **kwargs)
    return decorated

@app.route('/signup', methods=['POST'])
def signup():
    try:
        data = request.json
        required_fields = ['username', 'email', 'password', 'age', 'gender', 
                         'occupation', 'favorite_genres', 'preferred_year_range']
        
        if not all(k in data for k in required_fields):
            return jsonify({'error': 'Missing required fields'}), 400
            
        if User.query.filter_by(username=data['username']).first():
            return jsonify({'error': 'Username already exists'}), 400
        
        # Handle favorite_genres whether it's a string or list
        favorite_genres = data['favorite_genres']
        if isinstance(favorite_genres, list):
            favorite_genres = ','.join(favorite_genres)
            
        user = User(
            username=data['username'],
            email=data['email'],
            age=data['age'],
            gender=data['gender'],
            occupation=data['occupation'],
            favorite_genres=favorite_genres,  # Now handles both string and list
            preferred_year_min=data['preferred_year_range'][0],
            preferred_year_max=data['preferred_year_range'][1]
        )
        user.set_password(data['password'])
        
        db.session.add(user)
        db.session.commit()
        
        return jsonify({
            'message': 'User created successfully',
            'user': {
                'username': user.username,
                'email': user.email,
                'age': user.age,
                'gender': user.gender,
                'occupation': user.occupation,
                'favorite_genres': user.favorite_genres.split(','),
                'preferred_year_range': [user.preferred_year_min, user.preferred_year_max]
            }
        }), 201
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500

@app.route('/login', methods=['POST'])
def login():
    try:
        data = request.json
        user = User.query.filter_by(username=data['username']).first()
        
        if not user or not user.check_password(data['password']):
            return jsonify({'error': 'Invalid credentials'}), 401
            
        token = jwt.encode({
            'user_id': user.id,
            'exp': datetime.utcnow() + timedelta(days=1)
        }, JWT_SECRET)
        
        return jsonify({'token': token})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/user/profile', methods=['GET'])
@token_required
def get_profile(current_user):
    return jsonify({
        'username': current_user.username,
        'email': current_user.email,
        'age': current_user.age,
        'gender': current_user.gender,
        'occupation': current_user.occupation,
        'favorite_genres': current_user.favorite_genres.split(','),
        'preferred_year_range': [current_user.preferred_year_min, current_user.preferred_year_max]
    })

@app.route('/recommendations/content', methods=['GET'])
@token_required
def content_based_recommendations(current_user):
    try:
        # Fetch user preferences
        favorite_genres = current_user.favorite_genres.split(',')
        preferred_year_min = current_user.preferred_year_min
        preferred_year_max = current_user.preferred_year_max
        
        # Filter movies_df by user's preferred year range
        filtered_movies = movies_df[
            (movies_df['year'] >= preferred_year_min) & 
            (movies_df['year'] <= preferred_year_max)
        ]
        
        # Create a TF-IDF vectorizer for the genres
        vectorizer = TfidfVectorizer(tokenizer=lambda x: x.split('|'))  # Assuming genres are pipe-separated
        tfidf_matrix = vectorizer.fit_transform(filtered_movies['genres'])
        
        # Generate a user profile vector based on favorite genres
        user_profile = vectorizer.transform([','.join(favorite_genres)])
        
        # Compute cosine similarity between user profile and movies
        similarity_scores = cosine_similarity(user_profile, tfidf_matrix).flatten()
        
        # Add similarity scores to the filtered movies DataFrame
        filtered_movies['similarity_score'] = similarity_scores
        
        # Sort movies by similarity scores
        recommended_movies = filtered_movies.sort_values(by='similarity_score', ascending=False)
        
        # Select the top N recommendations (e.g., 10)
        top_recommendations = recommended_movies.head(10)
        
        return jsonify(top_recommendations.to_dict('records'))
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/recommendations', methods=['GET'])
@token_required
def get_recommendations(current_user):
    try:
        user_likes = UserLike.query.filter_by(user_id=current_user.id).all()
        
        # Cold start: Use knowledge-based recommendations
        if len(user_likes) > 5:
            user_profile = {
                'age': current_user.age,
                'gender': current_user.gender,
                'occupation': current_user.occupation,
                'favorite_genres': current_user.favorite_genres.split(','),
                'year_min': current_user.preferred_year_min,
                'year_max': current_user.preferred_year_max
            }
            recommendations = get_initial_recommendations(user_profile, movies_df)
        else:
            # Use collaborative filtering once we have enough user interactions
            liked_movie_ids = [like.movie_id for like in user_likes]
            print(liked_movie_ids)
            recommendations = get_collaborative_recommendations(
                current_user.id, 
                ratings_df, 
                movies_df
            )
        
        return jsonify(recommendations.to_dict('records'))
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/interaction/like', methods=['POST'])
@token_required
def like_movie(current_user):
    try:
        data = request.json
        movie_id = data.get('movie_id')
        rating = data.get('rating')  # Optional rating
        
        if not movie_id:
            return jsonify({'error': 'Missing movie_id'}), 400
            
        existing_like = UserLike.query.filter_by(
            user_id=current_user.id,
            movie_id=movie_id
        ).first()
        
        if existing_like:
            return jsonify({'error': 'Movie already liked'}), 400
            
        new_like = UserLike(
            user_id=current_user.id,
            movie_id=movie_id,
            rating=rating
        )
        
        db.session.add(new_like)
        db.session.commit()
        
        return jsonify({'message': 'Interaction recorded successfully'})
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500

@app.route('/logout', methods=['POST'])
@token_required
def logout(current_user):
    # In a more complete implementation, you might want to blacklist the token
    return jsonify({'message': 'Successfully logged out'})

@app.route('/user/preferences', methods=['POST'])
@token_required
def update_preferences(current_user):
    try:
        data = request.json
        
        # Update allowed fields
        allowed_fields = ['age', 'gender', 'occupation', 'favorite_genres', 'preferred_year_range']
        for field in allowed_fields:
            if field in data:
                if field == 'favorite_genres':
                    genres = data[field]
                    if isinstance(genres, list):
                        genres = ','.join(genres)
                    current_user.favorite_genres = genres
                elif field == 'preferred_year_range':
                    current_user.preferred_year_min = data[field][0]
                    current_user.preferred_year_max = data[field][1]
                else:
                    setattr(current_user, field, data[field])
        
        db.session.commit()
        return jsonify({'message': 'Preferences updated successfully'})
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500

@app.route('/content', methods=['GET'])
@token_required
def get_content(current_user):
    try:
        page = request.args.get('page', 1, type=int)
        per_page = request.args.get('per_page', 20, type=int)
        
        # Get paginated movies from movies_df
        total_movies = len(movies_df)
        start_idx = (page - 1) * per_page
        end_idx = start_idx + per_page
        
        movies_page = movies_df.iloc[start_idx:end_idx]
        
        return jsonify({
            'content': movies_page.to_dict('records'),
            'total': total_movies,
            'page': page,
            'per_page': per_page,
            'total_pages': (total_movies + per_page - 1) // per_page
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/content/add', methods=['POST'])
@token_required
def add_content(current_user):
    global movies_df
    try:
        if not current_user.is_admin:
            return jsonify({'error': 'Admin access required'}), 403
            
        data = request.json
        required_fields = ['title', 'genres', 'year']
        
        if not all(k in data for k in required_fields):
            return jsonify({'error': 'Missing required fields'}), 400
            
        # Add new movie to movies_df
        new_movie = pd.DataFrame([{
            'movieId': len(movies_df) + 1,
            'title': data['title'],
            'genres': data['genres'],
            'year': data['year']
        }])
        
        movies_df = pd.concat([movies_df, new_movie], ignore_index=True)
        
        return jsonify({'message': 'Content added successfully'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/recommendations/trending', methods=['GET'])
@token_required
def get_trending(current_user):
    try:
        days = request.args.get('days', 30, type=int)
        limit = request.args.get('limit', 10, type=int)
        
        # Calculate trending based on recent ratings and interactions
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        trending = db.session.query(
            UserLike.movie_id,
            func.count(UserLike.id).label('interaction_count'),
            func.avg(UserLike.rating).label('avg_rating')
        ).filter(
            UserLike.created_at >= cutoff_date
        ).group_by(
            UserLike.movie_id
        ).order_by(
            desc('interaction_count')
        ).limit(limit).all()
        
        # Get movie details for trending items
        trending_movies = []
        for movie_id, count, avg_rating in trending:
            movie_data = movies_df[movies_df['movieId'] == movie_id].iloc[0].to_dict()
            movie_data['interaction_count'] = count
            movie_data['average_rating'] = float(avg_rating) if avg_rating else None
            trending_movies.append(movie_data)
            
        return jsonify(trending_movies)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/interaction/rate', methods=['POST'])
@token_required
def rate_movie(current_user):
    try:
        data = request.json
        movie_id = data.get('movie_id')
        rating = data.get('rating')
        
        if not movie_id or not rating:
            return jsonify({'error': 'Missing movie_id or rating'}), 400
            
        if not (0 <= rating <= 5):
            return jsonify({'error': 'Rating must be between 0 and 5'}), 400
            
        # Update existing rating or create new one
        like = UserLike.query.filter_by(
            user_id=current_user.id,
            movie_id=movie_id
        ).first()
        
        if like:
            like.rating = rating
        else:
            like = UserLike(
                user_id=current_user.id,
                movie_id=movie_id,
                rating=rating
            )
            db.session.add(like)
            
        db.session.commit()
        return jsonify({'message': 'Rating recorded successfully'})
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Create the database and tables
    with app.app_context():
        db.drop_all()
        db.create_all()
        print("Database initialized successfully!")
    
    # Load your DataFrames
    movies_df = pd.read_csv('./ml-32m/movies.csv')
    ratings_df = pd.read_csv('./ml-32m/ratings.csv')
    
    app.run(debug=True)
