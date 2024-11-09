from flask import Flask, jsonify, request
from functools import wraps
import jwt
from datetime import datetime, timedelta
import pandas as pd
from knowledge_based_model import get_initial_recommendations, get_collaborative_recommendations
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash

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

@app.route('/recommendations', methods=['GET'])
@token_required
def get_recommendations(current_user):
    try:
        user_likes = UserLike.query.filter_by(user_id=current_user.id).all()
        
        # Cold start: Use knowledge-based recommendations
        if len(user_likes) < 5:
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

if __name__ == '__main__':
    # Create the database and tables
    with app.app_context():
        # Drop all tables first to ensure clean slate
        db.drop_all()
        # Create all tables
        db.create_all()
        print("Database initialized successfully!")
        
    # Load your DataFrames
    movies_df = pd.read_csv('./ml-32m/movies.csv')
    ratings_df = pd.read_csv('./ml-32m/ratings.csv')
    
    app.run(debug=True)
