# Movie Recommendation System

A personalized movie recommendation system built with Flask, SQLAlchemy, and machine learning algorithms. The system provides both knowledge-based and collaborative filtering recommendations based on user preferences and interactions.

## Features

- **User Authentication**
  - Secure signup and login with JWT tokens
  - Password hashing for security
  - Profile management

- **Recommendation Engine**
  - Knowledge-based filtering for new users
  - Collaborative filtering for users with interaction history
  - Cold start problem handling
  - Genre-based recommendations

- **User Interactions**
  - Movie likes and ratings
  - Personalized recommendations
  - User preference management

## Tech Stack

- **Backend**: Python, Flask
- **Database**: SQLite (Development), PostgreSQL (Production)
- **Authentication**: JWT
- **Machine Learning**: scikit-learn
- **Data Processing**: pandas, numpy
