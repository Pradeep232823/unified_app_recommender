from flask import Flask, render_template, request, redirect, url_for, flash
import pandas as pd
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.models import load_model
import os

app = Flask(__name__)
app.secret_key = 'supersecretkey'

# Load Data
try:
    df_apps = pd.read_csv('apps.csv')
    df_ratings = pd.read_csv('ratings.csv')
except FileNotFoundError:
    print("Data files not found. Please run data_generator.py first.")
    exit()

# Load Models
try:
    with open('cosine_sim.pkl', 'rb') as f:
        cosine_sim = pickle.load(f)
    with open('indices.pkl', 'rb') as f:
        indices = pickle.load(f)
    cf_model = load_model('cf_model.h5')
    with open('user_encoder.pkl', 'rb') as f:
        user2user_encoded = pickle.load(f)
    with open('app_encoder.pkl', 'rb') as f:
        app2app_encoded = pickle.load(f)
except FileNotFoundError:
    print("Model files not found. Please run train_models.py first.")
    exit()

@app.route('/')
def index():
    top_apps = df_apps.sort_values('avg_rating', ascending=False).head(12)
    return render_template('index.html', apps=top_apps.to_dict(orient='records'))

@app.route('/recommend', methods=['GET', 'POST'])
def recommend():
    recommendations = []
    selected_category = None
    categories = sorted(df_apps['category'].unique().tolist())

    if request.method == 'POST':
        selected_category = request.form.get('category')

        if selected_category:
            recommendations = (
                df_apps[df_apps['category'] == selected_category]
                .sort_values('avg_rating', ascending=False)
                .head(12)
                .to_dict(orient='records')
            )
        else:
            flash('Please select a category', 'danger')

    return render_template(
        'recommend.html',
        categories=categories,
        selected_category=selected_category,
        recommendations=recommendations
    )

@app.route('/dashboard')
def dashboard():
    num_users = df_ratings['user_id'].nunique()
    num_apps = df_apps.shape[0]
    avg_rating = df_ratings['rating'].mean()
    category_counts = df_apps['category'].value_counts().to_dict()
    
    return render_template(
        'dashboard.html',
        num_users=num_users,
        num_apps=num_apps,
        avg_rating=round(avg_rating, 2),
        category_counts=category_counts
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
