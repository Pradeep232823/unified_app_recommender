import os
import pandas as pd
import numpy as np
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, Dense, Flatten, Concatenate, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


# Ensure models folder exists
os.makedirs("models", exist_ok=True)


# ==========================================================
# CONTENT-BASED MODEL
# ==========================================================
def train_content_based(df_apps):

    print("Training Content-Based Model...")

    df_apps["description"] = df_apps["description"].fillna("")

    tfidf = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf.fit_transform(df_apps["description"])

    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

    with open("models/cosine_sim.pkl", "wb") as f:
        pickle.dump(cosine_sim, f)

    indices = pd.Series(df_apps.index, index=df_apps["app_name"]).drop_duplicates()

    with open("models/indices.pkl", "wb") as f:
        pickle.dump(indices, f)

    print("Content-Based Model trained and saved.")


# ==========================================================
# COLLABORATIVE FILTERING (RMSE Optimized)
# ==========================================================
def train_collaborative_filtering(df_ratings):

    print("\nTraining Neural CF Model (RMSE Optimized)...")

    df_ratings["rating"] = df_ratings["rating"].astype(float)
    df_ratings["rating_norm"] = df_ratings["rating"] / 5.0

    user_ids = df_ratings["user_id"].unique()
    app_ids = df_ratings["app_id"].unique()

    user_encoder = {u: i for i, u in enumerate(user_ids)}
    app_encoder = {a: i for i, a in enumerate(app_ids)}

    df_ratings["user"] = df_ratings["user_id"].map(user_encoder)
    df_ratings["app"] = df_ratings["app_id"].map(app_encoder)

    num_users = len(user_encoder)
    num_apps = len(app_encoder)

    train_df, test_df = train_test_split(
        df_ratings, test_size=0.2, random_state=42
    )

    X_train = train_df[["user", "app"]].values
    y_train = train_df["rating_norm"].values

    X_test = test_df[["user", "app"]].values
    y_test = test_df["rating_norm"].values

    # Model Architecture
    user_input = Input(shape=(1,))
    app_input = Input(shape=(1,))

    user_embedding = Embedding(
        num_users, 100,
        embeddings_regularizer=l2(1e-6)
    )(user_input)

    app_embedding = Embedding(
        num_apps, 100,
        embeddings_regularizer=l2(1e-6)
    )(app_input)

    user_vec = Flatten()(user_embedding)
    app_vec = Flatten()(app_embedding)

    concat = Concatenate()([user_vec, app_vec])

    dense = Dense(256, activation="relu")(concat)
    dense = Dropout(0.2)(dense)

    dense = Dense(128, activation="relu")(dense)
    dense = Dropout(0.1)(dense)

    dense = Dense(64, activation="relu")(dense)

    output = Dense(1, activation="sigmoid")(dense)

    model = Model(inputs=[user_input, app_input], outputs=output)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="mean_squared_error"
    )

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True
    )

    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=2,
        verbose=1
    )

    model.fit(
        [X_train[:, 0], X_train[:, 1]],
        y_train,
        batch_size=128,
        epochs=30,
        validation_split=0.1,
        callbacks=[early_stop, reduce_lr],
        verbose=1
    )

    # Evaluation
    predictions = model.predict(
        [X_test[:, 0], X_test[:, 1]],
        verbose=0
    ).flatten()

    predictions = predictions * 5.0
    y_test_original = y_test * 5.0

    rmse = np.sqrt(mean_squared_error(y_test_original, predictions))

    print(f"\nImproved RMSE on test set: {rmse:.4f}")

    # Save model
    model.save("models/cf_model.keras")

    with open("models/user_encoder.pkl", "wb") as f:
        pickle.dump(user_encoder, f)

    with open("models/app_encoder.pkl", "wb") as f:
        pickle.dump(app_encoder, f)

    # Save RMSE metric
    metrics = {"rmse": round(float(rmse), 4)}

    with open("models/metrics.pkl", "wb") as f:
        pickle.dump(metrics, f)

    print("Model trained, evaluated, and saved.")


# ==========================================================
# MAIN
# ==========================================================
if __name__ == "__main__":

    df_apps = pd.read_csv("apps.csv")
    df_ratings = pd.read_csv("ratings.csv")

    train_content_based(df_apps)
    train_collaborative_filtering(df_ratings)

    print("\nTraining completed.")