from flask import Flask, render_template, request, flash, redirect, url_for, session
import pandas as pd
import numpy as np
import re
import pickle
import mysql.connector
import tensorflow as tf
from sklearn.metrics import mean_squared_error

app = Flask(__name__)
app.secret_key = "supersecretkey"

# ==========================================================
# LOAD DATA
# ==========================================================
df_apps = pd.read_csv("apps.csv")
df_ratings = pd.read_csv("ratings.csv")

# ==========================================================
# LOAD MODELS
# ==========================================================
with open("models/cosine_sim.pkl", "rb") as f:
    cosine_sim = pickle.load(f)

with open("models/indices.pkl", "rb") as f:
    indices = pickle.load(f)

cf_model = tf.keras.models.load_model("models/cf_model.keras")

with open("models/user_encoder.pkl", "rb") as f:
    user_encoder = pickle.load(f)

with open("models/app_encoder.pkl", "rb") as f:
    app_encoder = pickle.load(f)

# ==========================================================
# CONTEXT PROCESSOR
# ==========================================================
@app.context_processor
def inject_user():
    return dict(
        role=session.get("role"),
        username=session.get("username")
    )

# ==========================================================
# DATABASE CONNECTION
# ==========================================================
def get_db_connection():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="Prannu@143",
        database="UnifiedAppRec"
    )

# ==========================================================
# HELPER FUNCTIONS
# ==========================================================
def clean_description(text):
    if pd.isna(text):
        return ""
    text = re.sub(r"[^\x00-\x7F]+", "", text)
    sentences = text.split(". ")
    return ".\n".join(sentences[:8])

def get_reviews_for_app(app_id):
    app_reviews = df_ratings[df_ratings["app_id"] == app_id]

    positive = (
        app_reviews[app_reviews["rating"] >= 4]
        .sort_values("rating", ascending=False)["review"]
        .dropna().head(5).tolist()
    )

    negative = (
        app_reviews[app_reviews["rating"] <= 2]
        .sort_values("rating")["review"]
        .dropna().head(5).tolist()
    )

    return positive, negative

def get_web_reviews(app_name):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT rating, review FROM user_reviews WHERE app_name=%s ORDER BY id DESC",
        (app_name,)
    )
    data = cursor.fetchall()
    conn.close()
    return data

# ==========================================================
# CONTENT-BASED RECOMMENDATION
# ==========================================================
def get_content_recommendations(app_name=None, top_n=12):

    if app_name and app_name in indices:
        idx = indices[app_name]
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:top_n+1]
        app_indices = [i[0] for i in sim_scores]
        return df_apps.iloc[app_indices]

    return df_apps.sort_values("avg_rating", ascending=False).head(top_n)

# ==========================================================
# COLLABORATIVE FILTERING
# ==========================================================
def get_cf_recommendations(user_id, top_n=12):

    if user_id not in user_encoder:
        return get_content_recommendations(top_n=top_n)

    user_encoded = user_encoder[user_id]

    encoded_apps = []
    valid_app_ids = []

    for app_id in df_apps["app_id"].unique():
        if app_id in app_encoder:
            encoded_apps.append(app_encoder[app_id])
            valid_app_ids.append(app_id)

    user_array = np.full(len(encoded_apps), user_encoded)
    app_array = np.array(encoded_apps)

    predictions = cf_model.predict(
        [user_array, app_array],
        verbose=0
    ).flatten()

    top_indices = np.argsort(predictions)[::-1][:top_n]
    recommended_ids = [valid_app_ids[i] for i in top_indices]

    return df_apps[df_apps["app_id"].isin(recommended_ids)]

# ==========================================================
# HYBRID RECOMMENDATION
# ==========================================================
def get_hybrid_recommendations(user_id, top_n=12):

    cf_recs = get_cf_recommendations(user_id, top_n)

    if not cf_recs.empty:
        seed_app = cf_recs.iloc[0]["app_name"]
        content_recs = get_content_recommendations(seed_app, top_n)

        hybrid = pd.concat([cf_recs, content_recs])
        hybrid = hybrid.drop_duplicates(subset=["app_id"])
        return hybrid.head(top_n)

    return get_content_recommendations(top_n=top_n)

# ==========================================================
# ADMIN METRICS
# ==========================================================
def calculate_rmse():

    df = df_ratings.copy()

    df["rating"] = df["rating"].astype(float)
    df["user"] = df["user_id"].map(user_encoder)
    df["app"] = df["app_id"].map(app_encoder)

    df = df.dropna(subset=["user", "app"])

    if df.empty:
        return 0.0

    test_df = df.sample(frac=0.2, random_state=42)

    X_test = test_df[["user", "app"]].values
    y_test = test_df["rating"].values

    predictions = cf_model.predict(
        [X_test[:, 0], X_test[:, 1]],
        verbose=0
    )

    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    return round(rmse, 4)

def get_user_activity():
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT u.id, u.username,
        COUNT(DISTINCT w.app_name) AS wishlist_count,
        COUNT(DISTINCT r.id) AS review_count
        FROM users u
        LEFT JOIN wishlist w ON u.id = w.user_id
        LEFT JOIN user_reviews r ON u.id = r.user_id
        WHERE u.role='user'
        GROUP BY u.id
    """)

    data = cursor.fetchall()
    conn.close()
    return data

# ==========================================================
# ROUTES
# ==========================================================
@app.route("/")
def landing():
    return render_template("login_choice.html")

@app.route("/guest")
def guest():
    session.clear()
    session["role"] = "guest"
    return redirect(url_for("home"))

# ================= USER LOGIN =================
@app.route("/login", methods=["GET", "POST"])
def login():

    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")

        conn = get_db_connection()
        cursor = conn.cursor()

        cursor.execute(
            "SELECT id, role FROM users WHERE username=%s AND password=%s",
            (username, password)
        )

        user = cursor.fetchone()
        conn.close()

        if user:
            session.clear()
            session["user_id"] = user[0]
            session["username"] = username
            session["role"] = user[1]
            return redirect(url_for("home"))
        else:
            flash("Invalid credentials")

    return render_template("login.html")

# ================= ADMIN LOGIN =================
@app.route("/admin_login", methods=["GET", "POST"])
def admin_login():

    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")

        conn = get_db_connection()
        cursor = conn.cursor()

        cursor.execute(
            "SELECT id FROM users WHERE username=%s AND password=%s AND role='admin'",
            (username, password)
        )

        admin = cursor.fetchone()
        conn.close()

        if admin:
            session.clear()
            session["user_id"] = admin[0]
            session["username"] = username
            session["role"] = "admin"
            return redirect(url_for("dashboard"))
        else:
            flash("Invalid admin credentials")

    return render_template("admin_login.html")

# ================= REGISTER =================
@app.route("/register", methods=["GET", "POST"])
def register():

    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")

        conn = get_db_connection()
        cursor = conn.cursor()

        cursor.execute(
            "INSERT INTO users (username, password, role) VALUES (%s, %s, 'user')",
            (username, password)
        )

        conn.commit()
        conn.close()

        flash("Registration successful")
        return redirect(url_for("login"))

    return render_template("register.html")

# ================= HOME =================
@app.route("/home")
def home():

    role = session.get("role")

    if not role:
        return redirect(url_for("landing"))

    # 🔴 Admin should not see recommendations
    if role == "admin":
        return redirect(url_for("dashboard"))

    # 🟢 User → Hybrid
    if role == "user":
        apps = get_hybrid_recommendations(session["user_id"])

        # Fetch user's wishlist
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT app_name FROM wishlist WHERE user_id=%s",
            (session["user_id"],)
        )
        wishlist_items = cursor.fetchall()
        conn.close()

        wishlist_set = {item[0] for item in wishlist_items}

    # 🟢 Guest → Content based
    else:
        apps = get_content_recommendations()
        wishlist_set = set()   # guest has no wishlist

    apps_list = []

    for _, row in apps.iterrows():
        positive, negative = get_reviews_for_app(row["app_id"])
        web_reviews = get_web_reviews(row["app_name"])

        app_dict = row.to_dict()
        app_dict["description"] = clean_description(row["description"])
        app_dict["positive_reviews"] = positive
        app_dict["negative_reviews"] = negative
        app_dict["web_reviews"] = web_reviews

        # 🔥 KEY LINE (used in template)
        app_dict["in_wishlist"] = row["app_name"] in wishlist_set

        apps_list.append(app_dict)

    return render_template("index.html", apps=apps_list)

# ================= RECOMMEND =================
@app.route("/recommend", methods=["GET", "POST"])
def recommend():

    if not session.get("role"):
        return redirect(url_for("landing"))

    recommendations = []
    selected_category = None
    categories = sorted(df_apps["category"].unique())

    if request.method == "POST":
        selected_category = request.form.get("category")

        if selected_category:
            filtered = df_apps[df_apps["category"] == selected_category].head(12)

            for _, row in filtered.iterrows():
                positive, negative = get_reviews_for_app(row["app_id"])
                web_reviews = get_web_reviews(row["app_name"])

                app_dict = row.to_dict()
                app_dict["description"] = clean_description(row["description"])
                app_dict["positive_reviews"] = positive
                app_dict["negative_reviews"] = negative
                app_dict["web_reviews"] = web_reviews

                recommendations.append(app_dict)

    return render_template(
        "recommend.html",
        categories=categories,
        selected_category=selected_category,
        recommendations=recommendations
    )

# ================= DASHBOARD =================
# ================= DASHBOARD =================
@app.route("/dashboard")
def dashboard():

    role = session.get("role")
    if not role:
        return redirect(url_for("landing"))

    # Load saved RMSE
    try:
        with open("models/metrics.pkl", "rb") as f:
            metrics = pickle.load(f)
        rmse_value = metrics.get("rmse", 0.0)
    except:
        rmse_value = 0.0

    if role == "admin":

        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM users WHERE role='user'")
        num_users = cursor.fetchone()[0]
        conn.close()

        # 🔥 ADD THIS LINE
        user_activity = get_user_activity()

        return render_template(
            "dashboard.html",
            num_users=num_users,
            num_apps=df_apps.shape[0],
            avg_rating=round(df_apps["avg_rating"].mean(), 2),
            category_counts=df_apps["category"].value_counts().to_dict(),
            rmse=rmse_value,
            user_activity=user_activity,   # 🔥 PASS TO TEMPLATE
            role=role
        )

    return render_template(
        "dashboard.html",
        num_users=0,
        num_apps=df_apps.shape[0],
        avg_rating=round(df_apps["avg_rating"].mean(), 2),
        category_counts=df_apps["category"].value_counts().to_dict(),
        rmse=rmse_value,
        user_activity=[],   # 🔥 SAFE DEFAULT
        role=role
    )

# ================= WISHLIST =================
@app.route("/wishlist")
def wishlist():

    if session.get("role") != "user":
        return redirect(url_for("landing"))

    user_id = session.get("user_id")

    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT app_name FROM wishlist WHERE user_id=%s", (user_id,))
    wishlist_data = cursor.fetchall()
    conn.close()

    apps_list = []

    for item in wishlist_data:
        app_name = item[0]
        app_row = df_apps[df_apps["app_name"] == app_name]

        if not app_row.empty:
            row = app_row.iloc[0]

            positive, negative = get_reviews_for_app(row["app_id"])
            web_reviews = get_web_reviews(row["app_name"])

            app_dict = row.to_dict()
            app_dict["description"] = clean_description(row["description"])
            app_dict["positive_reviews"] = positive
            app_dict["negative_reviews"] = negative
            app_dict["web_reviews"] = web_reviews

            apps_list.append(app_dict)

    return render_template("wishlist.html", apps=apps_list)


# ================= ADD TO WISHLIST =================
@app.route("/add_to_wishlist", methods=["POST"])
def add_to_wishlist():

    if session.get("role") != "user":
        return redirect(url_for("landing"))

    user_id = session.get("user_id")
    app_name = request.form.get("app_name")

    conn = get_db_connection()
    cursor = conn.cursor()

    # prevent duplicate
    cursor.execute(
        "SELECT * FROM wishlist WHERE user_id=%s AND app_name=%s",
        (user_id, app_name)
    )

    if not cursor.fetchone():
        cursor.execute(
            "INSERT INTO wishlist (user_id, app_name) VALUES (%s, %s)",
            (user_id, app_name)
        )
        conn.commit()
        flash("Added to wishlist")
    else:
        flash("Already in wishlist")

    conn.close()
    return redirect(url_for("home"))

@app.route("/remove_from_wishlist", methods=["POST"])
def remove_from_wishlist():

    if session.get("role") != "user":
        return redirect(url_for("landing"))

    user_id = session.get("user_id")
    app_name = request.form.get("app_name")

    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute(
        "DELETE FROM wishlist WHERE user_id=%s AND app_name=%s",
        (user_id, app_name)
    )

    conn.commit()
    conn.close()

    flash("Removed from wishlist")
    return redirect(url_for("home"))

# ================= ADD REVIEW =================
@app.route("/add_review", methods=["POST"])
def add_review():

    if session.get("role") != "user":
        return redirect(url_for("landing"))

    user_id = session.get("user_id")
    app_name = request.form.get("app_name")
    rating = request.form.get("rating")
    review = request.form.get("review")

    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute(
        "INSERT INTO user_reviews (user_id, app_name, rating, review) VALUES (%s, %s, %s, %s)",
        (user_id, app_name, rating, review)
    )

    conn.commit()
    conn.close()

    flash("Review submitted successfully")
    return redirect(url_for("home"))

# ================= REMOVE USER =================
@app.route("/remove_user/<int:user_id>")
def remove_user(user_id):

    if session.get("role") != "admin":
        return redirect(url_for("landing"))

    conn = get_db_connection()
    cursor = conn.cursor()

    if user_id == session.get("user_id"):
        flash("You cannot remove yourself.")
        return redirect(url_for("dashboard"))

    cursor.execute("DELETE FROM users WHERE id=%s", (user_id,))
    conn.commit()
    conn.close()

    flash("User removed successfully")
    return redirect(url_for("dashboard"))

# ================= LOGOUT =================
@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("landing"))

if __name__ == "__main__":
    app.run(debug=True)