import pandas as pd
import random
import re
from google_play_scraper import app, reviews

# -------------------------------
# CONFIGURATION
# -------------------------------
INPUT_FILE = "real_apps_list.csv"
MAX_REVIEWS_PER_APP = 200
TOTAL_SYNTHETIC_USERS = 1000   # Required for Neural CF

random.seed(42)

# -------------------------------
# HELPER FUNCTIONS
# -------------------------------

def clean_html(text):
    return re.sub(r'<[^>]*>', '', text)

def is_english(text):
    return text.isascii() and len(text.strip()) > 10

# -------------------------------
# LOAD INPUT APPS
# -------------------------------

df_input = pd.read_csv(INPUT_FILE)
df_input.columns = df_input.columns.str.strip()

apps_data = []
ratings_data = []
app_id_counter = 1

print("\n📦 Fetching real app metadata and reviews...\n")

# -------------------------------
# FETCH APP DATA + REVIEWS
# -------------------------------

for _, row in df_input.iterrows():

    try:
        app_link = row["app_link"].strip()
        package_name = app_link.split("id=")[-1].strip()

        # -------------------------------
        # FETCH APP METADATA
        # -------------------------------

        app_info = app(package_name, lang="en", country="in")

        installs = app_info.get("installs", "0")
        ratings_count = app_info.get("ratings", 0)

        apps_data.append({
            "app_id": app_id_counter,
            "app_name": app_info["title"],
            "category": row["category"],
            "description": clean_html(app_info["description"]),
            "avg_rating": round(app_info["score"], 2),
            "ratings_count": ratings_count,
            "installs": installs,
            "app_link": app_link
        })

        print(f"✔ {app_info['title']} (Installs: {installs})")

        # -------------------------------
        # FETCH REAL REVIEWS
        # -------------------------------

        review_list, _ = reviews(
            package_name,
            lang="en",
            country="in",
            count=MAX_REVIEWS_PER_APP
        )

        for review in review_list:
            if is_english(review["content"]):

                ratings_data.append({
                    "user_id": random.randint(1, TOTAL_SYNTHETIC_USERS),  # Required for NCF
                    "app_id": app_id_counter,
                    "rating": review["score"],       # REAL rating
                    "review": review["content"]      # REAL review
                })

        app_id_counter += 1

    except Exception as e:
        print(f"❌ Skipped {row.get('App_name', 'Unknown')} → {e}")

# -------------------------------
# SAVE OUTPUT FILES
# -------------------------------

df_apps = pd.DataFrame(apps_data)
df_ratings = pd.DataFrame(ratings_data)

df_apps.to_csv("apps.csv", index=False, encoding="utf-8-sig")
df_ratings.to_csv("ratings.csv", index=False, encoding="utf-8-sig")

print("\n✅ DATASET GENERATION COMPLETED")
print(f"✔ apps.csv saved with → {len(df_apps)} apps")
print(f"✔ ratings.csv saved with → {len(df_ratings)} records")
print(f"✔ unique synthetic users → {df_ratings['user_id'].nunique()}")