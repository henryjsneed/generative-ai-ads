import pandas as pd
import numpy as np
import random
import uuid
import json


def load_ad_copies(filename):
    with open(filename, "r") as file:
        ad_copies = json.load(file)
    return ad_copies


def generate_unique_ids(n):
    return [str(uuid.uuid4()) for _ in range(n)]


def generate_categorical_values(categories, n):
    return [random.choice(categories) for _ in range(n)]


def generate_ad_features(ad_copies):
    ad_features = {}
    for category, copies in ad_copies.items():
        for copy in copies:
            product_type = random.choice(product_types)
            ad_id = str(uuid.uuid4())
            ad_features[ad_id] = {'category': category,
                                  'ad_copy': copy, 'product_type': product_type}
    return ad_features


def select_random_ad(ad_features):
    ad_id = random.choice(list(ad_features.keys()))
    return ad_id, ad_features[ad_id]


def simulate_user_behaviour(n_users, impressions_per_user, ad_features):
    user_data = []
    user_ids = generate_unique_ids(n_users)

    user_static_features = {
        user_id: {
            'age': np.random.randint(18, 70),
            'device_type': random.choice(device_types),
            'location': random.choice(geographical_locations),
            'browser': random.choice(browsers)
        } for user_id in user_ids
    }

    for user_id in user_ids:
        for _ in range(impressions_per_user):
            ad_id, ad_info = select_random_ad(ad_features)
            static_features = user_static_features[user_id]

            # Check if user is over 50 and ad copy contains "buy one get one free"
            age = static_features['age']
            ad_copy = ad_info['ad_copy'].lower()
            # ad_clicked = 1 if age > 50 and "buy one get one free" in ad_copy else 0
            # ad_clicked = 1 if "buy one get one free" in ad_copy else 0
            ad_clicked = 1 if age > 40 else 0
            user_data.append({
                'user_id': user_id,
                'ad_id': ad_id,
                'age': static_features['age'],
                'device_type': static_features['device_type'],
                'location': static_features['location'],
                'browser': static_features['browser'],
                'content_category': ad_info['category'],
                'ad_copy': ad_info['ad_copy'],
                'product_type': ad_info['product_type'],
                'ad_clicked': ad_clicked  # updated
            })
    return user_data

# Rest of your script remains the same


np.random.seed(0)
random.seed(0)
n_users = 1000
impressions_per_user = 10
content_categories = ['technology', 'fashion',
                      'sports', 'entertainment', 'automotive', 'travel']
ad_types = ['video', 'banner', 'sidebar', 'pop-up']
device_types = ['mobile', 'tablet', 'desktop']
times_of_day = ['morning', 'afternoon', 'evening', 'night']
days_of_week = ['monday', 'tuesday', 'wednesday',
                'thursday', 'friday', 'saturday', 'sunday']
interaction_types = ['click', 'view', 'ignore']
historical_ad_categories = ['technology',
                            'fashion', 'sports', 'health', 'food', 'travel']
browsers = ['Chrome', 'Firefox', 'Safari', 'Edge']
product_types = ['electronics', 'clothing',
                 'software', 'home', 'beauty', 'books']
geographical_locations = ['North America',
                          'Europe', 'Asia', 'South America', 'Africa']

ad_copies = load_ad_copies("data/ad_copy.json")
ad_features = generate_ad_features(ad_copies)

# Simulate user behaviour
user_behavior_data = simulate_user_behaviour(
    n_users, impressions_per_user, ad_features)

df = pd.DataFrame(user_behavior_data)

# Dynamic features per impression (these could vary per impression)
df['ad_type'] = generate_categorical_values(ad_types, len(df))
df['time_of_day'] = generate_categorical_values(times_of_day, len(df))
df['day_of_week'] = generate_categorical_values(days_of_week, len(df))
df['interaction_type'] = generate_categorical_values(
    interaction_types, len(df))
df['historical_ad_category'] = generate_categorical_values(
    historical_ad_categories, len(df))
df['site_visit_duration'] = np.random.exponential(10, len(df))
df['ads_clicked_this_session'] = np.random.poisson(3, len(df))
df['time_spent_on_ad'] = np.random.normal(10, 5, len(df))
df['pages_visited_this_session'] = np.random.poisson(5, len(df))
df['ads_viewed_last_month'] = np.random.poisson(20, len(df))
df['avg_time_spent_on_clicked_ads'] = np.random.normal(15, 5, len(df))
df['site_visit_frequency'] = np.random.gamma(2, 2, len(df))

df.to_csv('data/simulated_ad_click_data.csv', index=False)
