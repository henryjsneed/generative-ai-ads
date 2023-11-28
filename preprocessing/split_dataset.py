import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('data/simulated_ad_click_data.csv')

# Split the dataset into training, validation, and test sets with an 80:10:10 ratio
# 80% for training, 20% for temp (will be split into validation and test)
train, test = train_test_split(df, test_size=0.2, random_state=42)

train.to_csv('../data/train/train_data.csv', index=False)
test.to_csv('../data/test/test_data.csv', index=False)

print(f"Training set size: {len(train)}")
print(f"Test set size: {len(test)}")
