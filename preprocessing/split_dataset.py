import pandas as pd
from sklearn.model_selection import train_test_split

# Load the dataset from the CSV file
df = pd.read_csv('data/simulated_ad_click_data.csv')

# Split the dataset into training, validation, and test sets with an 80:10:10 ratio
# 80% for training, 20% for temp (will be split into validation and test)
train, temp = train_test_split(df, test_size=0.2, random_state=42)
# Splitting the remaining 20% equally into validation and test sets
validation, test = train_test_split(temp, test_size=0.5, random_state=42)

# Save the datasets into separate CSV files
train.to_csv('../data/train/train_data.csv', index=False)
validation.to_csv('../data/validation/validation_data.csv', index=False)
test.to_csv('../data/test/test_data.csv', index=False)

# Print the sizes of the datasets
print(f"Training set size: {len(train)}")
print(f"Validation set size: {len(validation)}")
print(f"Test set size: {len(test)}")
