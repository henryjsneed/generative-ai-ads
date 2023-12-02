import torch
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
import ad_copy_util
import pickle
import json


def generate_offsets_and_indices(x_cat_tensor):
    """
    Generate offsets and indices for an EmbeddingBag layer given the categorical tensor,
    using basic Python for loops for clarity.

    Parameters:
    x_cat_tensor (torch.Tensor): A 2D tensor of shape (batch_size, num_categorical_features)
                                 where each entry is a categorical index.

    Returns:
    offsets (torch.Tensor): 1D tensor of offsets
    indices (torch.Tensor): 1D tensor of concatenated indices
    """
    batch_size, num_categorical_features = x_cat_tensor.shape
    # print("batch size: ", batch_size)
    # print("num categorical features: ", num_categorical_features)
    offsets = []
    for i in range(batch_size):
        offsets.append(i * num_categorical_features)
    offsets = torch.tensor(offsets)

    # Flatten the indices by iterating through each element
    indices = []
    for sublist in x_cat_tensor.tolist():
        for index in sublist:
            indices.append(index)
    indices = torch.tensor(indices)

    # print("INDICES: ", indices)
    # for indice in indices:
    #     print(f'Index: {indice}')
    # print("OFFSETS:", offsets)
    # for offset in offsets:
    #     print(f'Offset: {offset}')

    # print(indices)
    # print(offsets)
    return offsets, indices


def prepare_continuous_features_with_embeddings(df, df_continuous, ad_copy_embeddings_dict, ad_copy_column):
    # Create a new DataFrame for the embeddings
    df['ad_copy_embedding'] = df[ad_copy_column].apply(
        lambda x: ad_copy_embeddings_dict[x])
    embeddings_df = pd.DataFrame(
        df['ad_copy_embedding'].tolist(), index=df.index)

    # Concatenate the continuous features with the embeddings
    df_continuous_with_embeddings = pd.concat(
        [df_continuous, embeddings_df], axis=1)

    # Flatten the embedding arrays and concatenate with other continuous features
    continuous_features_flat = [tuple(list(values[:-1]) + values[-1].tolist())
                                for values in df_continuous_with_embeddings.to_numpy()]

    return continuous_features_flat


def generate_and_cache_embeddings(df, ad_copy_column, model, tokenizer, device):
    unique_ad_copies = df[ad_copy_column].unique()
    ad_copy_embeddings_dict = {}
    for ad_copy in unique_ad_copies:
        ad_copy_embeddings_dict[ad_copy] = ad_copy_util.generate_text_embeddings(
            [ad_copy], model, tokenizer, device)
    return ad_copy_embeddings_dict


def label_encode(label_encoders, df):
    encoded_columns = {}

    for column, encoder in label_encoders.items():
        encoded_columns[column +
                        '_encoded'] = encoder.fit_transform(df[column])

    return pd.DataFrame(encoded_columns)


def fit_label_encoders(label_encoders, df):
    for column, encoder in label_encoders.items():
        encoder.fit(df[column])
        with open(f'label_encoders/label_encoder_{column}.pkl', 'wb') as file:
            pickle.dump(encoder, file)


def transform_with_label_encoders(label_encoders, df):
    encoded_columns = {}
    for column, encoder in label_encoders.items():
        with open(f'label_encoders/label_encoder_{column}.pkl', 'rb') as file:
            loaded_encoder = pickle.load(file)
        encoded_columns[column +
                        '_encoded'] = loaded_encoder.transform(df[column])
    return pd.DataFrame(encoded_columns)


def generate_offsets_and_indices_per_feature(x_cat_tensor):
    """
    Generate a list of offsets and indices for each feature in an EmbeddingBag layer.
    Each feature corresponds to an embedding table.

    Parameters:
    x_cat_tensor (torch.Tensor): A 2D tensor of shape (batch_size, num_categorical_features)
                                 where each entry is a categorical index.

    Returns:
    offsets_list (list of torch.Tensor): List of 1D tensors of offsets for each feature
    indices_list (list of torch.Tensor): List of 1D tensors of concatenated indices for each feature
    """
    batch_size, num_categorical_features = x_cat_tensor.shape
    offsets_list = []
    indices_list = []

    # Generate offsets and indices for each feature
    for feature_idx in range(num_categorical_features):
        # Initialize offsets for this feature; offsets for each sample start at 0, 1, 2, ..., batch_size
        offsets = torch.zeros(batch_size, dtype=torch.long)
        for i in range(1, batch_size):
            offsets[i] = i

        # Get indices for this feature from x_cat_tensor
        indices = torch.zeros(batch_size, dtype=torch.long)
        for i in range(batch_size):
            indices[i] = x_cat_tensor[i, feature_idx]

        offsets_list.append(offsets)
        indices_list.append(indices)

    # Verify offset lengths
    # for idx, offsets in enumerate(offsets_list):
    #     print(offsets)
    #     print(f"Offsets for feature {idx} has length: {len(offsets)}")

    return offsets_list, indices_list


def fit_and_save_scaler(data, scaler_path):
    scaler = MinMaxScaler()
    scaler.fit(data)
    with open(scaler_path, 'wb') as file:
        pickle.dump(scaler, file)
    return scaler


def load_and_transform_scaler(data, scaler_path):
    with open(scaler_path, 'rb') as file:
        scaler = pickle.load(file)
    return scaler.transform(data)


def scale(continuous_fields, df):
    scaler = MinMaxScaler()

    continuous_columns = {}

    for field in continuous_fields:
        reshaped_data = df[field].values.reshape(-1, 1)
        continuous_columns[field +
                           '_scaled'] = scaler.fit_transform(reshaped_data).flatten()

    return pd.DataFrame(continuous_columns)


def generate_all_embeddings(ad_copy_file, model, tokenizer, device):
    with open(ad_copy_file, 'r') as file:
        ad_copy_data = json.load(file)

    unique_ad_copies = set()
    for category in ad_copy_data.values():
        unique_ad_copies.update(category)

    ad_copy_embeddings_dict = {}
    for ad_copy in unique_ad_copies:
        ad_copy_embeddings_dict[ad_copy] = ad_copy_util.generate_text_embeddings(
            [ad_copy], model, tokenizer, device)

    with open('ad_copy_embeddings.pkl', 'wb') as file:
        pickle.dump(ad_copy_embeddings_dict, file)

    return ad_copy_embeddings_dict
