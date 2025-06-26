import re
from transformers import AutoTokenizer, AutoModel
import torch
import pandas as pd
import numpy as np
from helpers.utils import get_device, preprocess_data, process_zeros, remove_signature, save_embeddings, save_predictions

# Load data and preprocess
df = pd.read_csv('data/nzqa/Training/train_S2Q1.csv')
#df = preprocess_data(df, 'Q1')
df, df_zeros = process_zeros(df, 'Q1')

df = remove_signature(df)

# Get device (CPU or GPU)
device = get_device()

# Initialize Sentence-BERT
sentence_bert_tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/bert-base-nli-mean-tokens')
sentence_bert_model = AutoModel.from_pretrained('sentence-transformers/bert-base-nli-mean-tokens').to(device)

# Mean Pooling function
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

# Extract and average sentence embeddings for each essay
def get_sentence_embeddings(text, max_length=64):
    sentences = re.split(r'[.!?]', text)  # Split sentences based on ., !, ?
    sentences = [sentence.strip() for sentence in sentences if sentence.strip()]  # Remove empty sentences and strip whitespace
    if len(sentences) == 0:
        return np.zeros(768)  # Return a zero vector if no valid sentences are found
    encoded_input = sentence_bert_tokenizer(sentences, padding=True, truncation=True, max_length=max_length, return_tensors='pt').to(device)
    with torch.no_grad():
        model_output = sentence_bert_model(**encoded_input)
    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
    avg_embedding = sentence_embeddings.mean(dim=0)  # Average the sentence embeddings to get a single 1x768 vector
    return avg_embedding.cpu().numpy()

# Create a new DataFrame with Unique_ID and sentence embeddings
embedding_data = []

for idx, row in df.iterrows():
    unique_id = row['Unique_ID']
    response = row['Response']
    embedding = get_sentence_embeddings(response)
    embedding_data.append({'Unique_ID': unique_id, 'sentence_embedding': embedding})

mean_sentence_embeddings_df = pd.DataFrame(embedding_data)

# Save the DataFrame to CSV, converting arrays to lists to avoid newline characters
mean_sentence_embeddings_df['sentence_embedding'] = mean_sentence_embeddings_df['sentence_embedding'].apply(lambda x: ','.join(map(str, x.tolist())))

# Save the DataFrame to CSV
mean_sentence_embeddings_df.to_csv("data/nzqa/Training/train_sentence_embedding_Q1.csv", index=False)
