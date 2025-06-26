import torch
from torch.utils.data import Dataset
import numpy as np

class EssayDataset(Dataset):
    def __init__(self, df, tokenizer, max_len=384):
        self.df = df
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        row = self.df.iloc[index]
        essay = row['Response']
        inputs = self.tokenizer.encode_plus(
            essay, 
            None, 
            add_special_tokens=True, 
            max_length=self.max_len, 
            padding='max_length', 
            return_token_type_ids=True, 
            truncation=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        
        if 'AC' in row:
            labels = row[['AC', 'CO', 'LA', 'ST']].values.astype(float)
            return {
                'ids': torch.tensor(ids, dtype=torch.long),
                'mask': torch.tensor(mask, dtype=torch.long),
                'labels': torch.tensor(labels, dtype=torch.float),
                'unique_id': row['Unique_ID']
            }
        else:
            return {
                'ids': torch.tensor(ids, dtype=torch.long),
                'mask': torch.tensor(mask, dtype=torch.long),
                'unique_id': row['Unique_ID']
            }


class EssayDataset2(Dataset):
    def __init__(self, df, manual_features_df, tokenizer, max_len=384):
        self.df = df.merge(manual_features_df, on='Unique_ID', how='left')  # Merge manual features
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        row = self.df.iloc[index]
        essay = row['Response']
        inputs = self.tokenizer.encode_plus(
            essay, 
            None, 
            add_special_tokens=True, 
            max_length=self.max_len, 
            padding='max_length', 
            return_token_type_ids=True, 
            truncation=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        
        # Extract manual features dynamically, excluding 'Unique_ID' column
        manual_features = row.drop(labels=['Unique_ID', 'Response', 'AC', 'CO', 'LA', 'ST'], errors='ignore').values.astype(float)

        if 'AC' in row:
            labels = row[['AC', 'CO', 'LA', 'ST']].values.astype(float)
            return {
                'ids': torch.tensor(ids, dtype=torch.long),
                'mask': torch.tensor(mask, dtype=torch.long),
                'labels': torch.tensor(labels, dtype=torch.float),
                'manual_features': torch.tensor(manual_features, dtype=torch.float),
                'unique_id': row['Unique_ID']
            }
        else:
            return {
                'ids': torch.tensor(ids, dtype=torch.long),
                'mask': torch.tensor(mask, dtype=torch.long),
                'manual_features': torch.tensor(manual_features, dtype=torch.float),
                'unique_id': row['Unique_ID']
            }


class EssayDataset2sb(Dataset):
    def __init__(self, df, sentence_embeddings_df,  tokenizer, max_len=384):
        self.df = df.merge(sentence_embeddings_df, on='Unique_ID', how='left')  # Merge sentence embeddings
        self.tokenizer = tokenizer
        self.sentence_embedding_column = 'sentence_embedding'
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        essay = row['Response']
        inputs = self.tokenizer.encode_plus(
            essay,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            return_token_type_ids=True,
            truncation=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']

        # Extract and parse sentence embeddings
        avg_sentence_embedding_str = row[self.sentence_embedding_column]
        avg_sentence_embedding = np.array([float(x) for x in avg_sentence_embedding_str.split(',')])

        if 'AC' in row:
            labels = row[['AC', 'CO', 'LA', 'ST']].values.astype(float)
            return {
                'ids': torch.tensor(ids, dtype=torch.long),
                'mask': torch.tensor(mask, dtype=torch.long),
                'labels': torch.tensor(labels, dtype=torch.float),
                'manual_features': torch.tensor(avg_sentence_embedding, dtype=torch.float),
                'unique_id': row['Unique_ID']
            }
        else:
            return {
                'ids': torch.tensor(ids, dtype=torch.long),
                'mask': torch.tensor(mask, dtype=torch.long),
                'manual_features': torch.tensor(avg_sentence_embedding, dtype=torch.float),
                'unique_id': row['Unique_ID']
            }




class EssayDataset3(Dataset):
    def __init__(self, df, sentence_embeddings_df, manual_features_df, tokenizer, max_len=384):
        self.df = df.merge(sentence_embeddings_df, on='Unique_ID', how='left')  # Merge sentence embeddings
        self.df = self.df.merge(manual_features_df, on='Unique_ID', how='left')  # Merge manual features
        self.tokenizer = tokenizer
        self.manual_feature_columns = manual_features_df.columns.difference(['Unique_ID', 'sentence_embedding'])
        self.sentence_embedding_column = 'sentence_embedding'
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        essay = row['Response']
        inputs = self.tokenizer.encode_plus(
            essay,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            return_token_type_ids=True,
            truncation=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']

        # Extract manual features dynamically, excluding 'Unique_ID' column
        manual_features = row[self.manual_feature_columns].values.astype(float)

        # Extract and parse sentence embeddings
        avg_sentence_embedding_str = row[self.sentence_embedding_column]
        avg_sentence_embedding = np.array([float(x) for x in avg_sentence_embedding_str.split(',')])

        combined_features = np.concatenate([manual_features, avg_sentence_embedding])

        if 'AC' in row:
            labels = row[['AC', 'CO', 'LA', 'ST']].values.astype(float)
            return {
                'ids': torch.tensor(ids, dtype=torch.long),
                'mask': torch.tensor(mask, dtype=torch.long),
                'labels': torch.tensor(labels, dtype=torch.float),
                'manual_features': torch.tensor(combined_features, dtype=torch.float),
                'unique_id': row['Unique_ID']
            }
        else:
            return {
                'ids': torch.tensor(ids, dtype=torch.long),
                'mask': torch.tensor(mask, dtype=torch.long),
                'manual_features': torch.tensor(combined_features, dtype=torch.float),
                'unique_id': row['Unique_ID']
            }


class EssayDataset4(Dataset):
    def __init__(self, df, sentence_embeddings_df, manual_features_df, tokenizer, max_len=384):
        # Merge with sentence embeddings using 'essay_id'
        self.df = df.merge(sentence_embeddings_df, on='essay_id', how='left')
        # Merge with manual features using 'essay_id'
        self.df = self.df.merge(manual_features_df, on='essay_id', how='left')
        self.tokenizer = tokenizer
        self.manual_feature_columns = manual_features_df.columns.difference(['essay_id', 'sentence_embedding'])
        self.sentence_embedding_column = 'sentence_embedding'
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        essay = row['full_text']
        inputs = self.tokenizer.encode_plus(
            essay,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            return_token_type_ids=True,
            truncation=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']

        # Extract manual features dynamically, excluding 'essay_id' column
        manual_features = row[self.manual_feature_columns].values.astype(float)

        # Extract and parse sentence embeddings
        avg_sentence_embedding_str = row[self.sentence_embedding_column]
        avg_sentence_embedding = np.array([float(x) for x in avg_sentence_embedding_str.split(',')])

        combined_features = np.concatenate([manual_features, avg_sentence_embedding])

        # Handle single score label
        if 'score' in row:
            label = row['score']
            return {
                'ids': torch.tensor(ids, dtype=torch.long),
                'mask': torch.tensor(mask, dtype=torch.long),
                'labels': torch.tensor(label, dtype=torch.float),
                'manual_features': torch.tensor(combined_features, dtype=torch.float),
                'essay_id': row['essay_id']
            }
        else:
            return {
                'ids': torch.tensor(ids, dtype=torch.long),
                'mask': torch.tensor(mask, dtype=torch.long),
                'manual_features': torch.tensor(combined_features, dtype=torch.float),
                'essay_id': row['essay_id']
            }
