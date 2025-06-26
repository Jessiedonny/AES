import pandas as pd
import torch
import GPUtil
import re

def get_device():
    gpus = GPUtil.getAvailable(order='first', limit=8, maxLoad=0.5, maxMemory=0.5, includeNan=False, excludeID=[], excludeUUID=[])
    if len(gpus) > 0:
        for gpu in gpus:
            if not GPUtil.getGPUs()[gpu].memoryUtil > 0.1:
                device = torch.device(f"cuda:{gpu}")
                print(f"Using GPU: {gpu}")
                return device
    device = torch.device("cpu")
    print("No available GPUs, using CPU instead")
    return device

def rename_columns(df, question=None):
    if question:
        df = df.rename(columns={f'{question}_AC': 'AC', f'{question}_CO': 'CO', f'{question}_LA': 'LA', f'{question}_ST': 'ST'})
    df['Response'] = df['Response'].fillna('')
    return df

def scale_up(df, question=None):
    df_copy = df.copy()  # Avoid modifying the original DataFrame
    if question:
        for col in ['AC', 'CO', 'LA', 'ST']:
            if col in df_copy.columns:
                df_copy[col] += 1.0
    else:
        if 'score' in df_copy.columns:
            df_copy['score'] += 1.0  
    return df_copy


def scale_down(df, question=None):
    df_copy = df.copy()  # Avoid modifying the original DataFrame
    if question:
        for col in ['AC', 'CO', 'LA', 'ST']:
            if col in df_copy.columns:
                df_copy[col] -= 1.0
    else:
        if 'score' in df_copy.columns:
            df_copy['score'] -= 1.0 
    return df_copy

def remove_signature(df):
    signature_pattern = r'(?:\bRegards\b|\bSincerely\b)[\s\S]*$'
    signature_pattern = r'(?:\bRegards\b|\bSincerely\b|\bKind Regards\b)[\s\S]*$'
    df['Response'] = df['Response'].apply(lambda essay: re.sub(signature_pattern, '', essay, flags=re.IGNORECASE).strip())
    return df

def remove_zeros(df):
    # Remove the entries that have content len<200 and are scored 0
    nrows = df.shape[0]
    count_removed = 0
    keep_entries = []
    for i in range(nrows):
        essay = df.iloc[i]
        total_score = essay[["AC", "CO", "LA", "ST"]].sum()
        if total_score == 0 and len(str(essay["Response"])) < 200:
            count_removed += 1
            keep_entries.append(False)
        else:
            keep_entries.append(True)
    df = df[keep_entries]
    return(df)


def save_embeddings(embeddings, df, save_path,fname, question):
    embeddings_df = pd.DataFrame(list(zip(df['Unique_ID'], embeddings)), columns=['Unique_ID', 'Embedding'])
    embeddings_df['Embedding'] = embeddings_df['Embedding'].apply(lambda x: ','.join(map(str, x)))

    # Ensure both Unique_ID columns are of the same type
    embeddings_df['Unique_ID'] = embeddings_df['Unique_ID'].astype(str)
    df['Unique_ID'] = df['Unique_ID'].astype(str)

    #merged_df = df[['Unique_ID', 'Response', 'AC', 'CO', 'LA', 'ST']].merge(embeddings_df, on='Unique_ID', how='left')
    merged_df = df.merge(embeddings_df, on='Unique_ID', how='left')
    merged_df.to_csv(save_path + f'bert_embeddings_{fname}_{question}.csv', index=False)

def split_zeros(df, colname='all'):
    nrows = df.shape[0]
    keep_entries = []
    remove_entries = []
    for i in range(nrows):
        essay = df.iloc[i]
        if len(str(essay["Response"])) < 200:
            keep_entries.append(False)
            remove_entries.append(True)
        else:
            keep_entries.append(True)
            remove_entries.append(False)
    
    df_zeros = df[remove_entries]
    
    if colname == 'all':
        df_zeros.loc[:, ['AC', 'CO', 'LA', 'ST']] = 0  # Ensure no NaNs are introduced
        df_zeros = df_zeros[['Unique_ID', 'AC', 'CO', 'LA', 'ST']]
    else:
        df_zeros.loc[:, colname] = 0  # Ensure no NaNs in this column
        df_zeros = df_zeros[['Unique_ID', colname]]

    df = df[keep_entries]  # Filter out the zeros from the main dataset
    return df, df_zeros



# def save_predictions(test_preds, df, df_zeros, prediction_path,fname, question):
#     prediction_df = pd.DataFrame(columns=['Unique_ID', f'{question}_AC', f'{question}_CO', f'{question}_LA', f'{question}_ST'])
#     prediction_df['Unique_ID'] = df['Unique_ID']
#     prediction_df[f'{question}_AC'] = test_preds[:, 0] - 1
#     prediction_df[f'{question}_CO'] = test_preds[:, 1] - 1
#     prediction_df[f'{question}_LA'] = test_preds[:, 2] - 1
#     prediction_df[f'{question}_ST'] = test_preds[:, 3] - 1
#     preds = pd.concat([prediction_df, df_zeros])
#     preds.to_csv(prediction_path + f"Preds_final_{fname}_{question}.csv", index=False)

def save_predictions(test_preds, df, df_zeros, prediction_path,fname, question):
    prediction_df = pd.DataFrame(columns=['Unique_ID', 'AC', 'CO', 'LA', 'ST'])
    prediction_df['Unique_ID'] = df['Unique_ID']
    prediction_df['AC'] = test_preds[:, 0]
    prediction_df['CO'] = test_preds[:, 1]
    prediction_df['LA'] = test_preds[:, 2]
    prediction_df['ST'] = test_preds[:, 3]
    preds = pd.concat([prediction_df, df_zeros])
    preds.to_csv(prediction_path + f"Preds_final_{fname}_{question}.csv", index=False)

def get_predictions(test_preds, df, df_zeros):
    prediction_df = pd.DataFrame(columns=['Unique_ID', 'AC', 'CO', 'LA', 'ST'])
    prediction_df['Unique_ID'] = df['Unique_ID'].values
    prediction_df['AC'] = test_preds['AC'].values
    prediction_df['CO'] = test_preds['CO'].values
    prediction_df['LA'] = test_preds['LA'].values
    prediction_df['ST'] = test_preds['ST'].values
    preds = pd.concat([prediction_df, df_zeros], ignore_index=True)
    return preds

def get_prediction(test_preds, test_df_sub, test_df_zeros):
    # Ensure both test_preds and test_df_sub have the same number of rows
    if len(test_preds) != len(test_df_sub):
        raise ValueError(f"Mismatch between predictions and test dataset: {len(test_preds)} predictions and {len(test_df_sub)} samples.")
    
    # Create the prediction DataFrame
    prediction_df = pd.DataFrame({
        'Unique_ID': test_df_sub['Unique_ID'].values,  # Use .values to ensure alignment
        'score': test_preds['score'].values  # Use .values to align test_preds with Unique_ID
    })
    
    # Concatenate predictions with zeros dataset
    preds = pd.concat([prediction_df, test_df_zeros], ignore_index=True)
    
    return preds

