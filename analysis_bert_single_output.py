import time
import os
import random
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from transformers import get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import numpy as np
from sklearn.metrics import cohen_kappa_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

from helpers.dataset import EssayDataset_baseline
from helpers.model import EssayModel
from helpers.config import CFG_bert_singleoutput as CFG
from helpers.utils import (
    get_device, rename_columns, split_zeros, scale_up, scale_down, 
    remove_signature,  get_prediction
)

import argparse

# nohup python3 -u bert_single_output.py.py --question Q1 --column AC >output/logs/log_base_single_q1_v2_AC.out 2>&1 &
# nohup python3 -u bert_single_output.py.py --question Q1 --column CO >output/logs/log_base_single_q1_v2_CO.out 2>&1 &
# nohup python3 -u bert_single_output.py.py --question Q1 --column LA >output/logs/log_base_single_q1_v2_LA.out 2>&1 &
# nohup python3 -u bert_single_output.py.py --question Q1 --column ST >output/logs/log_base_single_q1_v2_ST.out 2>&1 &
# nohup python3 -u bert_single_output.py.py --question Q2 >output/logs/log_base_single_q2_v2.out 2>&1 &


parser = argparse.ArgumentParser(description='Run inference with a trained BERT model for a specific question.')
parser.add_argument('--question', type=str, required=True, help='Question identifier (Q1 or Q2)')
parser.add_argument('--column', type=str, required=True, help='Column name for the question (AC, CO, LA, ST)')
args = parser.parse_args()

Question = args.question
colname = args.column

# SET TO TRUE FOR TRAINING, FALSE FOR INFERENCING TEST DATA
TRAINMODE = False
variant = 'v1'

start_time = time.time()
print(f"Start Time: {start_time}")

# Set seeds for reproducibility
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# Ensure paths are correct
train_path = f'data/nzqa/Training/train_S2{Question}.csv'
test_path = f'data/nzqa/Test/test_S2{Question}_with_score.csv'
prediction_path = f'output/predictions/bertsingle/{colname}/'
save_path = f'output/trained_model/bertsingle/{colname}/'
figure_path = os.path.join(save_path, 'figures')
analysis_path = 'output/analysis/'
os.makedirs(prediction_path, exist_ok=True)
os.makedirs(save_path, exist_ok=True)

# Load data
train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)
test_df = test_df.drop(test_df.columns[0], axis=1)

# Uncomment for debugging small samples
# train_df = train_df.head()
# test_df = test_df.head()

# Preprocess data
train_df = rename_columns(train_df, question=Question)
test_df = rename_columns(test_df, question=Question)


#select the column for the question based on the parlser argument
train_df = train_df[['Unique_ID','Response',colname]]
train_df = train_df.rename(columns={colname: 'score'})
train_df, _ = split_zeros(train_df,'score')
#train_df = scale_up(train_df)
if Question == "Q1":
    train_df = remove_signature(train_df)
    test_df = remove_signature(test_df)

# Set device and tokenizer
device = get_device()
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Prepare test data

test_df = test_df[['Unique_ID','Response',colname]]
test_df = test_df.rename(columns={colname: 'score'})
test_df_sub, test_df_zeros = split_zeros(test_df,'score')
#test_df_sub = scale_up(test_df_sub)

test_dataset = EssayDataset_baseline(test_df_sub, tokenizer)
test_loader = DataLoader(test_dataset, batch_size=CFG.eval_batch_size, shuffle=False)

# Training or inference based on TRAINMODE
if not TRAINMODE:
    model = EssayModel()
    model.load_state_dict(torch.load(save_path + f'baseline_model_{Question}_v1_seed589_{colname}.pt',map_location=device))
    model = model.to(device)
    model.eval()

    test_preds = []
    labs = []
    emb = []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Predicting"):
            ids = batch['ids'].to(device)
            mask = batch['mask'].to(device)
            labels = batch['labels'].to(device)

            outputs, embeddings= model(ids, mask)
            outputs = torch.clamp(outputs, 0.6, 5.4)
            outputs1 = torch.round(outputs-1)
            test_preds.append(outputs1.cpu().numpy())
            emb.append(embeddings.cpu().numpy())
            labs.append(labels.cpu().numpy())

    # Convert lists to numpy arrays
    embeddings = np.vstack(emb)
    
    rubric = colname
    # LOOP through the rubrics and label types
    for label_type in ['true', 'predicted']:
        if label_type == 'true':
            labels = np.vstack(labs)
        elif label_type == 'predicted':
            labels = np.vstack(test_preds)

        if rubric == 'CO':
            labels = labels[:, 1]
        elif rubric == 'AC':
            labels = labels[:, 0]
        elif rubric == 'LA':
            labels = labels[:, 2]
        elif rubric == 'ST':
            labels = labels[:, 3]
        labels = labels.astype(int)

        from sklearn.manifold import TSNE
        import matplotlib.pyplot as plt

        # Use t-SNE to reduce dimensions to 2D
        tsne = TSNE(n_components=2, random_state=42)
        embeddings_2d = tsne.fit_transform(embeddings)
        print(embeddings_2d)

        # After getting embeddings_2d
        labels = labels.flatten().astype(int)  # Ensure labels is a 1D array of integers

        # Map labels to colors
        label_colours = ['red', 'orange', 'yellow', 'green', 'blue']
        colors = [label_colours[label] for label in labels]

        # Plot the embeddings with the mapped colors
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(
            embeddings_2d[:, 0],
            embeddings_2d[:, 1],
            c=colors,
            s=50,
            edgecolor='none',
            alpha=1.0
        )

        # Create custom legend handles
        from matplotlib.lines import Line2D

        legend_elements = [
            Line2D(
                [0], [0],
                marker='o',
                color='w',
                label=str(i),
                markerfacecolor=label_colours[i],
                markersize=10
            ) for i in range(len(label_colours))
        ]

        plt.legend(handles=legend_elements, title="Score")
        plt.title(f"t-SNE of embeddings with {label_type} label for {Question} rubric {rubric}")
        #plt.show()

        # Save the plot
        plt.savefig(os.path.join(analysis_path, f'tsne_combined_{Question}_BERT_{rubric}_{label_type}_seed289.png'))

        # Use t-SNE to reduce dimensions to 3D
        tsne = TSNE(n_components=3, random_state=42)
        embeddings_3d = tsne.fit_transform(embeddings)
        #print(embeddings_3d)

        # After getting embeddings_3d
        labels = labels.flatten().astype(int)  # Ensure labels is a 1D array of integers

        # Map labels to colors
        label_colours = ['red', 'orange', 'yellow', 'green', 'blue']
        colors = [label_colours[label] for label in labels]

        # Plot the embeddings with the mapped colors in 3D
        plt.figure(figsize=(12, 10))
        ax = plt.axes(projection='3d')

        scatter = ax.scatter(
            embeddings_3d[:, 0],
            embeddings_3d[:, 1],
            embeddings_3d[:, 2],
            c=colors,
            s=50,
            edgecolor='none',
            alpha=1.0
        )

        # Create custom legend handles
        from matplotlib.lines import Line2D

        unique_labels = np.unique(labels)
        legend_elements = [
            Line2D(
                [0], [0],
                marker='o',
                color='w',
                label=str(label),
                markerfacecolor=label_colours[label],
                markersize=10
            ) for label in unique_labels
        ]

        ax.legend(handles=legend_elements, title="Score")
        ax.set_title(f"3D t-SNE of baseline BERT embeddings with {label_type} label for {Question} rubric {rubric}")
        ax.set_xlabel('Component 1')
        ax.set_ylabel('Component 2')
        ax.set_zlabel('Component 3')

        # Save the plot
        plt.savefig(os.path.join(analysis_path, f'tsne_3d_{Question}_bertsingle_{rubric}_{label_type}_seed289.png'))


