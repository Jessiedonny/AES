import time
import os
import random
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch import nn
from transformers import BertTokenizer
from transformers import get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import numpy as np
from sklearn.metrics import cohen_kappa_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from helpers.dataset import EssayDataset3v9 as EssayDataset3
from helpers.model import EssayModel3v10 as EssayModel3
from helpers.config import CFG_multiview
from helpers.utils import get_device, rename_columns, split_zeros, get_predictions
#removed scale_up and scale_down functions compared with v9

# Argument parsing
import argparse

# nohup python3 -u step3_multiview_v10.py --question Q1 >output/logs/log_multiview_q1_v10.out 2>&1 &
# nohup python3 -u step3_multiview_v10.py --question Q2 >output/logs/log_multiview_q2_v10.out 2>&1 &

parser = argparse.ArgumentParser(description='Run inference with a trained BERT model for a specific question.')
parser.add_argument('--question', type=str, required=True, help='Question identifier (Q1 or Q2)')
# parser.add_argument('--label', type=str, required=False, help='Label type (true or predicted)')
# parser.add_argument('--rubric', type=str, required=False, help='rubric type (AC or CO or LA or ST)')
args = parser.parse_args()
Question = args.question
# label_type = args.label
# rubric = args.rubric

#SET TO TRUE FOR TRAINING, FALSE FOR INFERENCING TEST DATA
TRAINMODE = False
variant = 'v10'

start_time = time.time()
print(start_time)

train_path = f'data/nzqa/Training/train_S2{Question}.csv'
test_path = f'data/nzqa/Test/test_S2{Question}_with_score.csv'
prediction_path = 'output/predictions/multiviewv10/'
save_path = 'output/trained_model/multiviewv10/'
analysis_path = 'output/analysis/'
figure_path = os.path.join(save_path, 'figures')

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)
test_df = test_df.drop(test_df.columns[0], axis=1)

# train_df = train_df.head(20)
# test_df = test_df.head(20)

manual_features_df_train = pd.read_csv(f'data/nzqa/Training/train_manual_features_{Question}.csv')
manual_features_df_test = pd.read_csv(f'data/nzqa/Test/test_manual_features_{Question}.csv')

sentence_embedding_dim = 768
sentence_embedding_train = pd.read_csv(f'data/nzqa/Training/train_sentence_embedding_{Question}.csv')
sentence_embedding_test = pd.read_csv(f'data/nzqa/Test/test_sentence_embedding_{Question}.csv')

train_df = rename_columns(train_df, question=Question)
train_df, _ = split_zeros(train_df)
#train_df = scale_up(train_df,Question)

# apply manual feature selection from the manual features of the training and test set
manual_feature_dim = 30
df = pd.merge(train_df,manual_features_df_train, left_on="Unique_ID", right_on="Unique_ID", how="inner")
x = df.iloc[:,6:]
y = df.iloc[:,2]
corr_matrix1 = x.corrwith(y)
y = df.iloc[:,3]
corr_matrix2 = x.corrwith(y)
y = df.iloc[:,4]
corr_matrix3 = x.corrwith(y)
y = df.iloc[:,5]
corr_matrix4 = x.corrwith(y)

# get the duplicated features that are highly correlated with the target
corr_matrix = (corr_matrix1 + corr_matrix2 + corr_matrix3 + corr_matrix4)/4
top_features = corr_matrix.abs().nlargest(manual_feature_dim).index
top_features = top_features.tolist()
selected_features = ["Unique_ID"]+top_features
manual_features_df_train = manual_features_df_train[selected_features]
manual_features_df_test = manual_features_df_test[selected_features]

# Prepare the tokenizer
device = get_device()
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

test_df = rename_columns(test_df, question=Question)
test_df_sub, test_df_zeros = split_zeros(test_df)
test_dataset = EssayDataset3(test_df_sub, sentence_embedding_test, manual_features_df_test, tokenizer)
test_loader = DataLoader(test_dataset, batch_size=CFG_multiview.eval_batch_size, shuffle=False)

train_dataset = EssayDataset3(train_df, sentence_embedding_train, manual_features_df_train, tokenizer)
train_loader = DataLoader(train_dataset, batch_size=CFG_multiview.train_batch_size, shuffle=True)



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

def train_model(model, dataloader, optimizer, scheduler, device):
    model.train()
    total_loss = 0
    preds = []
    actuals = []
    for batch in tqdm(dataloader, desc="Training"):
        ids = batch['ids'].to(device)
        mask = batch['mask'].to(device)
        additional_features = batch['additional_features'].to(device)
        manual_features = batch['manual_features'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()
        outputs, pooled_output = model(ids, mask, additional_features, manual_features)
        loss = torch.nn.MSELoss()(outputs, labels)  # loss for each score
        loss.backward()
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()

        preds.append(outputs.cpu().detach().numpy())
        actuals.append(labels.cpu().numpy())

    preds = np.concatenate(preds, axis=0)
    actuals = np.concatenate(actuals, axis=0)
    kappas = []
    for i in range(4):
        kappa = cohen_kappa_score(np.round(preds[:, i], 0), np.round(actuals[:, i]), weights='quadratic')
        kappas.append(kappa)

    return total_loss / len(dataloader), kappas

def evaluate_model(model, dataloader, device):
    model.eval()
    preds = []
    actuals = []
    total_loss = 0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            ids = batch['ids'].to(device)
            mask = batch['mask'].to(device)
            additional_features = batch['additional_features'].to(device)
            manual_features = batch['manual_features'].to(device)
            labels = batch['labels'].to(device)

            outputs, _ = model(ids, mask, additional_features, manual_features)
            loss = torch.nn.MSELoss()(outputs, labels)
            total_loss += loss.item()

            outputs = torch.clamp(outputs, 0.0, 4.4)
            outputs = torch.round(outputs)  # Round predictions to the nearest integer
            preds.append(outputs.cpu().numpy())
            actuals.append(labels.cpu().numpy())

    preds = np.concatenate(preds, axis=0)
    actuals = np.concatenate(actuals, axis=0)
    kappas = []
    for i in range(4):
        kappa = cohen_kappa_score(np.round(preds[:, i], 0), np.round(actuals[:, i]), weights='quadratic')
        kappas.append(kappa)
    return total_loss / len(dataloader), kappas, preds

if not TRAINMODE:
    model = EssayModel3(sentence_embedding_dim, manual_feature_dim).to(device)
    # Load the saved model
    model.load_state_dict(torch.load(save_path + f'multiview_model_{Question}_{variant}_seed289.pt', map_location=device))
    model.eval()

    emb = []
    labs = []  # or scores depending on what you're predicting
    test_preds = []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Predicting"):
                ids = batch['ids'].to(device)
                mask = batch['mask'].to(device)
                additional_features = batch['additional_features'].to(device)
                manual_features = batch['manual_features'].to(device)
                labels = batch['labels'].to(device)

                outputs, embeddings = model(ids, mask, additional_features, manual_features)
                outputs1 = torch.round(outputs)
                test_preds.append(outputs1.cpu().numpy())
                emb.append(embeddings.cpu().numpy())
                labs.append(labels.cpu().numpy())


    # Convert lists to numpy arrays
    embeddings = np.vstack(emb)
    
    # LOOP through the rubrics and label types
    for rubric in ['CO', 'AC', 'LA', 'ST']:
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
            plt.title(f"t-SNE of last layer embeddings with {label_type} label for {Question} rubric {rubric} in multiview")
            #plt.show()

            # Save the plot
            plt.savefig(os.path.join(analysis_path, f'tsne_lastlayer_{Question}_multiview_{rubric}_{label_type}_seed289.png'))

            # Plot the embeddings with the mapped colors in 3D
            from mpl_toolkits.mplot3d import Axes3D  # Import for 3D plotting

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
            ax.set_title(f"3D t-SNE of embeddings with {label_type} label for {Question} rubric {rubric}")
            ax.set_xlabel('Component 1')
            ax.set_ylabel('Component 2')
            ax.set_zlabel('Component 3')

            # Save the plot
            plt.savefig(os.path.join(analysis_path, f'tsne_3d_{Question}_multiview_{rubric}_{label_type}_seed289.png'))


  