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
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import cohen_kappa_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


from helpers.dataset import EssayDataset2sb
from helpers.model import EssayModel2v2
from helpers.config import CFG
from helpers.utils import get_device, rename_columns, split_zeros,scale_up, scale_down, remove_signature, save_embeddings, get_predictions,save_predictions

import argparse

# nohup python3 -u step2_hybrid_withSB_v2.py --question Q1 >output/logs/log_hybrid_withSB_q1_v2.out 2>&1 &
# nohup python3 -u step2_hybrid_withSB_v2.py --question Q2 >output/logs/log_hybrid_withSB_q2_v2.out 2>&1 &

# Argument parsing
parser = argparse.ArgumentParser(description='Run inference with a trained BERT model for a specific question.')
parser.add_argument('--question', type=str, required=True, help='Question identifier (Q1 or Q2)')
args = parser.parse_args()
Question = args.question

# SET TO TRUE FOR TRAINING, FALSE FOR INFERENCING TEST DATA
TRAINMODE = False
variant = 'v2'

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

sentence_embedding_dim = 768
sentence_embedding_train = pd.read_csv(f'data/nzqa/Training/train_sentence_embedding_{Question}.csv')
sentence_embedding_test = pd.read_csv(f'data/nzqa/Test/test_sentence_embedding_{Question}.csv')

prediction_path = 'output/predictions/hybridsb/'
save_path = 'output/trained_model/hybridsb/'
analysis_path = 'output/analysis/'
figure_path = os.path.join(save_path, 'figures')
os.makedirs(prediction_path, exist_ok=True)
os.makedirs(save_path, exist_ok=True)
os.makedirs(figure_path, exist_ok=True)

# Load data
train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)
test_df = test_df.drop(test_df.columns[0], axis=1)

# Uncomment for debugging small samples
# train_df = train_df.head()
# test_df = test_df.head()

# Preprocess data
train_df = rename_columns(train_df, question=Question)
train_df, train_df_zeros = split_zeros(train_df)
#train_df = scale_up(train_df,Question)
if Question == "Q1":
    train_df = remove_signature(train_df)
    test_df = remove_signature(test_df)

# Set device and tokenizer
device = get_device()
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Prepare test data
test_df = rename_columns(test_df, question=Question)
test_df_sub, test_df_zeros = split_zeros(test_df)
test_dataset = EssayDataset2sb(test_df_sub, sentence_embedding_test, tokenizer)
test_loader = DataLoader(test_dataset, batch_size=CFG.eval_batch_size, shuffle=False)

# Training or inference based on TRAINMODE
if not TRAINMODE:
    model = EssayModel2v2(sentence_embedding_dim)
    model.load_state_dict(torch.load(save_path + f'hybridsb_model_{Question}_{variant}_seed589.pt', map_location=torch.device('cpu')))
    model.to(device)
    # Access the weights of the first fully connected layer (fc1)
    #manual_feature_weights = model.fc1.weight[:, 768:]  # Assuming manual features are concatenated after BERT embeddings
    manual_feature_weights = model.fc1.weight
    manual_feature_weights = manual_feature_weights.cpu().detach().numpy()

    # plt.figure(figsize=(10, 6))
    # plt.bar(range(len(manual_feature_weights.mean(axis=0))), manual_feature_weights.mean(axis=0))
    # plt.xlabel("Feature Index")
    # plt.ylabel("Mean Weight")
    # plt.title("Weights of all Features in the First Fully Connected Layer of the Hybrid Model")
    # plt.show()

    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    plt.figure(figsize=(10, 6))
    mean_weights = manual_feature_weights.mean(axis=0)
    feature_indices = range(len(mean_weights))
    plt.bar(feature_indices, mean_weights)
    plt.xlabel("Feature Index")
    plt.ylabel("Mean Weight")
    plt.title("Weights of all Features")

    # Get current axes
    ax = plt.gca()

    # Calculate the coordinates for the rectangle
    num_features = len(mean_weights)
    x_start = num_features - 768 - 0.5  # Start position (adjusted for bar width)
    width = 768                         # Width covers the last 30 features
    ymin, ymax = ax.get_ylim()         # y-axis limits

    # Create a rectangle patch
    rect = patches.Rectangle(
        (x_start, ymin),       # (x,y) starting point
        width,                 # width
        ymax - ymin,           # height
        linewidth=2,
        edgecolor='red',
        facecolor='none'
    )

    # Add the rectangle to the plot
    ax.add_patch(rect)

    # Optionally, add text annotation to label the rectangle
    ax.text(
        x_start + width / 2, 
        ymax, 
        'SentenceBERT Features', 
        horizontalalignment='center', 
        verticalalignment='bottom', 
        fontsize=12, 
        color='red'
    )

    #plt.show()

    # save the plot
    plt.savefig(f'{analysis_path}weights_{Question}_{variant}_hybridsb_seed589_1.png')

