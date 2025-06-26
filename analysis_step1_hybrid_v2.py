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


from helpers.dataset import EssayDataset2
from helpers.model import EssayModel2v2
from helpers.config import CFG
from helpers.utils import get_device, rename_columns, split_zeros,scale_up, scale_down, remove_signature, save_embeddings
import matplotlib.pyplot as plt


import argparse

#  nohup python3 -u step1_hybrid_v3.py --question Q1 >output/logs/log_hybrid_q1_v3.out 2>&1 &
#  nohup python3 -u step1_hybrid_v3.py --question Q2 >output/logs/log_hybrid_q2_v3.out 2>&1 &

# Argument parsing
parser = argparse.ArgumentParser(description='Run inference with a trained BERT model for a specific question.')
parser.add_argument('--question', type=str, required=True, help='Question identifier (Q1 or Q2)')
args = parser.parse_args()
Question = args.question

# SET TO TRUE FOR TRAINING, FALSE FOR INFERENCING TEST DATA
TRAINMODE = False
variant = 'v3'

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
manual_features_path_train = f'data/nzqa/Training/train_manual_features_{Question}.csv'
manual_features_path_test = f'data/nzqa/Test/test_manual_features_{Question}.csv'

prediction_path = 'output/predictions/hybrid/'
save_path = 'output/trained_model/hybrid/'
figure_path = os.path.join(save_path, 'figures')
analysis_path = 'output/analysis/'
os.makedirs(prediction_path, exist_ok=True)
os.makedirs(save_path, exist_ok=True)
os.makedirs(figure_path, exist_ok=True)

# Load data
train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)
test_df = test_df.drop(test_df.columns[0], axis=1)
manual_features_df_train = pd.read_csv(manual_features_path_train)
manual_features_df_test = pd.read_csv(manual_features_path_test)
#manual_feature_dim = manual_features_df_train.shape[1] - 1


# Uncomment for debugging small samples
# train_df = train_df.head()
# test_df = test_df.head()

# Preprocess data
train_df = rename_columns(train_df, question=Question)
train_df, train_df_zeros = split_zeros(train_df)
train_df = scale_up(train_df,Question)
if Question == "Q1":
    train_df = remove_signature(train_df)

# apply manual feature selection from the manual features of the training and test set
manual_feature_dim = 30
df = pd.merge(train_df,manual_features_df_train, left_on="Unique_ID", right_on="Unique_ID", how="inner")
x = df.iloc[:,6:]
y = df.iloc[:,2]
corr_matrix = x.corrwith(y)
top_features = corr_matrix.abs().nlargest(manual_feature_dim).index
top_features = top_features.tolist()
selected_features = ["Unique_ID"]+top_features
manual_features_df_train = manual_features_df_train[selected_features]
manual_features_df_test = manual_features_df_test[selected_features]

# Set device and tokenizer
device = get_device()
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Prepare test data
test_df = rename_columns(test_df, question=Question)
test_df_sub, test_df_zeros = split_zeros(test_df)
test_dataset = EssayDataset2(test_df_sub, manual_features_df_test, tokenizer)
test_loader = DataLoader(test_dataset, batch_size=CFG.eval_batch_size, shuffle=False)


model = EssayModel2v2(manual_feature_dim)
model.load_state_dict(torch.load(save_path + f'hybrid_model_{Question}_v2_seed489.pt', map_location=torch.device('cpu')))
model.eval()

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
x_start = num_features - 30 - 0.5  # Start position (adjusted for bar width)
width = 30                         # Width covers the last 30 features
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
    'Manual Features', 
    horizontalalignment='center', 
    verticalalignment='bottom', 
    fontsize=12, 
    color='red'
)

#plt.show()

# save the plot
plt.savefig(f'{analysis_path}weights_{Question}_{variant}_hybrid_seed589_1.png')


