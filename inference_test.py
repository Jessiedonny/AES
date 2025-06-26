import time
import os
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from sklearn.model_selection import KFold
from tqdm import tqdm
import numpy as np
from sklearn.metrics import cohen_kappa_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


from helpers.dataset import EssayDataset
from helpers.model import EssayModel
from helpers.config import Tuning_Parameters
from helpers.utils import get_device, preprocess_data, preprocess_test_data,remove_signature, save_embeddings, save_predictions

start_time = time.time()
print(start_time)

Question = "Q1"
test_path = f'data/nzqa/Test/test_S2{Question}.csv'
prediction_path = 'output/predictions/bertbase/'
save_path = 'output/trained_model/bertbase/'
plot_path = 'output/plot/bertbase/'

test_df = pd.read_csv(test_path)
#test_df = test_df.head()

device = get_device()

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

test_df = preprocess_data(test_df)
test_df, test_df_zeros = preprocess_test_data(test_df, question=Question)
test_dataset = EssayDataset(test_df, tokenizer)
test_loader = DataLoader(test_dataset, batch_size=Tuning_Parameters.eval_batch_size, shuffle=False)

# Initialize the model
model = EssayModel()
# Load the saved model weights
model.load_state_dict(torch.load(save_path + f'bert_multitask_model_{Question}.pt',map_location=torch.device('cpu')))
model.eval() # Set the model to evaluation mode


test_preds = []
embeddings = []
with torch.no_grad():
    for batch in tqdm(test_loader, desc="Predicting"):
        ids = batch['ids'].to(device)
        mask = batch['mask'].to(device)
        outputs, pooled_output = model(ids, mask)  # Get both outputs and pooled_output
        outputs = torch.clamp(outputs, 0.6, 5.4)  # Clamp predictions to the range 1-5
        outputs = torch.round(outputs)
        test_preds.append(outputs.cpu().numpy())
        embeddings.append(pooled_output.cpu().numpy())

test_preds = np.concatenate(test_preds, axis=0)
save_predictions(test_preds, test_df, test_df_zeros, prediction_path, Question)

# Save the embeddings
embeddings = np.concatenate(embeddings, axis=0)
save_embeddings(embeddings, test_df, prediction_path, Question)

end_time = time.time()
run_time = end_time - start_time
print('Total time taken', run_time)

# Make sure the GPU resources are released
torch.cuda.empty_cache()
