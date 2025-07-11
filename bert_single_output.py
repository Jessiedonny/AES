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
TRAINMODE = True
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

# Train model function
def train_model(model, dataloader, optimizer, scheduler, device):
    model.train()
    total_loss = 0
    preds = []
    actuals = []
    for batch in tqdm(dataloader, desc="Training"):
        ids = batch['ids'].to(device)
        mask = batch['mask'].to(device)
        labels = batch['labels'].float().to(device)  # Ensure labels are floats

        optimizer.zero_grad()
        outputs, pooled_output = model(ids, mask)
        loss = torch.nn.MSELoss()(outputs, labels)  # MSE loss for regression
        loss.backward()
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()

        preds.append(outputs.cpu().detach().numpy())
        actuals.append(labels.cpu().numpy())

    preds = np.concatenate(preds, axis=0)
    actuals = np.concatenate(actuals, axis=0)
    #kappas = [cohen_kappa_score(np.round(preds[:, i]), np.round(actuals[:, i]), weights='quadratic') for i in range(4)]
    #preds and acturals are one float value for each category, adjust code to calculate kappa
    kappa = cohen_kappa_score(np.round(preds), np.round(actuals), weights='quadratic')


    return total_loss / len(dataloader), kappa

# Evaluate model function
def evaluate_model(model, dataloader, device):
    model.eval()
    preds = []
    actuals = []
    total_loss = 0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            ids = batch['ids'].to(device)
            mask = batch['mask'].to(device)
            labels = batch['labels'].float().to(device)

            outputs, _ = model(ids, mask)
            loss = torch.nn.MSELoss()(outputs, labels)
            total_loss += loss.item()

            outputs = torch.clamp(outputs, 0.0, 4.4)
            outputs = torch.round(outputs)

            preds.append(outputs.cpu().numpy())
            actuals.append(labels.cpu().numpy())

    preds = np.concatenate(preds, axis=0)
    actuals = np.concatenate(actuals, axis=0)
    kappas = cohen_kappa_score(np.round(preds), np.round(actuals), weights='quadratic')

    return total_loss / len(dataloader), kappas, preds

# Training or inference based on TRAINMODE
if not TRAINMODE:
    model = EssayModel()
    model.load_state_dict(torch.load(save_path + f'bert_base_model_{Question}_{colname}.pt', map_location=torch.device('cpu')))
    model.to(device)
else:
    performance_log = pd.DataFrame(columns=['Seed', 'Epoch', 'Train_QWK', 'Train_Loss', 'Val_QWK', 'Val_Loss'])
    all_val_kappas = []
    all_test_kappas = []

    for seed in CFG.seeds:
        set_seed(seed)
        train, val = train_test_split(train_df, test_size=0.1, random_state=seed)
        train_dataset = EssayDataset_baseline(train, tokenizer)
        train_loader = DataLoader(train_dataset, batch_size=CFG.train_batch_size, shuffle=True)

        val_dataset = EssayDataset_baseline(val, tokenizer)
        val_loader = DataLoader(val_dataset, batch_size=CFG.train_batch_size, shuffle=True)

        model = EssayModel().to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=CFG.lr)
        total_steps = len(train_loader) * CFG.train_epochs
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=CFG.warmup_steps, num_training_steps=total_steps)

        best_val_kappa_mean = float('-inf')
        patience_counter = 0

        # Lists to store losses for plotting
        train_losses = []
        val_losses = []

        for epoch in range(CFG.train_epochs):
            train_loss, train_kappa = train_model(model, train_loader, optimizer, scheduler, device)
            print(f"Epoch {epoch + 1}, Training Loss: {train_loss}, Training Kappa: {train_kappa}")
            val_loss, val_kappa, _ = evaluate_model(model, val_loader, device)
            print(f"Epoch {epoch + 1}, Validation Loss: {val_loss}, Validation Kappa: {val_kappa}")
            val_kappa_mean = np.mean(val_kappa)

            train_losses.append(train_loss)
            val_losses.append(val_loss)

            log = pd.Series({
                'Seed': seed,
                'Epoch': epoch + 1,
                'Train_QWK': train_kappa,
                'Train_Loss': train_loss,
                'Val_QWK': val_kappa,
                'Val_Loss': val_loss
            })
            
            performance_log = pd.concat([performance_log, log.to_frame().T], ignore_index=True)

            if val_kappa_mean > best_val_kappa_mean:
                best_val_kappa_mean = val_kappa_mean
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= CFG.early_stopping_patience:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break

        performance_log.to_csv(save_path + f'baseline_performance_log_{Question}_{variant}.csv', index=False)
           
        # Plot the training and validation loss
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label='Training Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title(f'Training and Validation Loss for {Question} Seed {seed}')
        plt.legend()
        plt.savefig(os.path.join(figure_path, f'training_val_loss_{Question}_seed_{seed}.png'))
        plt.close()

        #torch.save(model.state_dict(), save_path + f'baseline_model_{Question}_{variant}_seed{seed}.pt')
        all_val_kappas.append(val_kappa)

        model.eval()

        test_preds = []
        test_preds_float = []
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Predicting"):
                ids = batch['ids'].to(device)
                mask = batch['mask'].to(device)
                outputs, _ = model(ids, mask)
                outputs = torch.clamp(outputs, 0.0, 4.4)
                test_preds_float.append(outputs.cpu().numpy())
                outputs1 = torch.round(outputs)
                test_preds.append(outputs1.cpu().numpy())

        test_preds_float = np.concatenate(test_preds_float, axis=0)
        test_preds_float = pd.DataFrame(test_preds_float, columns=['score'])
        #test_preds_float = scale_down(test_preds_float)
        test_predictions_float = get_prediction(test_preds_float, test_df_sub, test_df_zeros)  
        test_predictions_float.columns = [f"{col}_pred_float" if col != "Unique_ID" else col for col in test_predictions_float.columns]
        test_predictions_float.to_csv(prediction_path + f'test_{Question}_bertbase{variant}_seed{seed}_float.csv', index=False)

        test_preds = np.concatenate(test_preds, axis=0)
        test_preds = pd.DataFrame(test_preds, columns=['score'])
        #test_preds = scale_down(test_preds)
        test_predictions = get_prediction(test_preds, test_df_sub, test_df_zeros)
       

        # Save predictions
        # save_predictions(test_preds, test_df_sub, test_df_zeros, prediction_path, f"test_{variant}_{seed}", Question)
        # save_predictions(test_preds_float, test_df_sub, test_df_zeros, prediction_path, f"test_{variant}_float_{seed}", Question)

        # Check the columns of test_predictions
        print("Columns in test_predictions before renaming:", test_predictions.columns)

        # Adjust the renaming logic
        test_predictions= test_predictions.rename(columns={'score': 'score_pred'})
        print("Columns in test_predictions after renaming:", test_predictions.columns)

        test_true = test_df.copy()
        test_true = test_true.rename(columns={'score': 'score_true'})

        dt = pd.merge(test_predictions, test_true, on="Unique_ID")
        dt.to_csv(prediction_path + f'test_{Question}_bertbase{variant}_seed{seed}_{colname}.csv', index=False)
        #dt['score_pred'] = dt['score_pred'].round().astype(int)
        #dt['score_true'] = dt['score_true'].round().astype(int)
        test_kappa = round(cohen_kappa_score(dt['score_pred'], dt['score_true'], weights='quadratic'), 3)
        all_test_kappas.append(test_kappa)
        print(f"Seed {seed} Test Kappa: {test_kappa}")

        log = pd.Series({
            'Seed': seed,
            'Epoch': "test",
            'Train_QWK': "--",
            'Train_Loss': "--",
            'Val_QWK': test_kappa,
            'Val_Loss': "--"
        })
        performance_log = pd.concat([performance_log, log.to_frame().T], ignore_index=True)

        # PLOT CONFUSION MATRIX
  
        y_true = dt['score_true']
        predictions = dt['score_pred']
        cm = confusion_matrix(y_true, predictions, labels=[x for x in range(0,5)])
        draw_cm = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=[x for x in range(0,5)])
        plt.figure(figsize=(10, 6))
        draw_cm.plot()
        plt.title(f'confusion matrix for {Question} Seed {seed}')
        plt.savefig(os.path.join(figure_path, f'confusion matrix for {Question} Seed {seed}-{colname}.png'))
        plt.close()



    train_time = time.time()

    # print("Training time:", train_time - start_time)
    #format the Training time in hours, minutes and seconds
    hours, rem = divmod(train_time - start_time, 3600)
    minutes, seconds = divmod(rem, 60)
    print("{:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))


    avg_val_kappas = np.mean(all_val_kappas)
    avg_test_kappas = np.mean(all_test_kappas)

    print(f"Average Validation Kappa: {avg_val_kappas}")
    print(f"Average Test Kappa: {avg_test_kappas}")

    log = pd.Series({
        'Seed': "average",
        'Epoch': "all_val",
        'Train_QWK': "--",
        'Train_Loss': "--",
        'Val_QWK': avg_val_kappas,
        'Val_Loss': "--"
    })
    performance_log = pd.concat([performance_log, log.to_frame().T], ignore_index=True)

    log = pd.Series({
        'Seed': "average",
        'Epoch': "all_test",
        'Train_QWK': "--",
        'Train_Loss': "--",
        'Val_QWK': avg_test_kappas,
        'Val_Loss': "--"
    })
    performance_log = pd.concat([performance_log, log.to_frame().T], ignore_index=True)
    performance_log.to_csv(save_path + f'baseline_performance_log_{Question}_{variant}_{colname}.csv', index=False)
    torch.save(model.state_dict(), save_path + f'baseline_model_{Question}_{variant}_seed{seed}_{colname}.pt')
        
