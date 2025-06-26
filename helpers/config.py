class CFG:  # for the baseline multitask
    #n_splits = 10  # k folds
    seeds = [189, 289, 389, 489, 589]
    #seeds = [189]
    max_length = 384
    lr = 2e-5
    train_batch_size = 32
    eval_batch_size = 64
    train_epochs = 30
    warmup_steps = 50  # Added warmup steps for learning rate scheduler
    early_stopping_patience = 8

class CFG_baseline:  # for the baseline multitask
    #n_splits = 10  # k folds
    seeds = [189, 289, 389, 489, 589]
    max_length = 384
    lr = 2e-5
    train_batch_size = 32
    eval_batch_size = 64
    train_epochs = 30
    warmup_steps = 50  # Added warmup steps for learning rate scheduler
    early_stopping_patience = 8

class CFG_multiview:
    seeds = [189, 289, 389, 489, 589]
    #seeds = [189]
    max_length = 384
    lr = 2e-5  
    train_batch_size = 32 
    eval_batch_size = 64
    train_epochs = 30
    warmup_steps = 50  # Added warmup steps for learning rate scheduler
    early_stopping_patience = 8

class CFG_multiview_kaggle:
    #seeds = [189, 289, 389, 489, 589]
    seeds = [189,289]
    max_length = 384
    lr = 2e-5  # Typically lower learning rate for BERT fine-tuning
    train_batch_size = 16  # Adjusted batch size for potential memory constraints
    eval_batch_size = 32
    train_epochs = 10
    warmup_steps = 50  # Added warmup steps for learning rate scheduler
    early_stopping_patience = 3

class CFG_bert_singleoutput:  # for the baseline bert with single output
    #n_splits = 10  # k folds
    seeds = [189, 289, 389, 489, 589]
    #seeds = [189]
    max_length = 384
    lr = 2e-5
    train_batch_size = 32
    eval_batch_size = 32
    train_epochs = 30
    warmup_steps = 50  # Added warmup steps for learning rate scheduler
    early_stopping_patience = 4

class CFG_test:  # for the baseline multitask
    #n_splits = 10  # k folds
    #seeds = [189, 289, 389, 489, 589]
    seeds = [189]
    max_length = 384
    lr = 2e-5
    train_batch_size = 32
    eval_batch_size = 32
    train_epochs = 30
    warmup_steps = 50  # Added warmup steps for learning rate scheduler
    early_stopping_patience = 0
