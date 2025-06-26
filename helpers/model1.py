import torch
from torch import nn
from transformers import BertModel



class EssayModel(torch.nn.Module):   # baseline model using bert with multi-head output
    def __init__(self):
        super(EssayModel, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.fc = torch.nn.Linear(768, 4)  # 768 is the hidden size of BERT, 4 is the number of scores to predict
    
    def forward(self, ids, mask):
        outputs = self.bert(ids, attention_mask=mask)
        pooled_output = outputs[1]  # this is the CLS representation
        return self.fc(pooled_output), pooled_output



class EssayModel2(torch.nn.Module):  # model 2 - changed embedding dimension to include manual features
    def __init__(self, additional_feature_dim):
        super(EssayModel2, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.fc = torch.nn.Linear(768 + additional_feature_dim, 4)  # Adjusted input dimension

    def forward(self, ids, mask, additional_features):
        outputs = self.bert(ids, attention_mask=mask)
        pooled_output = outputs[1]  # CLS representation
        
        # Concatenate BERT output with additional features
        combined_output = torch.cat((pooled_output, additional_features), dim=1)
        
        return self.fc(combined_output), pooled_output
    
class EssayModel2v2(torch.nn.Module):  # model 2 - changed embedding dimension to include manual features
    def __init__(self, additional_feature_dim):
        super(EssayModel2v2, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        
        # Adding a dense layer after concatenation
        self.fc1 = torch.nn.Linear(768 + additional_feature_dim, 512)  # First dense layer
        self.fc2 = torch.nn.Linear(512, 4)  # Final output layer for regression

        self.relu = torch.nn.ReLU()  # Activation function
        self.dropout = torch.nn.Dropout(0.3)  # Dropout for regularization

    def forward(self, ids, mask, additional_features):
        outputs = self.bert(ids, attention_mask=mask)
        pooled_output = outputs[1]  # CLS representation
        
        # Concatenate BERT output with additional features
        combined_output = torch.cat((pooled_output, additional_features), dim=1)
        
        # Pass through the additional dense layer
        x = self.fc1(combined_output)
        x = self.relu(x)
        x = self.dropout(x)
        
        # Final output layer
        x = self.fc2(x)
        
        return x, pooled_output
    
class EssayModel2v3(torch.nn.Module):  # model 2 - changed embedding dimension to include manual features
    def __init__(self, additional_feature_dim):
        super(EssayModel2v3, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        
        # Adding a dense layer after concatenation
        self.fc1 = torch.nn.Linear(768 + additional_feature_dim, 768)  # First dense layer
        self.fc2 = torch.nn.Linear(768, 4)  # Final output layer for regression

        self.relu = torch.nn.ReLU()  # Activation function
        self.dropout = torch.nn.Dropout(0.3)  # Dropout for regularization

    def forward(self, ids, mask, additional_features):
        outputs = self.bert(ids, attention_mask=mask)
        pooled_output = outputs[1]  # CLS representation
        
        # Concatenate BERT output with additional features
        combined_output = torch.cat((pooled_output, additional_features), dim=1)
        
        # Pass through the additional dense layer
        x = self.fc1(combined_output)
        x = self.relu(x)
        x = self.dropout(x)
        
        # Final output layer
        x = self.fc2(x)
        
        return x, pooled_output


class EssayModel3(nn.Module):  # model 3 - combined bert embedding with sentence bert embedding and manual features
    def __init__(self, sentence_embedding_dim, manual_feature_dim=0):
        super(EssayModel3, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.fc = nn.Linear(768 + sentence_embedding_dim + manual_feature_dim, 4)  # Adjusted input dimension

    def forward(self, ids, mask, manual_features):
        # BERT model forward pass
        outputs = self.bert(ids, attention_mask=mask)
        bert_embedding = outputs[1]  # CLS representation (shape: [batch_size, 768])

        # Concatenate BERT output with the averaged sentence embeddings and manual features
        combined_embedding = torch.cat((bert_embedding, manual_features), dim=1)  # shape: [batch_size, 768 + 768 + manual_feature_dim]

        # Pass the combined embedding through the fully connected layer
        scores = self.fc(combined_embedding)
        return scores, combined_embedding
    
class EssayModel3v2(nn.Module):  # model 3 - combined bert embedding with sentence bert embedding and manual features
    def __init__(self, sentence_embedding_dim, manual_feature_dim=0):
        super(EssayModel3v2, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.fc1 = nn.Linear(768 + sentence_embedding_dim + manual_feature_dim, 1024)  # Adjusted input dimension
        self.fc2 = nn.Linear(1024,4)

        self.relu = torch.nn.ReLU()  # Activation function
        self.dropout = torch.nn.Dropout(0.3)  # Dropout for regularization

    def forward(self, ids, mask, manual_features):
        # BERT model forward pass
        outputs = self.bert(ids, attention_mask=mask)
        bert_embedding = outputs[1]  # CLS representation (shape: [batch_size, 768])

        # Concatenate BERT output with the averaged sentence embeddings and manual features
        combined_embedding = torch.cat((bert_embedding, manual_features), dim=1)  # shape: [batch_size, 768 + 768 + manual_feature_dim]

        x = self.fc1(combined_embedding)
        x = self.relu(x)
        x = self.dropout(x)

        # Pass the combined embedding through the fully connected layer
        scores = self.fc2(x)
        return scores, bert_embedding
    
class EssayModel3v3(nn.Module):  # model 3 - combined bert embedding with sentence bert embedding and manual features
    def __init__(self, sentence_embedding_dim, manual_feature_dim=0):
        super(EssayModel3v3, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.fc1 = nn.Linear(768 + sentence_embedding_dim + manual_feature_dim, 768)  # Adjusted input dimension
        self.fc2 = nn.Linear(768,4)

        self.relu = torch.nn.ReLU()  # Activation function
        self.dropout = torch.nn.Dropout(0.3)  # Dropout for regularization

        self.attention = nn.MultiheadAttention(embed_dim=768, num_heads=4)

    def forward(self, ids, mask, manual_features):
        # BERT model forward pass
        outputs = self.bert(ids, attention_mask=mask)
        bert_embedding = outputs[1]  # CLS representation (shape: [batch_size, 768])

        # Concatenate BERT output with the averaged sentence embeddings and manual features
        combined_embedding = torch.cat((bert_embedding, manual_features), dim=1)  # shape: [batch_size, 768 + 768 + manual_feature_dim]
        y=combined_embedding

        combined_embedding = self.fc1(combined_embedding)

        # Apply attention
        attn_output, _ = self.attention(combined_embedding.unsqueeze(0), combined_embedding.unsqueeze(0), combined_embedding.unsqueeze(0))
        attn_output = attn_output.squeeze(0)

        x = self.relu(attn_output)
        x = self.dropout(x)

        # Pass the combined embedding through the fully connected layer
        scores = self.fc2(x)
        return scores, bert_embedding
    

class EssayModel3v4(nn.Module):
    def __init__(self, sentence_embedding_dim, manual_feature_dim=0):
        super(EssayModel3v4, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.fc1 = nn.Linear(768 + sentence_embedding_dim + manual_feature_dim, 512)  # Adjusted input dimension
        self.fc2 = nn.Linear(512, 1)  # Output layer to produce a single score

        self.relu = torch.nn.ReLU()  # Activation function
        self.dropout = torch.nn.Dropout(0.3)  # Dropout for regularization

        # Adjusted attention mechanism to align with the new input dimensions
        self.attention = nn.MultiheadAttention(embed_dim=512, num_heads=1)

    def forward(self, ids, mask, manual_features):
        # BERT model forward pass
        outputs = self.bert(ids, attention_mask=mask)
        bert_embedding = outputs[1]  # CLS token representation (shape: [batch_size, 768])

        # Concatenate BERT output with the sentence embeddings and manual features
        combined_embedding = torch.cat((bert_embedding, manual_features), dim=1)  # shape: [batch_size, 768 + sentence_embedding_dim + manual_feature_dim]

        # Pass through the first fully connected layer
        combined_embedding = self.fc1(combined_embedding)

        # Apply attention mechanism
        attn_output, _ = self.attention(combined_embedding.unsqueeze(0), combined_embedding.unsqueeze(0), combined_embedding.unsqueeze(0))
        attn_output = attn_output.squeeze(0)

        # Apply ReLU activation and dropout
        x = self.relu(attn_output)
        x = self.dropout(x)

        # Final output layer to predict the score
        scores = self.fc2(x)

        return scores, bert_embedding
    
class EssayModel3v5(torch.nn.Module):  # model 2 - changed embedding dimension to include manual features
    def __init__(self, additional_feature_dim, manual_feature_dim):
        super(EssayModel3v5, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        
        # Adding a dense layer after concatenation
        self.fc1 = torch.nn.Linear(768 + additional_feature_dim + manual_feature_dim, 542)  # First dense layer
        self.fc2 = torch.nn.Linear(542, 4)  # Final output layer for regression

        self.relu = torch.nn.ReLU()  # Activation function
        self.dropout = torch.nn.Dropout(0.3)  # Dropout for regularization

    def forward(self, ids, mask, additional_features):
        outputs = self.bert(ids, attention_mask=mask)
        pooled_output = outputs[1]  # CLS representation
        
        # Concatenate BERT output with additional features
        combined_output = torch.cat((pooled_output, additional_features), dim=1)
        
        # Pass through the additional dense layer
        x = self.fc1(combined_output)
        x = self.relu(x)
        x = self.dropout(x)
        final_embedding = x

        # Final output layer
        x = self.fc2(x)
        
        return x, final_embedding


