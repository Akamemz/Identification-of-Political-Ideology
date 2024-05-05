
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import csv

from gensim.parsing.preprocessing import remove_stopwords
import gensim.downloader as api

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, TensorDataset, DataLoader, RandomSampler,SequentialSampler
import torch.nn.functional as F

from torchtext.data.utils import get_tokenizer

from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score, f1_score, confusion_matrix


#------------------------------------------------------------------------------------
#
# Device set up and import json data
#
#------------------------------------------------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device available for running: ")
print(device)


print("Reading data...")
def read_json_folder(folder_path):
    """
    Read JSON files from a folder and return a list of dictionaries.
    Args:
        folder_path (str): Path to the folder containing JSON files.
    Returns:
        list: A list of dictionaries containing data from each JSON file.
    """
    json_data_list = []

    # Check if the folder path exists
    if not os.path.exists(folder_path):
        print(f"Folder '{folder_path}' does not exist.")
        return json_data_list

    # Iterate over files in the folder
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        # Check if the file is a JSON file
        if filename.endswith('.json'):
            with open(file_path, 'r') as f:
                # Load JSON data from the file
                try:
                    json_data = json.load(f)
                    json_data_list.append(json_data)
                except json.JSONDecodeError:
                    print(f"Error reading JSON from file: {file_path}")
                    continue

    df = pd.DataFrame.from_dict(json_data_list)

    return df, json_data_list


df, json_data_list = read_json_folder('data/jsons')
df['full_content'] = df['title'] + ' ' + df['content']
df = df.drop(['topic', 'source', 'url', 'date', 'authors','title', 'content',
              'content_original', 'source_url', 'bias_text','ID'], axis=1)


#------------------------------------------------------------------------------------
#
# Text preprocessing and data loading
#
#------------------------------------------------------------------------------------
class TextDataset(Dataset):
    def __init__(self, df, text_col, label_col, max_doc_length, tokenizer, word2vec_model):
        self.df = df
        self.text_col = text_col
        self.label_col = label_col
        self.max_doc_length = max_doc_length
        self.tokenizer = tokenizer
        self.word2vec_model = word2vec_model

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        text = self.df[self.text_col].iloc[index]
        label = self.df[self.label_col].iloc[index]

        tokens = self.tokenizer(text)

        embeddings = []
        for token in tokens:
            if token in self.word2vec_model.key_to_index:
                embedding = self.word2vec_model[token]
                embedding = torch.from_numpy(embedding)
            else:
                embedding = torch.zeros(self.word2vec_model.vector_size)
            embeddings.append(embedding)

        # Pad or truncate the embeddings to the desired length
        embeddings = embeddings[:self.max_doc_length]
        embeddings = embeddings + [torch.zeros(self.word2vec_model.vector_size)] * (
                    self.max_doc_length - len(embeddings))

        # Stack the embeddings into a tensor
        text_tensor = torch.stack(embeddings)

        # Convert label to tensor
        label_tensor = torch.tensor(label, dtype=torch.long)

        return text_tensor, label_tensor

def tokenizer(text):
    no_stop = remove_stopwords(text)
    tokenizer = get_tokenizer("basic_english")
    tokens = tokenizer(no_stop)
    return tokens

word2vec_model = api.load("word2vec-google-news-300")

text_column = 'full_content'
label_column = 'bias'
max_doc_length = 500
batch_size = 32

print('Loading and tokenizing data...')
dataset = TextDataset(df, text_column, label_column, max_doc_length, tokenizer, word2vec_model)
train_data, temp_data = train_test_split(dataset, test_size=0.2, random_state=42)
val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)

train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

#------------------------------------------------------------------------------------
#
# Build CNN model
#
#------------------------------------------------------------------------------------
print('Building model...')
class CNN(nn.Module):
    def __init__(self, embedding_dim, num_filters, filter_sizes, num_classes, dropout):
        super(CNN, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_filters = num_filters
        self.filter_sizes = filter_sizes
        self.num_classes = num_classes
        self.dropout = dropout

        self.convs = nn.ModuleList([
            nn.Conv2d(1, num_filters,(fsz,embedding_dim)) for fsz in filter_sizes
        ])
        self.fc = nn.Linear(len(filter_sizes) * num_filters, num_classes)


    def forward(self, x):
        x = x.unsqueeze(1)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]
        x = torch.cat(x, 1)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.fc(x)

        return x



embedding_dim = word2vec_model.vector_size
num_filters = 100
filter_sizes = [3, 4, 5]
num_classes = 3
dropout = 0.5
epochs = 7
lr = 0.001

print("Building model...")
model = CNN(embedding_dim, num_filters, filter_sizes, num_classes, dropout)
model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=lr)


#------------------------------------------------------------------------------------
#
# Train model
#
#------------------------------------------------------------------------------------

print("Training model...")
for epoch in range(epochs):
    model.train()
    running_loss = 0.0

    for batch_data in train_dataloader:
        inputs, labels = batch_data
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    epoch_loss = running_loss / len(train_dataloader)
    print(f"Epoch [{epoch + 1}/{epochs}], Loss: {epoch_loss:.4f}")

#------------------------------------------------------------------------------------
#
# Evaluate model
#
#------------------------------------------------------------------------------------

    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_data in test_dataloader:
            inputs, labels = batch_data
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # Print the accuracy for the testing set
    accuracy = correct / total
    print(f"Testing Accuracy: {accuracy:.4f}")



print('Testing model...')
model.eval()
correct = 0
total = 0
test_true_labels = []
test_predicted_labels = []

with torch.no_grad():
    for batch_data in test_dataloader:
        inputs, labels = batch_data
        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        test_true_labels.extend(labels.cpu().numpy())
        test_predicted_labels.extend(predicted.cpu().numpy())


# Calculate the test accuracy
test_accuracy = correct / total
print(f"Test Accuracy: {test_accuracy:.4f}")
f1 = f1_score(test_true_labels, test_predicted_labels, average='weighted')
print(f"F1 Score: {f1:.4f}")

metrics_data = [
    ['Accuracy', test_accuracy],
    ['F1 Score', f1]
]

with open('CNN_metrics.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Metric', 'Value'])
    writer.writerows(metrics_data)


# Calculate the confusion matrix
cm = confusion_matrix(test_true_labels, test_predicted_labels)

print("Confusion Matrix:")
plt.figure(figsize=(8, 6))
plt.imshow(cm, cmap='Blues')
plt.title('Confusion Matrix')
plt.colorbar()
plt.xticks(range(num_classes), range(num_classes))
plt.yticks(range(num_classes), range(num_classes))
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.tight_layout()
plt.savefig('CNN_confusion_matrix.png')
plt.show()
