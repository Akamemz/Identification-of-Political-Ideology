import time
import os
import csv
import nltk
import numpy as np
import pandas as pd
import json
import torch
import torch.nn as nn
from nltk.corpus import stopwords
from collections import Counter
import re
import gensim.downloader as api
from torch.utils.data import TensorDataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence


#nltk.download('wordnet')
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import cohen_kappa_score, confusion_matrix
from sklearn.metrics import accuracy_score, f1_score

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
import torch.nn.functional as F
import nltk
from nltk.tokenize.casual import casual_tokenize
from nltk.stem import WordNetLemmatizer


import re

is_cuda = torch.cuda.is_available()
if is_cuda:
    device = torch.device("cuda")
    print("GPU is available")
else:
    device = torch.device("cpu")
    print("GPU not available, CPU used")

#%%

# ******************
#     Read data
# ******************
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

df, json_data_list = read_json_folder('../data/jsons')

X,y = df['content'].values,df['bias'].values
x_train, x_test, y_train, y_test = train_test_split(X,y, test_size=0.2, stratify=y)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.25, stratify=y_train)
print(f'shape of train data is {x_train.shape}')
print(f'shape of validation data is {x_val.shape}')
print(f'Shape of test data is {x_test.shape}')


unique_classes = df['bias'].unique()
num_classes = len(unique_classes)
print("Unique classes:", unique_classes)
print("Number of classes:", num_classes)
#**********
#preprocessing
#*************



#nltk.download('stopwords')
stop_words = nltk.corpus.stopwords.words('english')

def preprocess_string(s):
    s = re.sub(r"[^\w\s]", '', s)
    s = re.sub(r"\s+", '', s)
    return s


def tockenize(x_train, y_train, x_val, y_val, x_test, y_test):
    word_list = []
    stop_words = set(stopwords.words('english'))
    for sent in x_train:
        for word in sent.lower().split():
            word = preprocess_string(word)
            if word not in stop_words and word != '':
                word_list.append(word)

    corpus = Counter(word_list)
    corpus_ = sorted(corpus, key=corpus.get, reverse=True)[:1000]
    onehot_dict = {w: i + 1 for i, w in enumerate(corpus_)}

    final_list_train, final_list_val, final_list_test = [], [], []
    for sent in x_train:
        final_list_train.append([onehot_dict[preprocess_string(word)] for word in sent.lower().split()
                                 if preprocess_string(word) in onehot_dict.keys()])
    for sent in x_val:
        final_list_val.append([onehot_dict[preprocess_string(word)] for word in sent.lower().split()
                                if preprocess_string(word) in onehot_dict.keys()])
    for sent in x_test:
        final_list_test.append([onehot_dict[preprocess_string(word)] for word in sent.lower().split()
                                if preprocess_string(word) in onehot_dict.keys()])

    encoded_train = y_train
    encoded_val = y_val
    encoded_test = y_test
    return final_list_train, encoded_train, final_list_test, encoded_test, final_list_val, encoded_val, onehot_dict



x_train,y_train,x_val, y_val, x_test,y_test,vocab = tockenize(x_train,y_train, x_val, y_val, x_test,y_test)
print("pre_process and tokenization complete")

### added pad_pack_sequence to the LSTM model
def preprocess_data(texts, labels, vocab, max_seq_len=512):
    # Convert words to indices using the vocabulary
    indexed_texts = []
    for text in texts:
        indexed_text = [vocab[word] for word in text if word in vocab]
        indexed_texts.append(torch.tensor(indexed_text, dtype=torch.long))

    # Pad and pack sequences
    padded_texts = pad_sequence(indexed_texts, batch_first=True, padding_value=0)
    seq_lengths = torch.tensor([min(len(text), max_seq_len) for text in indexed_texts])
    packed_texts = pack_padded_sequence(padded_texts, seq_lengths, batch_first=True, enforce_sorted=False)

    # Convert labels to tensor
    label_tensor = torch.tensor(labels, dtype=torch.long)

    return packed_texts, padded_texts, seq_lengths, label_tensor
print("padding complete")


# Create vocabulary
def create_vocabulary(texts):
    word_counts = {}
    for text in texts:
        for word in text:
            if word not in word_counts:
                word_counts[word] = 0
            word_counts[word] += 1

    vocab = {word: idx + 1 for idx, word in enumerate(sorted(word_counts, key=word_counts.get, reverse=True))}
    vocab['<pad>'] = 0  # Add padding token to vocabulary

    return vocab


# Create embedding layer
def create_embedding_layer(vocab_size, embedding_dim):
    embedding_layer = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
    return embedding_layer
print("embedding complete")

# Create vocabulary from training texts
vocab = create_vocabulary(x_train)


x_train_pad, x_train_padded, train_seq_lengths, y_train_tensor = preprocess_data(x_train, y_train, vocab, max_seq_len=512)
x_val_pad, x_val_padded, val_seq_lengths, y_val_tensor = preprocess_data(x_val, y_val, vocab, max_seq_len=512)
x_test_pad, x_test_padded, test_seq_lengths, y_test_tensor = preprocess_data(x_test, y_test, vocab, max_seq_len=512

                                                                             )


# unpacked the pad sequences
x_train_padded, train_seq_lengths = pad_packed_sequence(x_train_pad, batch_first=True)
x_val_padded, val_seq_lengths = pad_packed_sequence(x_val_pad, batch_first=True)
x_test_padded, test_seq_lengths = pad_packed_sequence(x_test_pad, batch_first=True)

train_data = TensorDataset(x_train_padded.long(), torch.tensor(y_train, dtype=torch.long))
valid_data = TensorDataset(x_val_padded.long(), torch.tensor(y_val, dtype=torch.long))
test_data = TensorDataset(x_test_padded.long(), torch.tensor(y_test, dtype=torch.long))

batch_size = 64

train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size, drop_last=True)
valid_loader = DataLoader(valid_data, shuffle=True, batch_size=batch_size, drop_last=True)
test_loader = DataLoader(test_data, shuffle=True, batch_size=batch_size, drop_last=True)

class ClassificationLSTM(nn.Module):
    def __init__(self, no_layers, vocab_size, hidden_dim, embedding_dim, output_dim, drop_prob=0.5):
        super(ClassificationLSTM, self).__init__()
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.no_layers = no_layers
        self.vocab_size = vocab_size
        self.device = device
        # Embedding layer
        self.embedding = create_embedding_layer(vocab_size, embedding_dim)

        # LSTM layer
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=self.hidden_dim,
                            num_layers=no_layers, batch_first=True, dropout=drop_prob)

        # Dropout layer
        self.dropout = nn.Dropout(drop_prob)

        # Fully connected layers
        self.fc1 = nn.Linear(self.hidden_dim, 128)
        self.fc2 = nn.Linear(128, output_dim)

        # Activation function
        self.activation = nn.ReLU()


    def forward(self, x_padded, hidden):
        # unpack the input sequences
        #x, seq_lengths = pad_packed_sequence(x_padded, batch_first=True)

        #hidden = (hidden[0].to(x.device), hidden[1].to(x.device))
        # Embed the input

        embedded = self.embedding(x_padded)
        # pack the embedded sequences
       # packed_embedded = pack_padded_sequence(embedded, seq_lengths, batch_first=True, enforce_sorted=False)

        #pass the packed embedded sequence through the LSTM layer
        # Pass the embedded input through the LSTM layer
        lstm_output, hidden = self.lstm(embedded, hidden)

        #lstm_output, _ = pad_packed_sequence(lstm_output, batch_first=True)

        # Apply dropout to the LSTM output
        lstm_output = self.dropout(lstm_output)

        # Take the last hidden state of the LSTM
        last_hidden_state = lstm_output[:, -1, :]

        # Pass the last hidden state through the fully connected layers
        fc_output = self.activation(self.fc1(last_hidden_state))
        fc_output = self.fc2(fc_output)
        #fc_output = self.activation(fc_output)

        return fc_output, hidden

    def init_hidden(self, batch_size):
        # Initialize the hidden state and cell state of the LSTM
        hidden = (torch.zeros(self.no_layers, batch_size, self.hidden_dim).to(self.device),
                  torch.zeros(self.no_layers, batch_size, self.hidden_dim).to(self.device))
        return hidden

# Define the parameters
no_layers = 3
vocab_size = len(vocab) + 1
embedding_dim = 100
output_dim = 3
hidden_dim = 256
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Initialize the model
model = ClassificationLSTM(no_layers, vocab_size, hidden_dim, embedding_dim, output_dim)
model.to(device)

# Set the criterion (loss function)
criterion = nn.CrossEntropyLoss()

# Set the optimizer
learning_rate = 2e-5
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

#*******
#Model training
#********

model.train()

# Training loop
num_epochs = 40
for epoch in range(num_epochs):
    model.train()
    epoch_train_loss = 0
    epoch_train_acc = 0
    for batch_idx, (batch_inputs, batch_labels) in enumerate(train_loader):
        # move the batch inputs and labels to the device
        batch_inputs = batch_inputs.to(device)
        batch_labels = batch_labels.to(device)

        # initialize the hidden state of each batch
        hidden = model.init_hidden(batch_inputs.size(0))
        hidden = (hidden[0].to(device), hidden[1].to(device))

        # Forward pass
        outputs, _ = model(batch_inputs, hidden)
        loss = criterion(outputs, batch_labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_train_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        epoch_train_acc += (predicted == batch_labels).sum().item()

    epoch_train_loss /= len(train_loader)
    epoch_train_acc = epoch_train_acc / len(train_data)*100
        # Print the loss for every 100 batches
       # if (batch_idx + 1) % 100 == 0:
           # print(
               # f"Epoch [{epoch + 1}/{num_epochs}], Batch [{batch_idx + 1}/{len(train_loader)}], Loss: {loss.item():.4f}")

    # Evaluate the model on the validation set
    #***************
    model.eval()
    #**************
    epoch_val_loss = 0
    epoch_val_acc = 0

    with torch.no_grad():
        for batch_inputs, batch_labels in valid_loader:
            batch_inputs = batch_inputs.to(device)
            batch_labels = batch_labels.to(device)

            #Initialize the hidden state for each batch
            hidden = model.init_hidden(batch_inputs.size(0))
            hidden = (hidden[0].to(device), hidden[1].to(device))


            outputs, _ = model(batch_inputs, hidden)
            loss = criterion(outputs, batch_labels)
            epoch_val_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            epoch_val_acc += (predicted == batch_labels).sum().item()

    epoch_val_loss /= len(valid_loader)
    epoch_val_acc = epoch_val_acc / len(valid_data)*100

    print(f'Epoch {epoch + 1}')
    print(f'train_loss: {epoch_train_loss:.4f} val_loss: {epoch_val_loss:.4f}')
    print(f'train_accuracy: {epoch_train_acc:.2f}% val_accuracy: {epoch_val_acc:.2f}%')
    print()

    #print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

    # Save the trained model
    #torch.save(model.state_dict(), "trained_model.pth")

#*** Testing the LSTM model
model.eval()
test_loss = 0
test_acc = 0
test_predictions = []
test_labels = []

with torch.no_grad():
    for batch_inputs, batch_labels in test_loader:
        batch_inputs = batch_inputs.to(device)
        batch_labels = batch_labels.to(device)

        # Initialize the hidden state for each batch
        hidden = model.init_hidden(batch_inputs.size(0))
        hidden = (hidden[0].to(device), hidden[1].to(device))

        outputs, _ = model(batch_inputs, hidden)
        loss = criterion(outputs, batch_labels)
        test_loss += loss.item()

        _, predicted = torch.max(outputs, 1)
        test_acc += (predicted == batch_labels).sum().item()

        test_predictions.extend(predicted.cpu().numpy())
        test_labels.extend(batch_labels.cpu().numpy())

test_loss /= len(test_loader)
test_acc *= 100 / len(test_data)

print(f'Test Loss: {test_loss:.4f}')
print(f'Test Accuracy: {test_acc:.2f}%')

# Calculate the weighted F1-score
from sklearn.metrics import f1_score

f1 = f1_score(test_labels, test_predictions, average='weighted')
print(f'Weighted F1-score: {f1:.4f}')

# Create a DataFrame with the first 5 lines of initial and predicted values
import pandas as pd

data = {'Initial': test_labels[:5], 'Predicted': test_predictions[:5]}
df = pd.DataFrame(data)
print('First 5 lines of initial and predicted values:')
print(df)

# Plot the confusion matrix
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

num_unique_classes = len(set(test_labels))  # Get the number of unique classes

cm = confusion_matrix(test_labels, test_predictions)
print("Confusion Matrix:")
plt.figure(figsize=(8, 6))
plt.imshow(cm, cmap='Blues')
plt.title('Confusion Matrix')
plt.colorbar()
plt.xticks(range(num_unique_classes), range(num_unique_classes))
plt.yticks(range(num_unique_classes), range(num_unique_classes))
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.tight_layout()
plt.savefig('LSTM_confusion_matrix.png')
plt.show()