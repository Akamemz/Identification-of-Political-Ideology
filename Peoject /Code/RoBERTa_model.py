# ******************************************
#               Load Libraries
# ******************************************
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from transformers import RobertaTokenizer, RobertaForSequenceClassification, get_scheduler
from torch.utils.data import DataLoader
import torch
import os
import json
import pickle
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split
from datasets import Dataset


# ******************************************
#         Check for GPU availability
# ******************************************
is_cuda = torch.cuda.is_available()
if is_cuda:
    device = torch.device("cuda")
    print("GPU is available")
else:
    device = torch.device("cpu")
    print("GPU not available, CPU used")


#%%
# ******************************************
# Load JSON Data and transform into pd.DataFrame
# ******************************************
def read_json_folder(folder_path):
    if not folder_path or not os.path.exists(folder_path):
        print("Invalid folder path.")
        return None

    json_data_list = []
    for filename in os.listdir(folder_path):
        file_paths = os.path.join(folder_path, filename)
        if filename.endswith('.json'):
            with open(file_paths, 'r', encoding='utf-8') as f:
                try:
                    json_data = json.load(f)
                    json_data_list.append(json_data)
                except json.JSONDecodeError:
                    print(f"Error reading JSON from file: {file_paths}")
                    continue

    if not json_data_list:
        print(f"No valid JSON files found in folder: {folder_path}")
        return None

    data_frame = pd.DataFrame(json_data_list)
    return data_frame


# Read JSON files and preprocess data
df = read_json_folder('Group Project/data/jsons')


# Drop unnecessary columns
df = df.drop(columns=['topic', 'source', 'url', 'date', 'authors',
                      'content_original', 'source_url', 'bias_text', 'ID'], axis=1)

df['content'] = df['title'] + ' ' + df['content']
df = df.drop(columns=['title'])


# ******************************************
#               Split Data set
# ******************************************
# Load data processing functions
X = df['content']
y = df['bias']

X = pd.DataFrame(X, columns=['content'])
y = pd.DataFrame(y, columns=['bias'])

# Split the data into training, validation, and test sets
X_train, X_val_test, y_train, y_val_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_val_test, y_val_test, test_size=0.5, random_state=42)

#%%
# ******************************************
#               Model Fine-Tuning
# ******************************************
num_epochs = 7
max_length = 512

checkpoint = "roberta-base"
tokenizer = RobertaTokenizer.from_pretrained(checkpoint)


# Tokenize and encode data
def encode_data(data, tokenizer, max_length):
    return tokenizer(data['content'], padding='max_length', truncation=True, return_tensors='pt',
                     max_length=max_length, return_attention_mask=True,)


# Encode training, validation, and test datasets
train_dataset = Dataset.from_pandas(X_train).map(lambda x: encode_data(x, tokenizer, max_length=max_length), batched=True)
val_dataset = Dataset.from_pandas(X_val).map(lambda x: encode_data(x, tokenizer, max_length=max_length), batched=True)
test_dataset = Dataset.from_pandas(X_test).map(lambda x: encode_data(x, tokenizer, max_length=max_length), batched=True)

# Add labels column to the datasets
train_dataset = train_dataset.add_column('labels', y_train[y_train.columns[0]])
val_dataset = val_dataset.add_column('labels', y_val[y_val.columns[0]])
test_dataset = test_dataset.add_column('labels', y_test[y_test.columns[0]])

# Set format for training, validation, and test datasets
train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
val_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

batch_size = 32
# Create PyTorch dataloaders
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
val_dataloader = DataLoader(val_dataset, shuffle=False, batch_size=batch_size)
test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size)

# Model initialization
model = RobertaForSequenceClassification.from_pretrained(checkpoint, num_labels=3)  # 3 classes for classification

# Optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5, eps=1e-5)

# Device setup
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)

# Calculate the number of training steps per epoch
num_training_steps_per_epoch = len(train_dataloader)
num_training_steps = num_epochs * num_training_steps_per_epoch


# Define the number of warmup steps
num_warmup_steps = int(0.5 * num_training_steps)  # You can adjust the warmup proportion as needed

# Create the scheduler
scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=num_warmup_steps,
    num_training_steps=num_training_steps
)


# Training loop
progress_bar = tqdm(range(num_training_steps))
model.train()
for epoch in range(num_epochs):
    epoch_losses = []
    for batch in train_dataloader:
        batch_encoded = {k: v.to(device) for k, v in batch.items() if k != 'labels'}
        labels = batch['labels'].to(device)

        outputs = model(**batch_encoded, labels=labels)
        loss = outputs.loss
        loss.backward()

        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        epoch_losses.append(loss.item())

        progress_bar.update(1)

    # Calculate average loss for this epoch
    avg_train_loss = sum(epoch_losses) / len(epoch_losses)

    # Evaluation
    model.eval()
    val_losses = []
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in val_dataloader:
            batch_encoded = {k: v.to(device) for k, v in batch.items() if k != 'labels'}
            labels = batch['labels'].to(device)

            outputs = model(**batch_encoded, labels=labels)
            loss = outputs.loss
            val_losses.append(loss.item())

            _, predicted = torch.max(outputs.logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_val_loss = sum(val_losses) / len(val_losses)
    val_acc = correct / total

    # Print epoch-wise training and validation loss, and validation accuracy
    print(f'Epoch: {epoch + 1:02} | Train Loss: {avg_train_loss:.3f} | Val Loss: {avg_val_loss:.3f} | Val Acc: {val_acc * 100:.2f}%')

    model.train()


#%%
# ******************************************
#    Model Evaluation on Validation set
# ******************************************
# Lists to store predictions and true labels
all_predictions = []
all_labels = []

# Evaluation loop
model.eval()
with torch.no_grad():
    for batch in val_dataloader:
        batch_encoded = {k: v.to(device) for k, v in batch.items() if k != 'labels'}
        labels = batch['labels'].to(device)

        outputs = model(**batch_encoded, labels=labels)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)

        all_predictions.extend(predictions.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Calculate evaluation metrics
accuracy = accuracy_score(all_labels, all_predictions)
precision = precision_score(all_labels, all_predictions, average='weighted')
recall = recall_score(all_labels, all_predictions, average='weighted')
f1 = f1_score(all_labels, all_predictions, average='weighted')

# Print the evaluation metrics and confusion matrix
print("Validation Set Evaluation:")
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)


# ******************************************
#       Model Evaluation on Test Set
# ******************************************
# Lists to store predictions and true labels
test_predictions = []
test_labels = []

# Evaluation loop
model.eval()
with torch.no_grad():
    for batch in test_dataloader:
        batch_encoded = {k: v.to(device) for k, v in batch.items() if k != 'labels'}
        labels = batch['labels'].to(device)

        outputs = model(**batch_encoded, labels=labels)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)

        test_predictions.extend(predictions.cpu().numpy())
        test_labels.extend(labels.cpu().numpy())

# Calculate evaluation metrics on test set
test_accuracy = accuracy_score(test_labels, test_predictions)
test_precision = precision_score(test_labels, test_predictions, average='weighted')
test_recall = recall_score(test_labels, test_predictions, average='weighted')
test_f1 = f1_score(test_labels, test_predictions, average='weighted')

# Print the evaluation metrics on test set
print("Test Set Evaluation:")
print("Accuracy:", test_accuracy)
print("Precision:", test_precision)
print("Recall:", test_recall)
print("F1 Score:", test_f1)


#%%
# ******************************************
#        Plot & Save Confusion matrix
# ******************************************
conf_matrix = confusion_matrix(all_labels, all_predictions)
print("Confusion Matrix:")
print(conf_matrix)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
# plt.savefig('confusion_matrix_RoBERTa.png') <--- Uncomment this section to save matrix in .png format
plt.show()


#%%
# ******************************************
#      Save trained model as .pkl file
# ******************************************
# Specify the directory where you want to save the file
save_directory = "....................."

# Ensure that the directory exists, create it if it doesn't
# os.makedirs(save_directory, exist_ok=True)

# Save the final model after training completion
file_path = os.path.join(save_directory, "model.pth")
torch.save(model.state_dict(), file_path)


#%%
# Save the final model after training completion
file_path = os.path.join(save_directory, "model2345.pkl")
with open(file_path, 'wb') as file:
    pickle.dump(model, file)


