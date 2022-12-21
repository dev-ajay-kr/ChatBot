import json
from preprocessor import tokenize, stem, bag_of_words
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from model import NeuralNet

# Load intents from the intents.json file
with open('intents.json', 'r') as f:
    data = json.load(f)
    intents = data['intents']

# Create lists to store the tags and words used in the intents
tags = []
all_words = []
xy = []

# Iterate through each intent
for intent in intents:
    # Get the tag for the intent
    tag = intent['tag']
    # Add the tag to the list of tags
    tags.append(tag)

    # Iterate through each pattern in the intent
    for pattern in intent['patterns']:
        # Tokenize the pattern to get a list of words
        words = tokenize(pattern)
        # Add the words to the list of all words
        all_words.extend(words)
        # Add the words and the tag to the list of training data
        xy.append((words, tag))

# Remove punctuation from the list of all words
ignore_words = ['?', '!', '.']
all_words = [stem(word) for word in all_words if word not in ignore_words]
# Sort the list of all words and tags alphabetically
all_words = sorted(set(all_words))
tags = sorted(set(tags))

# Create lists to store the input and output data for the model
X_train = []
Y_train = []

# Iterate through the training data
for words, tag in xy:
    # Get a bag of words representation of the words
    bag = bag_of_words(words, all_words)
    # Add the bag of words to the list of input data
    X_train.append(bag)
    # Get the index of the tag and add it to the list of output data
    label = tags.index(tag)
    Y_train.append(label)

# Convert the input and output data to numpy arrays
X_train = np.array(X_train)
Y_train = np.array(Y_train)

# Define a custom dataset class for the chat data
class ChatDataSet(Dataset):
    def __init__(self):
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = Y_train

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.n_samples

# Define the hyperparameters for the model
epochs = 1000
batch_size = 8
input_size = len(X_train[0])
hidden_size = 8
output_size = len(tags)
learning_rate = 0.001

# Create a dataset object from the ChatDataSet class
dataset = ChatDataSet()

# Create a dataloader to load the data in batchesdataset = ChatDataSet()

train_loader = DataLoader(dataset=dataset,
                         batch_size=batch_size,
                         shuffle=True,
                         num_workers=0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = NeuralNet(input_size, hidden_size, output_size).to(device)

criteria = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(epochs):
    for bagofwords, label in train_loader:
        bagofwords = bagofwords.to(device)
        label = label.to(dtype=torch.long).to(device)

        output = model(bagofwords)
        
        loss = criteria(output, label)

        optimizer.zero_grad()

        loss.backward()
        optimizer.step()

    if (epoch +1)%100==0:
        print(f'epoch: {epoch+1}/{epochs}, loss: {loss.item():.4f}')

print(f'final loss: {loss.item():.4f}')
