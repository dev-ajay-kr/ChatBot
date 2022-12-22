import torch
from model import NeuralNet
from preprocessor import tokenize, bag_of_words
import json
import random

with open("intents.json", 'r') as f:
    intents_data = json.load(f)
    intents = intents_data['intents']

FILE = "model_data.pth"
model_data = torch.load(FILE)

print(model_data)

model_state = model_data['model_state']
input_size = model_data['input_size']
hidden_size = model_data['hidden_size']
output_size = model_data['output_size']
all_words = model_data['all_words']
tags = model_data['tags']

print(input_size)
print(hidden_size)
print(output_size)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

def response(userInput):
    sentence = tokenize(userInput)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)
    out = model(X)
    _, predicted = torch.max(out, dim=1)
    tag = tags[predicted.item()]
    probs = torch.softmax(out, dim=1)
    prob = probs[0][predicted.item()]
    if prob.item() > 0.75:
        for intent in intents:
            if tag == intent['tag']:
                return random.choice(intent['responses'])
    return "I do not understand, kindly rephrase"

if __name__ == "__main__":
    print("Lest start chatting! type (quit) to exit the chat")
    while True:
        userInput = input("User: ")
        if userInput == "quit":
            break
        print(response(userInput))
