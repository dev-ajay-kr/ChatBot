import nltk
from nltk.stem.porter import PorterStemmer
import numpy as np

stemmer = PorterStemmer()

def tokenize(sentence):
    return nltk.word_tokenize(sentence)

def stem(word):
    return stemmer.stem(word)

def bag_of_words(tokenized_sentence, all_words):
    words = [stem(w) for w in tokenized_sentence]

    bag = np.zeros(len(all_words), dtype=np.float32)

    for idx, w in enumerate(all_words):
        if w in words:
            bag[idx] = 1

    return bag

# ['you', 'if', 'how', 'are', 'hello']

# [5] => [1, 0, 1, 1, 0]

# ['how are you'] => ['you', 'are', 'how', 'organ']

# sent = "Hey there how are you doing?"

# print(tokenize(sent))
