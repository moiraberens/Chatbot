import gzip
import xml.etree.ElementTree as ET
import pickle
import copy
from glob import glob
from tqdm import tqdm

DIGIT_STRINGS = ["1","2","3","4","5","6","7","8","9"]
START_VOCAB = ["_PAD", "_GO", "_EOS", "_UNK"]

def save(item,path):
    with gzip.open(path,"w") as f:
        pickle.dump(item,f)
        
def load(path):
    with gzip.open(path,"r") as f:
        item = pickle.load(f)
    return item

def extract_text(file_path):
    text = []
    with gzip.open(file_path) as f:
        tree = ET.parse(f)
        sentence = []
        for node in tree.iter():
            if node.tag == "w":
                sentence.append(node.text)
            elif len(sentence) > 0:
                text.append(sentence)
                sentence = []
    return text

def lowercase(text):
    text_lower = copy.deepcopy(text)
    for sentence_idx in range(len(text)):
        sentence = text[sentence_idx]
        for word_idx in range(len(sentence)):
            word = sentence[word_idx]
            text_lower[sentence_idx][word_idx] = word.lower()
    return text_lower
    
def normalize_digits(text):
    text_zero = copy.deepcopy(text)
    for sentence_idx in range(len(text)):
        sentence = text[sentence_idx]
        for word_idx in range(len(sentence)):
            word = sentence[word_idx]
            for digit_str in DIGIT_STRINGS:
                word = word.replace(digit_str,"0")
            text_zero[sentence_idx][word_idx] = word
    return text_zero

def create_vocab(data, max_vocab_len):
    vocab = {}
    for text in tqdm(data):
        for sentence in text:
            for word in sentence:
                if word in vocab:
                    vocab[word] += 1
                else:
                    vocab[word] = 1
    vocab_list = START_VOCAB + sorted(vocab, key=vocab.get, reverse=True)
    if len(vocab_list) > max_vocab_len:
        vocab_list = vocab_list[:max_vocab_len]
    vocab = dict(zip(vocab_list,range(len(vocab_list))))
    return vocab

def data_to_tokens(data, vocab):
    data_token = copy.deepcopy(data)
    for text_idx in tqdm(range(len(data))):
        text = data[text_idx]
        for sentence_idx in range(len(text)):
            sentence = text[sentence_idx]
            for word_idx in range(len(sentence)):
                word = sentence[word_idx]
                if word in vocab:
                    data_token[text_idx][sentence_idx][word_idx] = vocab[word]
                else:
                    data_token[text_idx][sentence_idx][word_idx] = vocab["_UNK"]
    return data_token

def pad_input(sentence,bucket):
    sentence += [vocab["_PAD"]] * (bucket - len(sentence))
    sentence = list(reversed(sentence))
    return sentence

def pad_output(sentence,bucket):
    sentence = [vocab["_GO"]] + sentence + [vocab["_EOS"]] + [vocab["_PAD"]] * (bucket - len(sentence))
    return sentence

def create_dataset(data):
    bucket_5 = []
    bucket_10 = []
    bucket_15 = []
    bucket_20 = []
    for text in tqdm(data):
        input_sentence = []
        for output_sentence in text:
            if len(input_sentence) > 0:
                if len(input_sentence) <= 5 and len(output_sentence) <= 5:
                    bucket_5.append([pad_input(input_sentence,5), pad_output(output_sentence,5)])
                elif len(input_sentence) <= 10 and len(output_sentence) <= 10:
                    bucket_10.append([pad_input(input_sentence,10), pad_output(output_sentence,10)])
                elif len(input_sentence) <= 15 and len(output_sentence) <= 15:
                    bucket_15.append([pad_input(input_sentence,15), pad_output(output_sentence,15)])
                elif len(input_sentence) <= 20 and len(output_sentence) <= 20:
                    bucket_20.append([pad_input(input_sentence,20), pad_output(output_sentence,20)])
            input_sentence = output_sentence
    dataset = [bucket_5,bucket_10,bucket_15,bucket_20]
    return dataset      

if __name__ == "__main__":
    # Loading and Preprocessing Text Files
    print("Loading and Preprocessing XML Files")
    movie_folders = glob("Data/*/*/*/*/*")
    data_raw = []
    for movie_folder in tqdm(movie_folders):
        file_path = glob(movie_folder+'/*')[0]
        text_raw = extract_text(file_path)
        text_lower = lowercase(text_raw)
        text = normalize_digits(text_lower)
        data_raw.append(text)
    
    # Creating and Saving Vocabulary
    print("\nCreating Vocabulary")
    vocab = create_vocab(data_raw, 50000)
    save(vocab,"Data/vocab.pkl.gz")
    
    # Use Vocabulary to rewrite Data
    print("\nTokenizing Data")
    data_token = data_to_tokens(data_raw, vocab)
    
    # Create the Final Dataset (with vocab-tokens, input-output sentences together + padded & in buckets)
    print("\nCreating Dataset")
    dataset = create_dataset(data_token)
    print("\nSaving Dataset")
    save(dataset,"Data/dataset.pkl.gz")