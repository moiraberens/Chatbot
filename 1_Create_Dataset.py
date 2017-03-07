import gzip
import xml.etree.ElementTree as ET
import pickle
import copy
from glob import glob
from tqdm import tqdm

DIGIT_STRINGS = ["1","2","3","4","5","6","7","8","9"]
START_VOCAB = ["_PAD", "_GO", "_EOS", "_UNK"]
BUCKETS = [5,10,15,20]

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

def create_vocab(vocab_count, max_vocab_len):
    vocab_token2text = START_VOCAB + sorted(vocab_count, key=vocab_count.get, reverse=True)
    if len(vocab_token2text) > max_vocab_len:
        vocab_token2text = vocab_token2text[:max_vocab_len]
    vocab_text2token = dict(zip(vocab_token2text,range(len(vocab_token2text))))
    return vocab_text2token, vocab_token2text

def pad_input(vocab,sentence,bucket):
    sentence += [vocab["_PAD"]] * (bucket - len(sentence))
    sentence = list(reversed(sentence))
    return sentence

def pad_output(vocab,sentence,bucket):
    sentence = [vocab["_GO"]] + sentence + [vocab["_EOS"]] + [vocab["_PAD"]] * (bucket - len(sentence))
    return sentence 

if __name__ == "__main__":
    
    movie_folders = glob("Data/*/*/*/*/*")[:1000]
    
    # Loading and Preprocessing Text Files
    # And Create Vocabulary
    print("Loading and Preprocessing XML Files to Create Vocabulary")
    vocab_count = {}
    for movie_folder in tqdm(movie_folders):
        file_path = glob(movie_folder+'/*')[0]
        text_raw = extract_text(file_path)
        text_lower = lowercase(text_raw)
        text = normalize_digits(text_lower)
        for sentence in text:
            for word in sentence:
                if word in vocab_count:
                    vocab_count[word] += 1
                else:
                    vocab_count[word] = 1
    vocab_text2token, vocab_token2text = create_vocab(vocab_count, 50000)
    save(vocab_text2token,"Data/vocab_text2token.pkl.gz")
    save(vocab_token2text,"Data/vocab_token2text.pkl.gz")
    
    # Loading and Preprocessing Text Files
    # And Create DataSet (of Tokens)
    print("\nLoading and Preprocessing XML Files to Create Dataset")
    dataset = []
    for _ in range(len(BUCKETS)):
        dataset.append([])
    for movie_folder in tqdm(movie_folders):
        file_path = glob(movie_folder+'/*')[0]
        text_raw = extract_text(file_path)
        text_lower = lowercase(text_raw)
        text = normalize_digits(text_lower)
        text_token = copy.deepcopy(text)
        for sentence_idx in range(len(text)):
            sentence = text[sentence_idx]
            for word_idx in range(len(sentence)):
                word = sentence[word_idx]
                if word in vocab_text2token:
                    text_token[sentence_idx][word_idx] = vocab_text2token[word]
                else:
                    text_token[sentence_idx][word_idx] = vocab_text2token["_UNK"]
        input_sentence = []
        for output_sentence in text_token:
            if len(input_sentence) > 0:
                for bucket_idx in range(len(BUCKETS)):
                    if len(input_sentence) <= BUCKETS[bucket_idx] and len(output_sentence) <= BUCKETS[bucket_idx]:
                        dataset[bucket_idx].append([pad_input(vocab_text2token,input_sentence,BUCKETS[bucket_idx]), pad_output(vocab_text2token,output_sentence,BUCKETS[bucket_idx])])
            input_sentence = output_sentence
    save(dataset,"Data/dataset.pkl.gz")