import gzip
import pickle

BUCKETS = [5,10,15,20]

def load(path):
    with gzip.open(path,"r") as f:
        item = pickle.load(f)
    return item

if __name__ == "__main__":
    # Load Dataset
    dataset = load("Data/dataset.pkl.gz")
    # Usage: Dataset consists of buckets(0,1,2,3, see BUCKETS), consisting of examples(x amount), consisting of input(0)/output(1), consisting of words(x amount).
    # Example: get from bucket 15, second example, input sentence, 11th word (you can check manually)
    print("dataset[2][1][0][10]    =",dataset[2][1][0][10])
    
    # Load Text2Token
    vocab_text2token = load("Data/vocab_text2token.pkl.gz")
    # Example: Get token for "example"
    print("Token for text: example =", vocab_text2token["example"])
    
    # Load Token2Text
    vocab_token2text = load("Data/vocab_token2text.pkl.gz")
    # Example: Inverse of previous example
    print("Text for token: 1655    =", vocab_token2text[1655])