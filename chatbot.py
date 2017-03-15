import gzip
import pickle
import seq2seq_model_adapted
import tensorflow as tf
import random
import time
import sys
import copy
import numpy as np


## ------------------ ##
## Settings for model ##
## ------------------ ##

buckets = [(5,7), (10,12), (15,17), (20,22)]

settings = {}
settings['vocab_size'] = 0
settings['buckets'] = len(buckets)
settings['num_units_per_layer'] = 512
settings['num_layers'] = 2
settings['max_gredient_norm'] = 5.0
settings['batch_size'] = 128
settings['learning_rate'] = 0.5
settings['learning_rate_decay'] = 1
settings['num_samples'] = 512
settings['num_epochs'] = 1000
settings['num_steps_per_epoch'] = 100

DIGIT_STRINGS = ["1","2","3","4","5","6","7","8","9"]
LETTER_STRING = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"

## ------------------ ##
##     functions      ##
## ------------------ ##

def load(path):
    with gzip.open(path,"r") as f:
        item = pickle.load(f)
    return item

def create_model(session, forward_only):
    model = seq2seq_model_adapted.Seq2SeqModel(
        settings['vocab_size'],
        settings['vocab_size'],
        buckets,
        settings['num_units_per_layer'],
        settings['num_layers'],
        settings['max_gredient_norm'],
        settings['batch_size'],
        settings['learning_rate'],
        settings['learning_rate_decay'],
        use_lstm=True,
        forward_only = forward_only)
    session.run(tf.global_variables_initializer())
    return model
            
def train_chatbot(dataset_train, dataset_test):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        print("Creating Model")
        model = create_model(sess,False)
        
        best_test_perp = 1000.0
        
        print("Start Training")
        for epoch in range(settings['num_epochs']):
            start_time = time.time()
            
            # Training Steps
            train_loss = 0.0
            for step in range(settings['num_steps_per_epoch']):
                bucket_id = random.choice(range(len(buckets)))
                encoder_inputs, decoder_inputs, target_weights = model.get_batch(settings, dataset_train, buckets, bucket_id)
                _, loss, a = model.step(sess, encoder_inputs, decoder_inputs, target_weights, bucket_id, False)
                train_loss += loss / settings['num_steps_per_epoch']
            # After training reduce learning rate
            sess.run(model.learning_rate_decay_op)
            train_perp = np.exp(train_loss)
            
            # Test Steps
            test_loss = 0.0
            for bucket_id in range(len(buckets)):
                for step in range(np.round(settings['num_steps_per_epoch']*0.1).astype(int)):
                    encoder_inputs, decoder_inputs, target_weights = model.get_batch(settings, dataset_test, buckets, bucket_id)
                    _, loss, a = model.step(sess, encoder_inputs, decoder_inputs, target_weights, bucket_id, True)
                    test_loss += loss / (len(buckets) * np.round(settings['num_steps_per_epoch']*0.1).astype(int))
            test_perp = np.exp(test_loss)
            if test_perp < best_test_perp:
                best_test_perp = test_perp
                model.saver.save(sess, "Model/model.ckpt")
                
            print("Epoch {} --- train_perp = {:.2f}, test_perp = {:.2f}, time = {:.2f}".format(epoch+1,train_perp,test_perp,time.time()-start_time))
        
        #model.saver.save(sess, "Model/model.ckpt")
        
def clean_and_tokenize(input_sentence, vocab_text2token):
    output_sentence = copy.deepcopy(input_sentence)
    for word_idx in range(len(input_sentence)):
        word = input_sentence[word_idx].lower()
        for digit_str in DIGIT_STRINGS:
            word = word.replace(digit_str,"0")
            if word in vocab_text2token:
                output_sentence[word_idx] = vocab_text2token[word]
            else:
                output_sentence[word_idx] = vocab_text2token["_UNK"]
    return output_sentence

def pad_input(vocab_text2token,sentence,bucket):
    sentence += [vocab_text2token["_PAD"]] * (bucket - len(sentence))
    sentence = list(reversed(sentence))
    return sentence
        
def decoder(vocab_text2token, vocab_token2text):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        print("Loading Model")
        model = create_model(sess,True)
        model.saver.restore(sess,"Model/model.ckpt")
        model.batch_size = 1
        
        print("----- Start of Conversation (Type STOP to stop) -----")
        sys.stdout.write("USER:")
        sentence = input()
        while sentence != 'STOP':
            for char in set(sentence):
                if (char not in LETTER_STRING) and (char not in DIGIT_STRINGS):
                    sentence = sentence.replace(char,' '+char)
            sentence = sentence.split()
            sentence = clean_and_tokenize(sentence, vocab_text2token)
            bucket_id = 100
            for i in reversed(range(len(buckets))):
                if len(sentence) <= buckets[i][0]:
                    bucket_id = i
            if bucket_id != 100:
                sentence = pad_input(vocab_text2token,sentence,buckets[bucket_id][0])
                encoder_inputs, decoder_inputs, target_weights = model.get_decoder_batch(sentence, buckets[bucket_id])
                _,_,output_logits = model.step(sess, encoder_inputs, decoder_inputs, target_weights, bucket_id, True)
                outputs = [int(np.argmax(logit, axis=1)) for logit in output_logits]
                if 2 in outputs:
                    outputs = outputs[:outputs.index(2)]
                output_sentence = ''
                for output in outputs:
                    output_sentence += vocab_token2text[output] + ' '
                print("\t\t\t\tCHATBOT:")
                print("\t\t\t\t"+output_sentence)
            else:
                print("ERROR: INPUT SENTENCE TO LONG")
            sys.stdout.write("USER:")
            sentence = input()
        
## ------------------ ##
##        Main        ##
## ------------------ ##

if __name__ == "__main__":
    # Load Dataset
    print("Loading Data")
    dataset_train = load("Data/dataset_train.pkl.gz")
    dataset_test = load("Data/dataset_test.pkl.gz")
    vocab_text2token = load("Data/vocab_text2token.pkl.gz")
    vocab_token2text = load("Data/vocab_token2text.pkl.gz")
    settings['vocab_size'] = len(vocab_text2token)
    
    train_chatbot(dataset_train, dataset_test)
    
    decoder(vocab_text2token, vocab_token2text)