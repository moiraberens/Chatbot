# Chatbot

This repository is our solution to the chatbot assignment as mentioned in CHATBOT_INSTRUCTIONS.txt.

# Data set

We used the Opensubtitles2016 english dataset which can be found here http://opus.lingfil.uu.se/download.php?f=OpenSubtitles2016/en.tar.gz

For our model we only used the first 1000 subtitle files (split 80-20 in train-test) and a vocabulary of 50.000

Preprocessing is done by tokenization, digit normalization, lowercase, reversal of input sentence, and padding to fit bucket size.

# Model

With our model we wanted to recreate the results found in the paper "A Neural Conversational Model" (https://arxiv.org/pdf/1506.05869.pdf).

We used a modified version of the seq2seq_model from a language translation tensorflow example which can be found here:  https://github.com/tensorflow/models/tree/master/tutorials/rnn/translate. 

The code for our model can be found in the file seq2seq_model_adapted.py. We changed the get_batch function in this file since our preprocessing steps were different than in the translate model and therefore the get_batch function did not work anymore

Beside seq2seq_model_adapted.py we made the files create_dataset.py and chatbot.py

In create_dataset.py we loaded the raw data, preprocessed it and saved it again.

In the file chatbot.py we create the model, trained the bot and a function is included to use the trained model for a conversation.


# Results

In the end our results are not that great, sadly. We have tried several things to improve the results. 

For example, different amount of layers per unit, different amount of layers, and different batch sizes. 

In the end all the models did reach a perplexity around 30, but the results are not that great. 
 
The bot gives mostly very short answers, it repeats a lot of answers and it uses a lot of punctuation marks.
 
Moreover, the bot has some problems with punctuation marks in the question. If you ask the question "Are you mark" the bot will answer "-", 
but if you ask "Are you Mark?" it will say "Yes". We think this characteristic of the bot is not nice.
 
In the end we think that we could improve the model by using more data, more training iteration and/or a larger model.
 
However, after some research on the internet we found out that we are not the only one who have some problems with reproducing the results of the paper. 

See for example this link: https://github.com/nicolas-ivanov/tf_seq2seq_chatbot

To give you an impression of our bot, you can see some results below:


# Somewhat good examples

	USER   : Hello there
	CHATBOT: - hello
	
	USER   : Why can't you behave
	CHATBOT: - i 'm sorry .
	
	USER   : Is there any reason to believe unicorns exist
	CHATBOT: and the most one of the world.
	
	USER   : I'm going to make dinner
	CHATBOT: - i 'm gonna go to you .
	
	USER   : I love you
	CHATBOT: i love you
	
	USER   : Are you mark?
	CHATBOT: - yes .
	
	USER   : Stupid thing
	CHATBOT: - i'm sorry
	
	USER   : Is red a nice color?
	CHATBOT: - yes .
	
	USER   : Can I speak to your father
	CHATBOT: - i'm not a good time
	
	USER   : Goodbye
	CHATBOT: - bye-bye
	
# Somewhat bad examples	
	
	USER   : What is your name?
	CHATBOT: -
	
	USER   : (
	CHATBOT: )
	
	USER   : (
	CHATBOT: - yes .
	
	USER   : Good morning
	CHATBOT: - good
	
	USER   : Bye
	CHATBOT: - hey