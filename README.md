# AI Text Detection
ECGR 4106/5106 (Real Time Machine Learning) final project under the supervision of Dr. Hamed Tabkhi. This project aims to classify the origin of a tweet as an AI generated text or a human written one.

Due to the rise of high-level and easily accessible natural language processing (NLP) models such as OpenAI’s GPT or Google’s Bard, there has been a rising concern that the technology will be used in negative ways, such as plagiarism or the generation of misinformation. Particularly with misinformation, there is a real danger in the domain of both news and politics, as a model could be trained for tasks such as writing a fake news article, or even a fake speech from an important figure, such as a global leader. 

As social media such as Twitter or Facebook  behaves as a large gathering place online, it poses itself as a prime target for the dissemination of such misinformation. One way to combat this would be the development of a methodology to generate a text authentication system, which can quickly and accurately determine whether a text is a genuine human creation, or an AI generated text.

## Dataset and Data Preporcessing
a “Twitter deep fake” dataset was obtained from Kaggle, called TweetFake (https://www.kaggle.com/datasets/mtesconi/twitter-deep-fake-text). The TweetFake dataset includes 25,000 tweets, split evenly between human and bot classes. It also includes columns for “screen name,” which is the Twitter username, and “class type,” which defines the type of model being used as human, GPT-2, RNN, or others.
Initial inspection of the original dataset revealed a stark contrast between the quality of RNN generated texts vs the other models which led to all the RNN class data points being removed. The Folder data includes the three original train, validation, and test csv files as well as the outputted csv file which combines all the data and applies all cleaning and preprocessing.

The csv file that includes cleaned up data points was then tokenized before being split back into training, validation, and testing datasets. For tokenization, the built-in Keras tokenizer was used for all trained models. Due to Twitter having a 240-character limit on tweets, including spaces, a max sequence length of 100 tokens was chosen, as a full-length 240 character tweet consisting of only 3 letter words would be 60 tokens long. This scenario likely is an edge case, with a higher number of tokens than typical, signifying that this sequence length could potentially be tuned down. However, the max sequence any tweet could possibly need would be larger, at 120 tokens, with a tweet consisting of 120 single characters followed by spaces.

## Approach 1: Word Embedding


To resolve the problem and achieve the desired results, we first adopted a common encoder-decoder architecture, as depicted in figure above. The Encoder component of this architecture used word embeddings, with the pretrained Global Vectors (GloVe) being used in this project. GloVe encoded each token into a vector representation of a preset dimension of 100, and the resulting vector representations were then passed on to the decoder, which acted as the classifier in this project.
