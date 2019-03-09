# deep_learning_language_bot
A language model bot built on IMDB dataset
# Introduction
A language model bot is built to learn the structure of engilsh language. It learns language by trying to predict the next word given a set of previous words(ngrams).
# Getting the Dataset

We need to download the IMDB large movie reviews from this site: http://ai.stanford.edu/~amaas/data/sentiment/ Direct link : http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz

# Standardize format
The pandas dataframe is used to store text data in a newly evolving standard format of label followed by text columns. 
This was influenced by a paper by Yann LeCun:
(https://drive.google.com/drive/u/0/folders/0Bz8a_Dbh9Qhbfll6bVpmNUtUcFdjYmF2SEpmZUZUcVNiMUw1TWN6RDV3a0JHT3kxLVhVR2M).
The LM can benefit from all the textual data and there is no need to exclude the unsup/unclassified movie reviews.

We first concat all the train(pos/neg/unsup = 75k) and test(pos/neg=25k) reviews into a big chunk of 100k reviews. And then we use sklearn splitter to divide up the 100k texts into 90% training and 10% validation sets.

# Language model tokens

In this section, we start cleaning up the messy text. There are 2 main activities we need to perform:

Clean up extra spaces, tab chars, new ln chars and other characters and replace them with standard ones
Use the awesome spacy library to tokenize the data. Since spacy does not provide a parallel/multicore version of the tokenizer, the fastai library adds this functionality. This parallel version uses all the cores of your CPUs and runs much faster than the serial version of the spacy tokenizer.
Tokenization is the process of splitting the text into separate tokens so that each token can be assigned a unique index. This means we can convert the text into integer indexes our models can use.

We use an appropriate chunksize as the tokenization process is memory intensive

# wikitext103 conversion

We are now going to build an english language model for the IMDB corpus using a technique called transfer learning to make this process easier.  In transfer learning (a fairly recent idea for NLP) a pre-trained LM that has been trained on a large generic corpus(like wikipedia articles) can be used to transfer it's knowledge to a target LM and the weights can be fine-tuned.

Our source LM is the wikitext103 LM created by Stephen Merity @ Salesforce research. Link to dataset The language model for wikitext103 (AWD LSTM) has been pre-trained and the weights can be downloaded here: Link. Our target LM is the IMDB LM.

# Language model
It is fairly straightforward to create a new language model using PyTorch or fastai. But we have designed our model in a way that our model will have a backbone which is nothing but IMDB LM pre-trained with wikitext

bptt (also known traditionally in NLP LM as ngrams) in fastai LMs is approximated to a std. deviation around 70, by perturbing the sequence length on a per-batch basis because in NLP we cannot shuffle inputs and we have to maintain statefulness.

Since we are predicting words using ngrams, we want our next batch to line up with the end-points of the previous mini-batch's items. batch-size is constant and but the fastai library expands and contracts bptt each mini-batch using a clever stochastic implementation of a batch.

The goal of the LM is to learn to predict a word/token given a preceeding set of words(tokens). We take all the movie reviews in both the 90k training set and 10k validation set and concatenate them to form long strings of tokens. In fastai, we use the LanguageModelLoader to create a data loader which makes it easy to create and use bptt sized mini batches. The LanguageModelLoader takes a concatenated string of tokens and returns a loader.

Fastai has a special modeldata object class for LMs called LanguageModelData to which we can pass the training and validation loaders and get in return the model itself.

We setup the dropouts for the model - these values have been chosen after experimentation. If you need to update them for custom LMs, you can change the weighting factor (0.7 here) based on the amount of data you have. For more data, you can reduce dropout factor and for small datasets, you can reduce overfitting by choosing a higher dropout factor. No other dropout value requires tuning.

We first tune the last embedding layer so that the missing tokens initialized with mean weights get tuned properly. So we freeze everything except the last layer.

We also keep track of the accuracy metric.

We set learning rates and fit our IMDB LM. We first run one epoch to tune the last layer which contains the embedding weights. This should help the missing tokens in the wikitext103 learn better weights.

Note that we print out accuracy and keep track of how often we end up predicting the target word correctly. While this is a good metric to check, it is not part of our loss function as it can get quite bumpy. We only minimize cross-entropy loss in the LM.

The exponent of the cross-entropy loss is called the perplexity of the LM. (low perplexity is better).
