# Sentiment Analysis Prediction

## Table of Contents
* [Problem Statement](#problemstatement)
* [Approach](#approach)
  * [Introduction](#introduction)
  * [Data Preprocessing](#data_preprocess)
  * [Models Used](#models_used)
* [Analysis of the Output](#analysis)
* [Conclusion](#conclusion)
* [Future Scope](#future_scope)
* [References](#references)

## Problem Statement <a name="problemstatement"></a>
Twitter is a social-media platform, that is used by a vast number of people on the internet. People tend to share their experiences on social-media with respect to some product or service that they liked or disliked. In such a case, having a sentiment analysis system can help an organization to understand whether their customers are satisfied by the product or service offered by them. Such analysis can help an organization to improve their service for their customers. It will also save a lot of time, as a person or a group of people will take more time to analyze enormous amounts of data on the internet. It's often seen that such tweets get misunderstood and hence, having a system that can not only classify the tweet on the basis of words present in it, but can also understand the overall meaning of the sentence is a boon.

Example : "This place lacks good food, good ambience and good desert."

Explanation : Here the person tries to tell that the place or the restaurant that he/she visited DOESNOT provide a good service. But a traditional system that looks at the words individually would classify the review as a positive one as a result of the word "good" that appears 3 times in the sentence.

The dataset that would be used for training the model is sentiment140. It has 1.6M tweets that are extracted using the twitter API. Each tweet is annoated with either 0 (negative) or 4 (positive) and they can be used to detect sentiments.

Dataset can be found <a target="_blank" href="http://cs.stanford.edu/people/alecmgo/trainingandtestdata.zip">here</a>.

## Approach <a name="approach">
### 1. Introduction <a name="introduction">
Recurrent Neural Networks (RNNs), Long Short Term Memory (LSTMs) and Gated Recurrent Units (GRUs) are types of neural networks that have been firmly established as the state of the art approaches when it comes to dealing with sequential data. Inorder to have an intution of how do LSTMs and GRUs work, we first need to understand the working of RNNs. 
#### i) Recurrent Neural Network:
-> A RNN firstly converts words into machine readable vectors. These vectors are then processed as sequences of data one by one.
  ![Alt Text](https://miro.medium.com/max/875/1*AQ52bwW55GsJt6HTxPDuMA.gif)

-> While processing, it passes the previous hidden state to the next step of the sequence. The hidden state acts as a neural network memory. It holds information on the previous data that was seen before.
  ![Alt Text](https://miro.medium.com/max/875/1*o-Cq5U8-tfa1_ve2Pf3nfg.gif)

-> Now this previous hidden state that was passed is combined with the input vector. This newly created vector is has the information of the current input and previous inputs. This vector is now processed using the tanh activation function (this helps to squish the values between -1 and 1) and is passed as an the hidden state input to the next cell.
  ![Alt Text](https://miro.medium.com/max/875/1*WMnFSJHzOloFlJHU6fVN-g.gif)

#### ii) Long Short Term Memory
LSTMs have a similar work flow as that of the RNNs. It consists of different gates that help the model to remember or forget data that is useful or useless as it propogates in a forward direction.

-> A sigmoid activation function is used in the LSTM as it helps to squish the output values between 0 and 1. Thus, this activation function can be used for the gates which would output 0 (considered as negative) or 1(considered as positive).
 ![Alt Text](https://miro.medium.com/max/875/1*rOFozAke2DX5BmsX2ubovw.gif)

-> Firstly, we have the forget gate. This gate decides what information should be kept or thrown away. Information from the previous hidden state and the current are passed to through sigmoid function and the value comes out between 0 and 1. 0 signifies forget and 1 signifies to keep the data.
  ![Alt Text](https://miro.medium.com/max/875/1*GjehOa513_BgpDDP6Vkw2Q.gif)
  
-> To update the cell state, we have the input gate. First, we pass the previous hidden state and current input into a sigmoid function. That decides which values will be updated by transforming the values to be between 0 and 1. 0 means not important, and 1 means important. You also pass the hidden state and current input into the tanh function to squish values between -1 and 1 to help regulate the network. Then you multiply the tanh output with the sigmoid output. The sigmoid output will decide which information is important to keep from the tanh output.
  ![Alt Text](https://miro.medium.com/max/875/1*TTmYy7Sy8uUXxUXfzmoKbA.gif)
  
-> Now we should have enough information to calculate the cell state. First, the cell state gets pointwise multiplied by the forget vector. This has a possibility of dropping values in the cell state if it gets multiplied by values near 0. Then we take the output from the input gate and do a pointwise addition which updates the cell state to new values that the neural network finds relevant. That gives us our new cell state.
  ![Alt Text](https://miro.medium.com/max/875/1*S0rXIeO_VoUVOyrYHckUWg.gif)
  
-> Last we have the output gate. The output gate decides what the next hidden state should be. Remember that the hidden state contains information on previous inputs. The hidden state is also used for predictions. First, we pass the previous hidden state and the current input into a sigmoid function. Then we pass the newly modified cell state to the tanh function. We multiply the tanh output with the sigmoid output to decide what information the hidden state should carry. The output is the hidden state. The new cell state and the new hidden is then carried over to the next time step.
  ![Alt Text](https://miro.medium.com/max/875/1*VOXRGhOShoWWks6ouoDN3Q.gif)

-> The above steps are calculated the following formulae:
  
  `curr_candidate<t> = tanh(Wc[a<t-1>, x<t-1>] + bc)`
  
  `update_gate = sigmoid(Wu[a<t-1>, x<t-1>] + bu)`
  
  `forget_gate = sigmoid(Wf[a<t-1>, x<t-1>] + bf)`
  
  `candidate<t> = update_gate * curr_candidate<t> + forget_gate * candidate<t-1>`
  
  `output_gate = sigmoid(Wo[a<t-1>, x<t-1>] + bo)`
  
  `a<t> = output_gate * tanh(candidate<t>)`
  
### 2. Data Pre-processing <a name="data_preprocess">
LSTM models don't work on raw text. We firstly need to process the data so that we can convert the text into machine readable vectors. Our dataset doesn't include emoticons, but incase we have then we need to convert them as well into machine readable format. The steps followed for data pre-processing are as follows:
  * Decoding the sentiments. (0 -> Negative, 2 -> Neutral, 4 -> Positive)
  * Remove the stopwords. (Stopwords are those words that don't carry much information. Eg => I, myself, the, a, an, etc.)
  * Apply stemming. (Stemming is the process of reducing a word to its word stem that affixes to suffixes and prefixes or to the roots of words known as a lemma. Eg => "having" gets converted to "have".)
  * All sequences must be of the same length. (Add padding if required.)
  
### 3. Model Used <a name="models_used">
The model used for the project has various layers. Firstly, it consists of an Embedding layer. The embedding layer is followed by a dropout layer. Then comes the LSTM layer, which is then followed by a dense layer, that gives the output prediction.
#### i) Embedding Layer:
Embedding layer is among one of the layers that is present in keras. It is mainly used for Natural language processing, but can also be used for other tasks that involve neural networks. For our problem statement, we will be using it along with pre-trained word embeddings such as Word2Vec or GloVe.

##### Word embeddings from Word2Vec :
Word Embeddings can be thought of as one-hot encoders along with dimensionality reduction. It contains the information about a word interms of various features. For our problem statement we have trained the words appearing in our dataset and created word embeddings that have 300 features for each word.
Example :
|   Words  |  Man | Woman |  King | Queen | Orange | Apple |
|:--------:|:----:|:-----:|:-----:|:-----:|:------:|:-----:|
| Features |      |       |       |       |        |       |
|  Gender  |  -1  |   1   | -0.95 |  0.97 |    0   |  0.01 |
| Royality | 0.01 |  0.02 |  0.93 |  0.95 |  -0.01 |   0   |
|    Age   | 0.03 |  0.02 |  0.7  |  0.69 |  0.03  | -0.02 |
|  is food | 0.09 |  0.01 |  0.02 |  0.01 |  0.95  |  0.97 |
  
In the above example, I have considered a few features as example. Making use of such word embedding will help the model to understand the similarities and differences between words. For example, it can compare a man and a woman and notice that they are almost similar expect the gender. The same goes for king and queen. Apart from these, when it compares a man to a king, it would understand that king is just a royal and an older version of a man. This will not only help in sentiment analysis, but would also help the model to correlate the words.

##### Word embeddings from GloVe :
Unlike Word2Vec, GloVe representaion makes use of a different algorithm. It takes into account that how many times does a word appear with another word. Using this it makes use of probability and forms the word embedding for each word in accordance to the other words in the dataset.
  
#### ii) Dropout Layer:
The next layer is the dropout layer. The Dropout layer randomly sets input units to 0 with a frequency of rate at each step during training time, which helps prevent overfitting. Inputs not set to 0 are scaled up by 1/(1 - rate) such that the sum over all inputs is unchanged.
  
#### iii) LSTM Layer:
Further we have the LSTM layer that works similar to the RNN, but with numerous gates. These gates help the LSTM model to keep a track of what information is necessary and what information is not important and thus keep or throw away that data.
  
#### iv) Dense Layer:
Finally the dense layer converts the input given by LSTM into a single number which is the probability of the number to be positive in nature. This number is then with the help of a function checked and thus the output is provided to the user.
  
## Analysis of the output <a name="analysis">
* ACCURACY: 0.7892781496047974
* LOSS: 0.4492000937461853
  
### i) Training and validation accuracy
![image](https://user-images.githubusercontent.com/69694674/139589410-9a9fc0b5-3a97-49b6-a019-b34080ab5dc8.png)

### ii) Training and validation loss
![image](https://user-images.githubusercontent.com/69694674/139589416-d622ce8c-e7fa-40f0-bdda-ac5e641d08d8.png)

In both the graphs, we can clearly see that after each epoch, the parameters are trained in such a way that the accuracy increases and loss decreases for both the training and the validation dataset.
  
### iii) Confusion Matrix
![image](https://user-images.githubusercontent.com/69694674/139589971-edfbc802-898a-4a7c-92de-d882ee9e106c.png)

This matrix shows the percentage of prediction made that were true-positives, true-negatives, false-positive or true-negatives. We can clearly see that the model is has an accuracy of 0.78 for predicting POSITIVE when it is POSITIVE and an accuracy of 0.80 for predicting NEGATIVE when it is NEGATIVE. 
  
## Conclusion <a name="conclusion">
Sentiment analysis or opinion mining is a field of study that analyzes people's sentiments, attitudes or emotions towards certain entities. In this case study, we made a model that can help us get more intution about the different sections of a sentiment analysis model. We used the dataset that comes from the people like ourselves using the twitter API, made a list of the most commonly used words from the dataset, then trained a model for the word embeddings. We later used these embedding to train a model that can analyze the data and give a prediction.
  
## Future Scope <a name="future_scope">
In future, we can make use of Transformers[[1]](#1). Transformers are an advanced version of the RNNs which were first developed by people at Google Brain in 2017. The main advantage of these transformers comes into the picture when the input sequential data would be large enough. It deals with long-term dependencies and precluded parallelization within training examples. Like RNNs, transformers are designed to handle sequential input data, such as natural language, for tasks such as translation and text summarization. However, unlike RNNs, transformers do not necessarily process the data in order. Rather, the attention mechanism provides context for any position in the input sequence. For example, if the input data is a natural language sentence, the transformer does not need to process the beginning of the sentence before the end. Rather, it identifies the context that confers meaning to each word in the sentence. This feature allows for more parallelization than RNNs and therefore reduces training times. Directly inspired from this, BERT[[2]](#2), or Bidirectional Encoder Representations from Transformers, was released a year later by the developers at Facebook. BERT is bidirectional that it allows the model to look back at he previous words, whereas the transformers have the ability to look at any section of the sequence. RNNs, LSTMs and BRUs reading the data sequentially (left-to-right or right-to-left).
  
## References <a name="references"></a>
<a name="1"></a> [1] Vaswani, A., Shazeer, N.M., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A.N., Kaiser, L., & Polosukhin, I. (2017). Attention is All you Need. _ArXiv, abs/1706.03762_.
<br>
<a name="2"></a> [2] Devlin, J., Chang, M., Lee, K., & Toutanova, K. (2019). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. _NAACL_.
