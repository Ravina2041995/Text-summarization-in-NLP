# Text-summarization-in-NLP

Title: Abstractive Text Summarization
By: Ravina Ingole
Contents:
1. Introduction
2.Methods
3.Contribution
4.Results & Discussion

1.Introduction:
This project is regarding Text summarization which is the problem of reducing the number of sentences
and words of the article without changing its meaning. There are different techniques to extract information
from raw text data and use it for a summarization model, overall, they can be categorized
as Extractive and Abstractive. Extractive methods select the most important sentences within a text
(without necessarily understanding the meaning), therefore the result summary is just a subset of the full
text. On the contrary, Abstractive models use advanced NLP (word embeddings) to understand the
semantics of the text and generate a meaningful summary. Consequently.
In this project I have used Abstractive method (Sequence 2 Sequence) for text summarization, The Gated
Recurrent Unit (GRU) is the younger sibling of the more popular Long Short-Term Memory (LSTM)
network, and also a type of Recurrent Neural Network (RNN). Below steps are followed for training model.
Data is collected from “The Charges Bulletin”.
Link of data Source: https://chargerbulletin.com/
Label Preparation: Heading is considered as the target value, and context as text(input).
Mycharger


2.Methods:
For text summarization I have used PyTorch to build a sequence 2 sequence (encoder-decoder) model with simple dot product attention using Gated Recurrent Unit GRU and evaluate their attention scores.
2.1 Algorithms:
The structure of the GRU allows it to adaptively capture dependencies from large sequences of data without discarding information from earlier parts of the sequence. This is achieved through its gating units, like the ones in LSTMs, which solve the vanishing/exploding gradient problem of traditional RNNs. These gates are responsible for regulating the information to be kept or discarded at each time step.
Other than its internal gating mechanisms, the GRU functions just like an RNN, where sequential input data is consumed by the GRU cell at each time step along with the memory, or otherwise known as the hidden state. The hidden state is then re-fed into the RNN cell together with the next input data in the sequence. This process continues like a relay system, producing the desired output.
The GRU cell contains only two gates: the Update gate and the Reset gate. These gates are essentially vectors containing values between 0 to 1 which will be multiplied with the input data and/or hidden state. A 0 value in the gate vectors indicates that the corresponding data in the input or hidden state is unimportant and will, therefore, return as a zero. On the other hand, a 1 value in the gate vector means that the corresponding data is important and will be used.

Encoder: The encoder layer of the seq2seq model extracts information from the input text and encodes it into a single vector, that is a context vector. I have used GRU(Gated Recurrent Unit) for the encoder layer in order to capture long term dependencies - mitigating the vanishing/exploding gradient problem encountered while working with vanilla RNNs. The GRU cell reads one word at a time and using the update and reset gate, computes the hidden state content and cell state.
Decoder: The decoder layer of a seq2seq model uses the last hidden state of the encoder i.e., the context vector and generates the output words. The decoding process starts once the sentence has been encoded and the decoder is given a hidden state and an input token at each step/time.
Model Architecture:
2.2 objective function and network architectures for transfer learning:
A transformer is a deep learning model that adopts the mechanism of self-attention, differentially weighting the significance of each part of the input data. It is used primarily in the fields of natural language processing (NLP) and computer vision. Below is the architecture for T5 Model.

2.3 Mini Network:
For min Network reduced GRU layer from 7 to 4. Also did some changes in the code. Below is Model on which dataset is trained on.
Model Architecture:
2.4 Evaluation:
Used Rouge to evaluate training performance. ROUGE, or Recall-Oriented Understudy for Evaluation, is a set of metrics and a software package used for evaluating automatic summarization and machine translation software in natural language processing. The metrics compare an automatically produced summary or translation against a reference or a set of references (human-produced) summary or translation.
Note: As of now, ROUGE score is coming 0.021739129702859194 after 15000 iterations, it will improve when number of datasets is increased.

3. Contribution:
Note: Everything is done by me it’s an individual project.
Dataset Creation and cleaning dataset:
Dataset was created using University of New Haven charger Bulletin.
Customized dataset:
Created different functions to customize dataset that can fit according to model.

Vocabulary Creation:
Hyperparameter Tuning Changed:
1. Training hyperparameters:
Num_epochs
Learning_rate
Batch_size
2. Model Hyperparameters:
Load_model=False
Device
Input_size_encoder
Input_size_decoder
Output_size
Encoder_embedding_size
Decoder_embedding_size
Hidden_size
Num_layers
Customized Model for Mini-Network:
Created functions for Encoder, Decoder, and model. Edited layers, changes input, embedding and output dimensions to fit in model.

Evaluation and Training Functions:
Changes code for Evaluation and training model.

4. Results:
Used Rouge to evaluate training performance. ROUGE, or Recall-Oriented Understudy for Evaluation, is a set of metrics and a software package used for evaluating automatic summarization and machine translation software in natural language processing. The metrics compare an automatically produced summary or translation against a reference or a set of references (human-produced) summary or translation.
Note: As of now, ROUGE score is coming 0.021739129702859194 after 15000 iterations, it will improve when number of datasets is increased.
Discussion:
For this project Rouge score we very low, but it can get better by increasing dataset. GRU is good model for text summarization. GRU is less complex than LSTM because it has a smaller number of gates. If the dataset is small, then GRU is preferred otherwise LSTM for the larger dataset. GRU exposes the complete memory and hidden layers, but LSTM doesn't.
