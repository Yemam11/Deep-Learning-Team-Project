# BTC1889 - Deep Learning Team Project
# Authors: Kimberly Corniel, Trinley Palmo, Youssef Emam
# Date: Mar. 31st 2025

# Notes: To run the script ensure the following files are in the same (working directory):
  # main.R (This file)
  # Corona_NLP_test.csv
  # Corona_NLP_train.csv


#### Library Calls and Function Definitions ####

library(keras)
library(tidyverse)
library(ggplot2)


#### C2.1 Importing the data ####

#setting the working directory
setwd(getwd())

# importing the data

training_data <- read.csv("Corona_NLP_train.csv")
testing_data <- read.csv("Corona_NLP_test.csv")

#### C2.2 Preprocessing Text and labels ####

#### shuffling data/EDA ####

indicies <- sample(nrow(training_data))
training_data <- training_data[indicies,]

#shuffling testing data
indicies <- sample(nrow(testing_data))
testing_data <- testing_data[indicies,]


#Distribution of sentiments looks reasonably balanced
training_labels <- training_data$Sentiment

ggplot(data.frame(Sentiment = training_labels), 
       aes(x = fct_relevel(Sentiment, 
                           "Extremely Negative", 
                           "Negative", 
                           "Neutral", 
                           "Positive", 
                           "Extremely Positive"), 
           fill = fct_relevel(Sentiment, 
                              "Extremely Negative", 
                              "Negative", 
                              "Neutral", 
                              "Positive", 
                              "Extremely Positive"))) +
  geom_bar(col = "black") +
  labs(
    x = "Sentiment",
    y = "Count",
    title = "Class Distribution",
    fill = "Sentiment"
  ) +
  theme(
    plot.title = element_text(hjust = 0.5)
  ) + 
  scale_fill_manual(values = c(
    "Extremely Negative" = "#d73027",  # deep red
    "Negative" = "#fc8d59",            # orange-red
    "Neutral" = "#fee08b",             # yellow
    "Positive" = "#91cf60",            # light green
    "Extremely Positive" = "#1a9850"   # deep green
  ))

#### One hot encoding ####

# Pulling the labels, one-hot encoding
training_labels <- training_data$Sentiment %>% 
  as.factor()

#need to zero index to use the to_categorical function
training_labels <- (as.numeric(training_labels)-1) %>% 
  to_categorical()

#testing label to one-hot
testing_labels <- testing_data$Sentiment %>% 
  as.factor()

#need to zero index to use the to_categorical function
testing_labels <- (as.numeric(testing_labels)-1) %>% 
  to_categorical()

#### Processing training data ####

#Isolating the tweets
training_data <- training_data$OriginalTweet
testing_data <- testing_data$OriginalTweet

summary(training_data)
summary(testing_data)

summary(training_labels)
summary(testing_labels)


# Replace invalid characters with blanks
training_data <- iconv(training_data, from = "UTF-8", to = "UTF-8", sub = "")
testing_data <- iconv(testing_data, from = "UTF-8", to = "UTF-8", sub = "")

#### Tokenization ####

#use the 10000 most common words to start
max_features <- 10000

#Tokenizing training data

# tokenizing based on the upper 90% of most frequent words

tokenizer <- text_tokenizer(num_words = max_features) %>%
   fit_text_tokenizer(training_data)

training_sequences <- texts_to_sequences(tokenizer, training_data)

# #Look at how the word frequencies drop off
# #pull the word counts
word_counts <- tokenizer$word_counts

#remove from list format and sort
word_freq <- sort(unlist(word_counts), decreasing = TRUE)

#plot
plot(log10(word_freq), type = "l",
  main = "Word Frequencies (Log Scale)",
  ylab = "Log(Frequency)")

#finding the 90th percentile of frequency
freq90 <- quantile(word_freq, 0.9)

#find the index of the first word that is less frequent than the cutoff
which(word_freq <= freq90)[1]

# This should be our max_features
max_features <- which(word_freq <= freq90)[1]

abline(v = max_features, col = "red", lty = 2, lwd = 2)


# repeat tokenization
#Tokenizing training data
tokenizer <- text_tokenizer(num_words = max_features) %>%
  fit_text_tokenizer(training_data)

training_sequences <- texts_to_sequences(tokenizer, training_data)

#Tokenizing testing data
testing_sequences <- texts_to_sequences(tokenizer, testing_data)

## Padding Sequences ##

#find the lengths of all tweets
lengths <- sapply(training_sequences, length)

#Distribution of lengths
summary(lengths)

#pick the 90th percentile to pad to
# this covers most sequences, but will exclude any outliers
max_length<- quantile(lengths, 0.9)

#padding sequences
training_data <- pad_sequences(training_sequences, maxlen = max_length)
testing_data <- pad_sequences(testing_sequences, maxlen = max_length)

#View the distribution of the lengths 
#make lengths a data frame
tweet_df <- data.frame(length = lengths)
#Compute quantiles for 5% and 95%
upper_bound <- quantile(tweet_df$length, 0.90)
#Classfiy data as within the middle 90% or outside it
tweet_df$category <- ifelse(tweet_df$length < upper_bound, "Lower 90%", "Upper 10%")
# Make the levels into a fator
tweet_df$category <- factor(tweet_df$category, levels = c("Upper 10%", "Lower 90%"))
# Make a histogram with ggplot to show distribution and inside 90%
ggplot(tweet_df, aes(x = length, fill = category)) +
  geom_histogram(binwidth = 5, color = "black", alpha = 0.7) +
  scale_fill_manual(values = c("Lower 90%" = "blue", "Upper 10%" = "red")) +
  labs(title = "Distribution of Tweet Length", x = "Tweet Length", y = "Count", fill = "Legend") +
  theme(
    plot.title = element_text(hjust = 0.5)
  )

#### C2.3 - ####

model <- keras_model_sequential() %>%
  #embedding layer
  layer_embedding(input_dim = max_features, output_dim = 128, input_length = max_length) %>%
  #flattening
  layer_flatten() %>%  
  #One hidden layer
  layer_dense(units = 256, activation = "relu") %>% 
  #output layer
  layer_dense(units = 5, activation = "softmax")

model

#Compiling the model
model %>% compile(
  optimizer = "rmsprop",
  loss = "categorical_crossentropy",
  metrics = c("acc")
)

#### C2.4 ####

#training the model
history <- model %>% fit(
  training_data,
  training_labels,
  epochs = 3, # overfitting starts after 3 epochs accoring to the tuning
  batch_size = 32,
  validation_split = 0.2
)

plot(history)

perf <- evaluate(model, testing_data, testing_labels)
perf

#### C2.5 ####

## Baseline Model
model_rnn <- keras_model_sequential() %>%
  # embedding layer
  layer_embedding(input_dim = max_features, output_dim = 128, input_length = max_length) %>%
  # recurrent layer
  layer_simple_rnn(units = 128) %>%
  # One hidden layer
  layer_dense(units = 256, activation = "relu") %>% 
  # output layer
  layer_dense(units = 5, activation = "softmax")

model_rnn

#Compiling the model
model_rnn %>% compile(
  optimizer = "rmsprop",
  loss = "categorical_crossentropy",
  metrics = c("acc")
)

# #training the model
# history_rnn <- model_rnn %>% fit(
#   training_data,
#   training_labels,
#   epochs = 10,
#   batch_size = 32,
#   validation_split = 0.2
# )

# # test the model 
# perf_rnn <- evaluate(model_rnn, testing_data, testing_labels)
# perf_rnn
# ## Test performance - 0.5847815
# plot(history_rnn)
# save(history_rnn, perf_rnn, file = "rnn_res.RData")

# Train with less epochs
history_rnn_0.1 <- model_rnn %>% fit(
  training_data,
  training_labels,
  epochs = 6,
  batch_size = 32,
  validation_split = 0.2
)

# Test the model
perf_rnn_0.1 <- evaluate(model_rnn, testing_data, testing_labels)
perf_rnn_0.1

plot(history_rnn_0.1)
# save(history_rnn_0.1, perf_rnn_0.1, file = "rnn_res_0.1.RData")

# ## Test model with less number of units
# model_rnn_1.1 <- keras_model_sequential() %>%
#   # embedding layer
#   layer_embedding(input_dim = max_features, output_dim = 128, input_length = max_length) %>%
#   # recurrent layer
#   layer_simple_rnn(units = 64) %>%
#   # One hidden layer
#   layer_dense(units = 256, activation = "relu") %>% 
#   # output layer
#   layer_dense(units = 5, activation = "softmax") %>% compile(
#     optimizer = "rmsprop",
#     loss = "categorical_crossentropy",
#     metrics = c("acc")
#   )

# # training the model
# history_rnn_1.1 <- model_rnn_1.1 %>% fit(
#   training_data,
#   training_labels,
#   epochs = 3,
#   batch_size = 32,
#   validation_split = 0.2
# )

# # test the model 
# perf_rnn_1.1 <- evaluate(model_rnn_1.1, testing_data, testing_labels)
# perf_rnn_1.1
# ## Performance accuracy - 0.6061085

# save(history_rnn_1.1, perf_rnn_1.1, file = "rnn_res_1.1.RData")
# plot(history_rnn_1.1) 

# ## Test model with more number of units
# model_rnn_1.2 <- keras_model_sequential() %>%
#   # embedding layer
#   layer_embedding(input_dim = max_features, output_dim = 128, input_length = max_length) %>%
#   # recurrent layer
#   layer_simple_rnn(units = 256) %>%
#   # One hidden layer
#   layer_dense(units = 256, activation = "relu") %>% 
#   # output layer
#   layer_dense(units = 5, activation = "softmax") %>% compile(
#     optimizer = "rmsprop",
#     loss = "categorical_crossentropy",
#     metrics = c("acc")
#   )

# # training the model
# history_rnn_1.2 <- model_rnn_1.2 %>% fit(
#   training_data,
#   training_labels,
#   epochs = 13,
#   batch_size = 32,
#   validation_split = 0.2
# )

# # test the model 
# perf_rnn_1.2 <- evaluate(model_rnn_1.2, testing_data, testing_labels)
# perf_rnn_1.2

# save(history_rnn_1.2, perf_rnn_1.2, file = "rnn_res_1.2.RData")
# plot(history_rnn_1.2)


#### C2.6 ####
model_rnn2 <- keras_model_sequential() %>%
  # embedding layer
  layer_embedding(input_dim = max_features, output_dim = 128, input_length = max_length) %>%
  # 1st recurrent layer
  layer_simple_rnn(units = 128, return_sequences = T) %>%
  # 2nd recurrent layer
  layer_simple_rnn(units = 64) %>%
  # One hidden layer
  layer_dense(units = 256, activation = "relu") %>% 
  # output layer
  layer_dense(units = 5, activation = "softmax")

model_rnn2

#Compiling the model
model_rnn2 %>% compile(
  optimizer = "rmsprop",
  loss = "categorical_crossentropy",
  metrics = c("acc")
)

#training the model
history_rnn2 <- model_rnn2 %>% fit(
  training_data,
  training_labels,
  epochs = 15,
  batch_size = 32,
  validation_split = 0.2
)

# test the model 
perf_rnn2 <- evaluate(model_rnn2, testing_data, testing_labels)
perf_rnn2

# save(history_rnn2, perf_rnn2, file = "rnn2_res.RData")
plot(history_rnn2)

# # Tuning with more units
# model_rnn2.1 <- keras_model_sequential() %>%
#   # embedding layer
#   layer_embedding(input_dim = max_features, output_dim = 128, input_length = max_length) %>%
#   # 1st recurrent layer
#   layer_simple_rnn(units = 128, return_sequences = T) %>%
#   # 2nd recurrent layer
#   layer_simple_rnn(units = 128) %>%
#   # One hidden layer
#   layer_dense(units = 256, activation = "relu") %>% 
#   # output layer
#   layer_dense(units = 5, activation = "softmax")

# model_rnn2.1

# #Compiling the model
# model_rnn2.1 %>% compile(
#   optimizer = "rmsprop",
#   loss = "categorical_crossentropy",
#   metrics = c("acc")
# )

# #training the model
# history_rnn2.1 <- model_rnn2.1 %>% fit(
#   training_data,
#   training_labels,
#   epochs = 15,
#   batch_size = 32,
#   validation_split = 0.2
# )

# # test the model 
# perf_rnn2.1 <- evaluate(model_rnn2.1, testing_data, testing_labels)
# perf_rnn2.1

# save(history_rnn2.1, perf_rnn2.1, file = "rnn2_res2.1.RData")
# plot(history_rnn2.1)

# # Tuning with less units
# model_rnn2.2 <- keras_model_sequential() %>%
#   # embedding layer
#   layer_embedding(input_dim = max_features, output_dim = 128, input_length = max_length) %>%
#   # 1st recurrent layer
#   layer_simple_rnn(units = 128, return_sequences = T) %>%
#   # 2nd recurrent layer
#   layer_simple_rnn(units = 32) %>%
#   # One hidden layer
#   layer_dense(units = 256, activation = "relu") %>% 
#   # output layer
#   layer_dense(units = 5, activation = "softmax")

# model_rnn2.2

# #Compiling the model
# model_rnn2.2 %>% compile(
#   optimizer = "rmsprop",
#   loss = "categorical_crossentropy",
#   metrics = c("acc")
# )

# #training the model
# history_rnn2.2 <- model_rnn2.2 %>% fit(
#   training_data,
#   training_labels,
#   epochs = 15,
#   batch_size = 32,
#   validation_split = 0.2
# )

# # test the model 
# perf_rnn2.2 <- evaluate(model_rnn2.2, testing_data, testing_labels)
# perf_rnn2.2

# save(history_rnn2.2, perf_rnn2.2, file = "rnn2_res2.2.RData")
# plot(history_rnn2.2)

#### C2.7 ####

# Replace the simple RNN with a LSTM model, using also dropout. Comment on any improvement in the performance.

# LSTM 4
model_lstm4 <- keras_model_sequential() %>%
  # embedding layer
  layer_embedding(input_dim = max_features, output_dim = 64) %>% 
  # lstm layer
  layer_lstm(units = 64, dropout = 0.2, recurrent_dropout = 0.2) %>% # set the dropout rate to 0.2 -> balance overfitting and accuracy  
  # dense layer
  layer_dense(units = 5, activation = "sigmoid")

# compile the model
model_lstm4 %>% compile(
  optimizer = "rmsprop",
  loss = "categorical_crossentropy",
  metrics = c("acc")
)

# run the model 
history_lstm4 <- model_lstm4 %>% fit(
  training_data, 
  training_labels,
  epochs = 10, 
  batch_size = 32, 
  validation_split = 0.2
)

# test the model 
perf_lstm4 <- evaluate(model_lstm4, testing_data, testing_labels)
perf_lstm4

# save the model and its history
save(history_lstm4, perf_lstm4, file = "lstm4.RData")
plot(history_lstm4)

# other LSTM models ran with less high accurscies (testing different parameters)
# models commented out since not used for purpose of this assignment
# model results in report under C2Q7

# model_lstm1 <- keras_model_sequential() %>%
#   layer_embedding(input_dim = max_features, output_dim = 128) %>% # output dim same as others, as is input dim !!! ask youssef
#   layer_lstm(units = 128, dropout = 0.2, recurrent_dropout = 0.2) %>% # set the dropout rate to 0.2 -> balance overfitting and accuracy  
#   layer_dense(units = 5, activation = "sigmoid")
# 
# # compile the model
# model_lstm1 %>% compile(
#   optimizer = "rmsprop",
#   loss = "categorical_crossentropy",
#   metrics = c("acc")
# )
# 
# # run the model 
# history_lstm1 <- model_lstm1 %>% fit(
#   training_data, 
#   training_labels,
#   epochs = 20, 
#   batch_size = 32, 
#   validation_split = 0.2
# )
# 
# 
# # test the model 
# perf_lstm1 <- evaluate(model_lstm1, testing_data, testing_labels)
# perf_lstm1
# 
# # save the model and its history
# save(history_lstm1, perf_lstm1, file = "lstm1.RData")
# plot(history_lstm1)
# 
# # 2 try with 3 epcohs
# model_lstm2 <- keras_model_sequential() %>%
#   layer_embedding(input_dim = max_features, output_dim = 64) %>% # output dim same as others, as is input dim !!! ask youssef
#   layer_lstm(units = 64, dropout = 0.2, recurrent_dropout = 0.2) %>% # set the dropout rate to 0.2 -> balance overfitting and accuracy  
#   layer_dense(units = 5, activation = "sigmoid")
# 
# 
# # compile the model
# model_lstm2 %>% compile(
#   optimizer = "rmsprop",
#   loss = "categorical_crossentropy",
#   metrics = c("acc")
# )
# 
# # run the model 
# history_lstm2 <- model_lstm2 %>% fit(
#   training_data, 
#   training_labels,
#   epochs = 3, 
#   batch_size = 32, 
#   validation_split = 0.2
# )
# 
# # test the model 
# perf_lstm2 <- evaluate(model_lstm2, testing_data, testing_labels)
# perf_lstm2
# 
# # save the model and its history
# save(history_lstm2, perf_lstm2, file = "lstm2.RData")
# plot(history_lstm2)
# 
# # up the dropoiut
# # LSTM 3
# model_lstm3 <- keras_model_sequential() %>%
#   layer_embedding(input_dim = max_features, output_dim = 64) %>% # output dim same as others, as is input dim !!! ask youssef
#   layer_lstm(units = 64, dropout = 0.3, recurrent_dropout = 0.3) %>% # set the dropout rate to 0.2 -> balance overfitting and accuracy  
#   layer_dense(units = 5, activation = "sigmoid")
# 
# 
# # compile the model
# model_lstm3 %>% compile(
#   optimizer = "rmsprop",
#   loss = "categorical_crossentropy",
#   metrics = c("acc")
# )
# 
# # run the model 
# history_lstm3 <- model_lstm3 %>% fit(
#   training_data, 
#   training_labels,
#   epochs = 15, 
#   batch_size = 32, 
#   validation_split = 0.2
# )
# 
# # test the model 
# perf_lstm3 <- evaluate(model_lstm3, testing_data, testing_labels)
# perf_lstm3
# 
# # save the model and its history 
# save(history_lstm3, perf_lstm3, file = "lstm3.RData")
# plot(history_lstm3)
#
# LSTM 5
# model_lstm5 <- keras_model_sequential() %>%
#   layer_embedding(input_dim = max_features, output_dim = 64) %>% # output dim same as others, as is input dim !!! ask youssef
#   layer_lstm(units = 32, dropout = 0.3, recurrent_dropout = 0.3) %>% # set the dropout rate to 0.2 -> balance overfitting and accuracy  
#   layer_dense(units = 5, activation = "sigmoid")
# 
# # compile the model
# model_lstm5 %>% compile(
#   optimizer = "rmsprop",
#   loss = "categorical_crossentropy",
#   metrics = c("acc")
# )
# 
# # run the model 
# history_lstm5 <- model_lstm5 %>% fit(
#   training_data, 
#   training_labels,
#   epochs = 10, 
#   batch_size = 32, 
#   validation_split = 0.2
# )
# 
# # test the model 
# perf_lstm5 <- evaluate(model_lstm5, testing_data, testing_labels)
# perf_lstm5
# 
# # save the model and its history
# save(history_lstm5, perf_lstm5, file = "lstm5.RData")
# plot(history_lstm5)
# 
# # LSTM 6
# model_lstm6 <- keras_model_sequential() %>%
#   layer_embedding(input_dim = max_features, output_dim = 64) %>% # output dim same as others, as is input dim !!! ask youssef
#   layer_lstm(units = 64, dropout = 0.3, recurrent_dropout = 0.3) %>% # set the dropout rate to 0.2 -> balance overfitting and accuracy  
#   layer_dense(units = 5, activation = "sigmoid")
# 
# # compile the model
# model_lstm6 %>% compile(
#   optimizer = "rmsprop",
#   loss = "categorical_crossentropy",
#   metrics = c("acc")
# )
# 
# # run the model 
# history_lstm6 <- model_lstm6 %>% fit(
#   training_data, 
#   training_labels,
#   epochs = 10, 
#   batch_size = 32, 
#   validation_split = 0.2
# )
# 
# # test the model 
# perf_lstm6 <- evaluate(model_lstm6, testing_data, testing_labels)
# perf_lstm6
# 
# save(history_lstm6, perf_lstm6, file = "lstm6.RData")
# plot(history_lstm6)
# 

