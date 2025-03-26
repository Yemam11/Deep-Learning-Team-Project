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

#shuffling data
indicies <- sample(nrow(training_data))
training_data <- training_data[indicies,]

#shuffling testing data
indicies <- sample(nrow(testing_data))
testing_data <- testing_data[indicies,]


#Distribution of sentiments looks reasonably balanced
training_labels <- training_data$Sentiment


ggplot(data.frame(Sentiment = training_labels), aes(x = Sentiment, fill = Sentiment))+
  geom_bar()+
  labs(
    y = "Count",
    title = "Class Distribution"
  )+
  theme(
    plot.title = element_text(hjust = 0.5)
  )


# Pulling the labels, one-hot encoding
training_labels <- training_data$Sentiment %>% 
  as.factor()

#need to zero index to use the to_categorical function
training_labels <- (as.numeric(training_labels)-1) %>% 
  to_categorical()

testing_labels <- testing_data$Sentiment %>% 
  as.factor()

#Isolating the tweets
training_data <- training_data$OriginalTweet
testing_data <- testing_data$OriginalTweet

summary(training_data)
summary(testing_data)

summary(training_labels)
summary(testing_labels)

#### C2.2 Preprocessing Text and labels ####

# Replace invalid characters with blanks
training_data <- iconv(training_data, from = "UTF-8", to = "UTF-8", sub = "")
testing_data <- iconv(testing_data, from = "UTF-8", to = "UTF-8", sub = "")


#find the lengths of all tweets
lengths <- nchar(training_data)

#Distribution of lengths
summary(lengths)

## ADD TO MAIN BRANCH !!!
# View the distribution of the lengths 

# make lengths a data frame
tweet_df <- data.frame(length = lengths)

# Compute quantiles for 5% and 95%
lower_bound <- quantile(tweet_df$length, 0.05)
upper_bound <- quantile(tweet_df$length, 0.95)

tweet_df$category <- ifelse(tweet_df$length < lower_bound | tweet_df$length > upper_bound, "Outside", "Middle 90%")

# Make a histogram with ggplot to show distribution and inside 90%
p <- ggplot(tweet_df, aes(x = length, fill = category)) +
  geom_histogram(binwidth = 15, color = "black", alpha = 0.7) +
  scale_fill_manual(values = c("Middle 90%" = "blue", "Outside" = "red")) +
  labs(title = "Distribution of Tweet Length", x = "Tweet Length", y = "Count") +
  theme_minimal()
print(p)
 
#pick the 90th percentile to pad to
# this covers most sequences, but will exclude any outliers
max_length<- quantile(lengths, 0.9)

#use the 1000 most common words
max_features <- 10000

#Tokenizing training data
tokenizer <- text_tokenizer(num_words = max_features) %>%
  fit_text_tokenizer(training_data)

training_sequences <- texts_to_sequences(tokenizer, training_data)


#Tokenizing testing data

testing_sequences <- texts_to_sequences(tokenizer, testing_data)

#padding sequences
training_data <- pad_sequences(training_sequences, maxlen = max_length)
testing_data <- pad_sequences(testing_sequences, maxlen = max_length)


#### C2.3 ####

model <- keras_model_sequential() %>%
  #embedding layer
  layer_embedding(input_dim = max_features, output_dim = 8, input_length = max_length) %>%
  #flattening
  layer_flatten() %>%  
  #One hidden layer
  layer_dense(units = 512, activation = "relu") %>% 
  #output layer
  layer_dense(units = 5, activation = "softmax")

model

#Compiling the model
model %>% compile(
  optimizer = "rmsprop",
  loss = "categorical_crossentropy",
  metrics = c("acc")
)

#training the model
history <- model %>% fit(
  training_data,
  training_labels,
  epochs = 10,
  batch_size = 32,
  validation_split = 0.2
)

#### C2.7 ####

# Replace the simple RNN with a LSTM model, using also dropout. Comment on any improvement in the performance.

# train the LSTM model the same way as the RNN was trained
# add dropout -> both overall and recurrent dropout -> set to 0.2 !!!3
model_lstm <- keras_model_sequential() %>%
  layer_embedding(input_dim = max_features, output_dim = 8) %>% ## output dim same as others, as is input dim !!!
  layer_lstm(units = 8, dropout = 0.2, recurrent_dropout = 0.2) %>%
  layer_dense(units = 5, activation = "sigmoid")

# compile the model
model_lstm %>% compile(
  optimizer = "rmsprop",
  loss = "categorical_crossentropy",
  metrics = c("acc")
)

# run the model 
history_lstm <- model_lstm %>% fit(
  training_data, 
  training_labels,
  epochs = 10,
  batch_size = 32,
  validation_split = 0.2
)

# set up testing labels -> ADD EARLIER TO MAIN !!!
testing_labels <- (as.numeric(testing_labels)-1) %>% 
  to_categorical()

# test the model 
perf_lstm <- evaluate(model_lstm, testing_data, testing_labels)
perf_lstm
