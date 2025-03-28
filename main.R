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
tokenizer <- text_tokenizer(num_words = max_features) %>%
  fit_text_tokenizer(training_data)

training_sequences <- texts_to_sequences(tokenizer, training_data)

#Look at how the word frequencies drop off
#pull the word counts
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

