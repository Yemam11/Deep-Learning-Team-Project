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

#shuffling training data
indicies <- sample(nrow(training_data))
training_data <- training_data[indicies,]

#shuffling testing data
indicies <- sample(nrow(testing_data))
testing_data <- testing_data[indicies,]


# Pulling the labels, factorizing
training_labels <- training_data %>%
  select(Sentiment) %>% 
  mutate(Sentiment = as.factor(Sentiment))

testing_labels <- testing_data %>%
  select(Sentiment) %>% 
  mutate(Sentiment = as.factor(Sentiment))

#Isolating the tweets
training_data <- training_data$OriginalTweet
testing_data <- testing_data$OriginalTweet

#

summary(training_data)
summary(testing_data)

summary(training_labels)
summary(testing_labels)

#Distribution of sentiments looks reasonably balanced
ggplot(training_labels, aes(x = Sentiment, fill = Sentiment))+
  geom_bar()+
  labs(
    y = "Count",
    title = "Class Distribution"
  )+
  theme(
    plot.title = element_text(hjust = 0.5)
  )


#### C2.2 Preprocessing Text and labels ####

# Replace invalid characters with blanks
training_data <- iconv(training_data, from = "UTF-8", to = "UTF-8", sub = "")
testing_data <- iconv(testing_data, from = "UTF-8", to = "UTF-8", sub = "")


#find the lengths of all tweets
lengths <- nchar(training_data)

#Distribution of lengths
summary(lengths)

#pick the 90th percentile to pad to
# this covers most sequences, but will exclude any outliers
max_length<- quantile(lengths, 0.9)


#Tokenizing training data
training_tokenizer <- text_tokenizer(num_words = max_length) %>%
  fit_text_tokenizer(training_data)

training_sequences <- texts_to_sequences(training_tokenizer, training_data)


#Tokenizing testing data
testing_tokenizer <- text_tokenizer(num_words = max_length) %>%
  fit_text_tokenizer(testing_data)

testing_sequences <- texts_to_sequences(testing_tokenizer, testing_data)

#padding sequences
training_data <- pad_sequences(training_sequences, maxlen = max_length)
testing_data <- pad_sequences(testing_sequences, maxlen = max_length)

