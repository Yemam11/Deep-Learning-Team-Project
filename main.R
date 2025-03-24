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

#pick the 90th percentile to pad to
# this covers most sequences, but will exclude any outliers
max_length<- quantile(lengths, 0.9)

#use the 1000 most common words
max_features <- 10000

#Tokenizing training data
training_tokenizer <- text_tokenizer(num_words = max_features) %>%
  fit_text_tokenizer(training_data)

training_sequences <- texts_to_sequences(training_tokenizer, training_data)


#Tokenizing testing data
testing_tokenizer <- text_tokenizer(num_words = max_features) %>%
  fit_text_tokenizer(testing_data)

testing_sequences <- texts_to_sequences(testing_tokenizer, testing_data)

#padding sequences
training_data <- pad_sequences(training_sequences, maxlen = max_length)
testing_data <- pad_sequences(testing_sequences, maxlen = max_length)


#### C2.3 Creating a simple FFNN ####



## Tuning ##

if (F){
  # Hyperparameter grids to search
  hidden_units_grid <- c(128, 256, 512)
  learning_rates    <- c(0.01, 0.001)
  momentums         <- c(0.0, 0.9)
  batch_sizes       <- c(32, 64, 128)
  
  tuning_results <- list()  # store results
  
  best_acc <- 0
  best_config <- NULL
  best_model <- NULL
  
  counter <- 1
  
  for (units in hidden_units_grid) {
    for (lr in learning_rates) {
      for (mom in momentums) {
        for (bs in batch_sizes) {
          
          cat(sprintf("\n=== Training model %d with units=%d, lr=%.4f, momentum=%.1f, batch_size=%d ===\n",
                      counter, units, lr, mom, bs))
          
          # Define the model
          model <- keras_model_sequential() %>%
            layer_embedding(input_dim = max_features, output_dim = 8, input_length = max_length) %>%
            layer_flatten() %>%
            layer_dense(units = units, activation = "relu") %>%
            layer_dense(units = 5, activation = "softmax")
          
          # Compile using SGD with custom lr and momentum
          model %>% compile(
            optimizer = optimizer_sgd(
              learning_rate = lr,
              momentum = mom),
            loss = "categorical_crossentropy",
            metrics = c("accuracy")
          )
          
          # Fit the model
          history <- model %>% fit(
            training_data,
            training_labels,
            epochs = 5,        # set fewer epochs while searching
            batch_size = bs,
            validation_split = 0.2,
            verbose = 0        #Silence output
          )
          
          # Extract final validation accuracy
          val_acc <- tail(history$metrics$val_accuracy, 1)
          cat(sprintf("Final val_accuracy: %.4f\n", val_acc))
          
          # Save results
          tuning_results[[counter]] <- list(
            units      = units,
            learning_rate         = lr,
            momentum   = mom,
            batch_size = bs,
            val_acc    = val_acc
          )
          
          # Keep track of best accuracy
          if (val_acc > best_acc) {
            best_acc <- val_acc
            best_config <- tuning_results[[counter]]
            best_model <- model
          }
          
          counter <- counter + 1
        }
      }
    }
  }
  
  #Looking at the results
  View(tuning_results)
  
  #pulling the best configuration
  best_config
  
  tuning_results[["best_config"]] <- best_config
  
  #Save so you dont have to keep rerunning to get the results
  save(tuning_results, file = "tuning_results.RData")
}


#loading the results
load(file="tuning_results.Rdata")
best_config <- tuning_results[["best_config"]]

#creating the model
model <- keras_model_sequential() %>%
  #embedding layer
  layer_embedding(input_dim = max_features, output_dim = 8, input_length = max_length) %>%
  #flattening
  layer_flatten() %>%  
  #One hidden layer
  layer_dense(units = best_config$units, activation = "relu") %>% 
  #output layer
  layer_dense(units = 5, activation = "softmax")

model

#Compiling the model
model %>% compile(
  optimizer = optimizer_sgd(
    learning_rate = best_config$learning_rate,
    momentum = best_config$learning_rate
  ),
  loss = "categorical_crossentropy",
  metrics = c("acc")
)

#training the model
history <- model %>% fit(
  training_data,
  training_labels,
  epochs = 20,
  batch_size = best_config$batch_size,
  validation_split = 0.2
)


#Compare to 20 epochs using base RMSprop

model <- keras_model_sequential() %>%
  #embedding layer
  layer_embedding(input_dim = max_features, output_dim = 8, input_length = max_length) %>%
  #flattening
  layer_flatten() %>%  
  #One hidden layer
  layer_dense(units = best_config$units, activation = "relu") %>% 
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
  epochs = 20,
  batch_size = best_config$batch_size,
  validation_split = 0.2
)

# RMSprop is better