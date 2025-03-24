# Deep-Learning-Team-Project
COVID19 Tweet Categorization Using Deep Learning

# Component 1

Each team will investigate, collect and synthesize information around the concept and use of cloud computing and how it can be used for designing and running Deep Learning models and applications. The team will need to answer the following questions:

- What is cloud computing and what do we mean running deep learning in the cloud?
- What are the required steps and other conditions that are needed for running DL in the cloud?
- What are the different prices and costing models for doing so?
- How do the three main providers (Amazon, Google, Microsoft) compare, in terms of the services offered, prices, tools, etc.? Are there any other options?
- Although the previous questions will concern any possible language or package used for training DL models, you will need to also make specific comment on the use of R and the R keras package.

---

# Component 2

After reviewing available options, you will need to select one approach where you can use freely the R keras package (or by using a “credit”). You will then use strictly the online functionalities and tools from the selected provider and run any models “in the cloud”. Specific details on the modeling application follow:

You will use a dataset on tweets on Coronavirus, previously pulled from Twitter. The tweets have been tagged manually, and assigned one of five types of sentiments (from extremely negative to extremely positive). The data have divided into two csv files, one for training and one for testing. The csv files contain the actual tweets, the sentiment class, and some other information, which you will ignore here.

For this exercise, you will need to perform the following tasks:

- Download the training and test data into R.
- Preprocess the text data and labels, using steps similar to the ones followed in the examples in tutorials 9-10. Present descriptive statistics and characteristics of the data and use reasonable values for the parameters num_words and maxlen.
- Define a simple NN that has one embedding layer, one dense hidden layer and one output layer. Use appropriate parameters and settings for the network, consistent with the size and dimensionality of the data. Choose proper loss and performance metrics.
- Compile and train the network, using a reasonable batch size, and using 20% of the data for validation. Make an optimal choice for the number of epochs using the validation performance. Record and report the results.
- Replace the previous network, with a RNN with one recurrent layer, keeping the embedding layer. Use reasonable values for any remaining hyperparameters. Record and compare the results.
- Now add a second recurrent layer, and observe and report and improvement in the model. Select a “best RNN model” based on the validation performance.
- Replace the simple RNN with a LSTM model, using also dropout. Comment on any improvement in the performance.
- Finally, evaluate your best-performing models one from each type (FFNN, RNN, LSTM) using the test data and labels.
- Include a section of lessons learned, conclusions, limitations and potential next steps, reflecting on your analysis.
