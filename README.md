# Multi-facet_softmax

We used the Amazon books dataset (30 gb). The first task in processing the dataset is to create a purchase history of each buyer. For this we create a dictionary where keys are the buyers and values are lists of books purchased by each buyer. The lists are sorted in order of their purchase date. The next step is to convert this dictionary into indices. This means each english word is given a numeric value. After this, the next task is to convert to tensor so that data could be fed to the GPT-2 model. For our model, we created two lists, one is features and the other is labels. The alternative values of labels were assigned to -100 which were categories because we need to predict only products. We create the train, test and validation sets that are 90%, 5%,5% respectively of the entire dataset.

After this the dataset is prepared, we can feed it to the GPT-2 model. The model could only take one tensor list of either features or labels. It needs to be changed to accommodate both labels and features. Then we trained the model for around 100 epochs.

