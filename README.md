# Transfer_learning

### Requirements:
Python 3
Python 3 libraries: numpy, sklearn, scipy, pandas, matplotlib, math

### Description:
Realisation of the sample reweighting transfer learning algorythm. The program contains three main steps:

1) Initialisation. We first define variable data (sourse data) as "Div" for HumDiv data or "Var" for HumVar data. After loading necessary libraries, we define the following functions:
reading() - to read and imput the data
cv_wo_protein_intersection () - to define specific folds for cross validation, that don't have the same protein in two folds.
roc_plot() - to plot the roc curves of different classifiers on source data on the same plot.
Then we read the data and initialize the classifiers, according to their optimal parameters on human data.

2) Transfer. We define the weights of the samples from source data according to their distances to target data. We then train each classifier on source data with the weights and without the weights, test on target dataset and print the results.

3) Visualisation. Using cv_wo_protein_intersection() function, we create specific folds for cross validation. Then for each classifier we use the 5-fold cross-validation and measure the classifiers performance with AUC and accuracy metrics. We then give this information as parameters to roc_plot() function. 
