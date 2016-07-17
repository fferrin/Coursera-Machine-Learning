Machine Learning
=====================

Code for programming assignments in Octave from the Coursera course Machine
Learning, given by Andrew Ng for Stanford University.

Assignment 1 - Linear Regression (100%)
----------------------------
 - **computeCost.m** - Compute cost for linear regression.
 - **computeCostMulti.m** - Compute cost for linear regression with multiple
 variables.
 - **featureNormalize.m** - Normalizes the features in a vector.
 - **gradientDescent.m** - Performs gradient descent to learn parameters.
 - **gradientDescentMulti.m** - Performs gradient descent to learn parameters
 with multiple variables.
 - **normalEqn.m** - Computes the closed-form solution to linear regression.
 - **warmUpExercise.m** - Example function in octave.

Assignment 2 - Logistic Regression (100%)
----------------------------
 - **costFunction.m** - Compute cost and gradient for logistic regression.
 - **costFunctionReg.m** - Compute cost and gradient for logistic regression
 with regularization.
 - **predict.m** - Predict whether the label is 0 or 1 using learned logistic.
 - **sigmoid.m** - Compute sigmoid function.

Assignment 3 - Multiclass Classification and Neural Networks (100%)
----------------------------
 - **lrCostFunction.m** - Compute cost and gradient for logistic regression with
 regularization.
 - **oneVsAll.m** - Trains multiple logistic regression classifiers and returns.
 - **predict.m** - Predict the label of an input given a trained neural network.
 - **predictOneVsAll.m** - Predict the label for a trained one-vs-all classifier.

Assignment 4 - Neural Networks Learning (100%)
----------------------------
 - **nnCostFunction.m** - Implements the neural network cost function for a two
 layer neural network which performs classification.
 - **sigmoidGradient.m** - Returns the gradient of the sigmoid function.

Assignment 5 - Regularized Linear Regression and Bias vs. Variance (100%)
----------------------------
 - **learningCurve.m** - Generates the train and cross validation set errors
 needed to plot a learning curve.
 - **linearRegCostFunction.m** - Compute cost and gradient for regularized
 linear regression with multiple variables.
 - **polyFeatures.m** - Maps X (1D vector) into the p-th power.
 - **validationCurve.m** - Generate the train and validation errors needed to
 plot a validation curve that we can use to select lambda.

Assignment 6 - Support Vector Machines (100%)
----------------------------
 - **dataset3Params.m** - Returns your choice of C and sigma for Part 3 of the
 exercise.
 - **emailFeatures.m** - Takes in a word_indices vector and produces a feature
 vector from the word indices.
 - **gaussianKernel.m** - Returns a radial basis function kernel between two
 points.
 - **processEmail.m** - Preprocesses a the body of an email and returns a list
 of word indices.

Assignment 7 - K-means Clustering and Principal Component Analysis (100%)
----------------------------
 - **computeCentroids.m** - Returs the new centroids by computing the means of
 the data points assigned to each centroid.
 - **findClosestCentroids.m** - Computes the centroid memberships for every
 example.
 - **pca.m** - Run principal component analysis on the dataset X.
 - **projectData.m** - Computes the reduced data representation when projecting
 on eigenvectors.
 - **recoverData.m** - Recovers an approximation of the original data when using
 the projected data.

Assignment 8 - Anomaly Detection and Recommender Systems (100%)
----------------------------
 - **cofiCostFunc.m** - Collaborative filtering cost function.
 - **estimateGaussian.m** - Estimates the parameters of a Gaussian distribution
 using the data in X.
 - **selectThreshold.m** - Find the best threshold to use for selecting outliers.
