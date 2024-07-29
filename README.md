# AI - application which can perform its own task without any human intervention
# ML - it provides stats to analyze, visualize, predict, forecast the data.
# DL - mimics the human brain. multilayer neural network
# Data Science - overlaps over all the fields
# EDA - Exploratory Data Analysis

# imputer, standard scaling, onehotencoder
# difference between fit and fit_transform  -  fit calcs parameters needed for transformation, fit_transformation combines both fitting and transforming in one step
# np.c_   concatenate arrays along columns in numpy to combine i/p and o/p variables

# ------------------------------------------------------------------------------------------------------------
# Numpy - Numerical Python, supports large nD arrays with math fns.

# Pandas - data analysis and manipulation lib built ontop of numpy
# Data Frame - 2D, size mutable, heterogeneous tabular data struct with labeled axes
# Series - 1D heterogeneous labeled array
# describe
# isnull
# fillna

# Matplotlib - creates visualizations, histograms, bar, pie, graph etc

# Seaborn - draws better stats graphs, box, violin, histo, line, kde, pairplot, heatmap etc

# Logging - tracks events, diognoses problems, debugs

# Pickle    - used for serializing and deserializing py objects, to be saved as a file
# --------------------------------------------------------------------------------------------------------------------
# Statistics - field that deals with calculation, organization, analysis, interpretation and presentation of data.
# Descriptive - organizing and summarizing of data(avg)
# Measure of central tendency - Mean, Median, Mode
# Measure of Dispersion - Variance, Standard Deviation

# Inferential - conclusion based on experiments(conclude)
# Sample Data - part of the total points (bessel correction, dof)
# Population Data - total data points

# Varirable - 
# Quantitative - Discrete, Continuous
# Qualitative/Categorical 

# Percentiles - a value below which certain percentage of observations lie
# Quartiles - quarter perncentile

# Probability - It is about determining the likelihood of an event
# Addition Rule - 
# mutually exlclusive events - if events cannot occur at the same time
# Non mutually exclusive - if events can 

# Multiplication Rule - 
# independent events
# dependent events - conditional probab
#  -----------------------------------------------------------------------------------------------------------------------
# SQL - structured query language for managing relational databases
# SQLite - self-contained, serverless and zero config DB engine for embedded DB systems

# Docker - open platform for developing, shipping and running apps by separating apps form infra
# Docker Image(package/artifact) - list of all images of all the dependencies installed, virtualizes only app layer
# Container - a running docker image
# Virtual Machine - virtualizes both app layer and kernel
# Operating System - 
# Kernel - Layer 1 
# Application Layer - Layer 2
# -----------------------------------------------------------------------------------------------------------------
# Machine Learning Techniques - 
# Supervised MLT - Has an independent input and dependent output feature  
# if Continuous -> Regression -> Linear, Ridge & Lasso, ElasticNet
# if Categorical -> Classification ->binary or multiclass -> Logistic
# Both Classification and Regression- Decision Tree, Random Forest, AdaBoost, XGBoost

# Unsupervised MLT - create clusters ->K-means, Hierarichal mean, DBScan clustering

# Reinforcement Learning - agent learns to make decisions by receiving rewards or penalties


# Instance based learning - completely dependent on data, knn, memorization method
# Model based learning - identifies pattern, generalization method
# .......................................................................................................................................................................................................

# Simple Linear Regression - Prediction done based on Best Fit Line which is calculated
# Gradeint Descent
# Convergence algo
# learning rate

# Multiple Linear Regression - multiple input features(independent) 
# OLS Linear Regression - ordinary least squares - estimates coeffs by minimizing SSr(sum of squared residuals)(pred-act)
# Ridge Regression(L2 regularization) - adds penalty to size of coeffs to reduce overfitting and improve generalization
# Lasso Regression(L1 regularization) (Least Absolute Shrinkage and Selection Operator)- adds penalty to absolute size of coeffs to reduce overfitting and perform feature selection by shrinking some coeffs to zero
# ElasticNet Regression - combination of Ridge and Lasso

# Polynomial Regression - Non linear, degree
# ............................................................................................................................................................................................................
# Performance metrics - cost function
# Rsq = 1-SSres/SStotal, SS=sum of squared
# adjusted Rsq
# Mean Squared Error = MSE = Not Robust, not same units, differentiable
# Mean absolute Error = MAE = little Robust, subgradients, same units
# Root Mean Square Error = Not Robust, same units, differentiable
# LogRMSE

# Dataset -
# Training - 
# train 
# validation - Hyperparameter tuning 
# Testing

# Low bias - good accuracy
# High variance - bad accuracy in testing
# Overfitting - good accuracy in training but bad in testing, low bias high variance
# Underfitting - bad in training and testing, high bias high variance

# Types of Cross Validation - 
# Leave One Out CV(LOOCV) - each datapoint used as single test instance, while remaining are used as training set
# Leave P out CV - P datapoints used as test instance, while remaininf as training set
# K Fold CV - dataset divided into K folds, each fold as test while remaining as training
# Stratified K Fold CV - Each fold has same proportion of classes
# Time Series CV - spiltting done based on chronological order, expanding/rolling window
# ..............................................................................................................................................................................................................

# Handling missing values - 
# delete rows that have missing values
# delete columns that have the most missing values
# imputation techniques - 
# Mean value Imputation - for normal dist. data
# Median value Imputation - for datasets with outliers
# Random sampling
