# Mod 2 Final Project - Selling a House in King County: 
# Predicting and Maximizing Sale Price with Multiple Linear Regression

#### Author: Max Steele

The goal of this project was to build a classifier to predict whether someone was vaccinated against the seasonal flu or not as accurately as possible, while maximizing recall of those that elect not to get the vaccine. The main purposes of this modeling effort is to provide insight into:
 - factors that influence whether someone elects to get vaccinated against the seasonal flu, and 
 - which subsets of the population pro-vaccine campaigns should target when hoping to increase the total number of people receiving the vaccine each year.

## Data
The data used were obtained from 


## Methods
I followed the OSEMN data science process to approach this problem. Initial exploration and scrubbing of the dataset identified and dealt with null or otherwise missing values. 

### Missing Values and Feature Engineering



### Selecting Possible Predictors






### Modeling
Models were built using Scikit-Learn



Model quality and performance were assessed based on 


## Results

### Logistic Regression
#### Baseline Model



#### GridSearch Tuned for `macro_recall` and for `accuracy`


### Decision Tree
#### Baseline Model



#### GridSearch Tuned for `macro_recall` and for `accuracy`


### Random Forest
#### Baseline Model



#### GridSearch Tuned for `macro_recall` and for `accuracy`


### Random Forest
#### Baseline Model



#### GridSearch Tuned for `macro_recall` and for `accuracy`



### XGradient Boosted
#### Baseline Model



#### GridSearch Tuned for `macro_recall` and for `accuracy`



### Stacking Classifier




### Final Model


#### Interpretation of Final Model


## Conclusions


## Recommendations



## Future Work
* The models were all built using data collected in 2009, so a logical next step would be to find or obtain similar datafrom more recent years. This would help make the model more relevant.
* Including additional data from more recent years could allow for a better understanding of how attitudes towards vaccination has changed over time.
* 