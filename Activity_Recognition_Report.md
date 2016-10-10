# Activity Recognition of Weight Lifting Exercises
Siva Edupuganti  
October 8, 2016  
##### Source: https://github.com/sivaedupuganti/wle-activity-recognition

## Introduction
This document presents a report on activity recognition of weight lifting exercises. It is created as part of the [Practical Machine Learning ](https://www.coursera.org/learn/practical-machine-learning) course on **Coursera**.

## Synopsis
In this report we will review the Human Activity Recognition [HAR] dataset used for the publication^[1]^ below.

The HAR dataset contains data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. 

We will build a model to predict the manner in which they did the exercise based on the data from accelerometers.

[1] *Ugulino, W.; Cardador, D.; Vega, K.; Velloso, E.; Milidiu, R.; Fuks, H. [Wearable Computing: Accelerometers' Data Classification of Body Postures and Movements](http://groupware.les.inf.puc-rio.br/work.jsf?p1=10335). Proceedings of 21st Brazilian Symposium on Artificial Intelligence. Advances in Artificial Intelligence - SBIA 2012. In: Lecture Notes in Computer Science. , pp. 52-61. Curitiba, PR: Springer Berlin / Heidelberg, 2012. ISBN 978-3-642-34458-9. DOI: 10.1007/978-3-642-34459-6_6.*

## Environment setup
This report is built with Rstudio on Mac, using the R libraries below.

```r
library(caret) # Functions to build prediction model
library(randomForest) # Random forest prediction method
```

## Data Source
The training data for this project are available here:
[https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv](https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv)

The test data are available here:
[https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv](https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv)

The data for this project come from this source: http://groupware.les.inf.puc-rio.br/har.

## Read data
Read the training and testing data sets from the urls provided. 

```r
training.url = "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
testing.url = "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"

# Avoid reloading data when re-knitting the document
if (!exists("training.data") || !is.data.frame(training.data)) {
    training.data = read.csv(url(training.url), na.strings = c("NA", "#DIV/0!", ""))
}

if (!exists("testing.data") || !is.data.frame(testing.data)) {
    testing.data = read.csv(url(testing.url), na.strings = c("NA", "#DIV/0!", ""))
}

dim(training.data)
```

```
## [1] 19622   160
```

```r
unique(training.data$classe)
```

```
## [1] A B C D E
## Levels: A B C D E
```

The training data set has 19622 observations and 160 variables. The prediction model needs to be built for outcome variable *classe*, which has 5 levels one each for the way in which the exercise was performed.

## Pre-process data
Pre-process data to exclude columns that may not impact the prediction model.

```r
allCols = names(training.data)
# Identify columns that largely have "NA" values. 
# Using a threshold of 10%, any columns that have more NAs than the threshold will be dropped
NAThreshold = 0.10 
NACols = allCols[colSums(is.na(training.data))/nrow(training.data) > NAThreshold]

# Identify columns that have near zero values as identified from nearZeroVar()
nzvOut = nearZeroVar(training.data, saveMetrics = TRUE)
nzvCols = rownames(nzvOut[nzvOut$nzv == TRUE, ])

# Remove X column as it is a continuously incremented id, and might skew the prediction model
irrelevantCols = c("X")

# Exclude columns identified above to have largely NA values, near-zero variance, or irrelevant to the prediction model
training.data.processed = training.data[, !names(training.data) %in% NACols 
                                          & !names(training.data) %in% nzvCols 
                                          & !names(training.data) %in% irrelevantCols]
dim(training.data.processed)
```

```
## [1] 19622    58
```

Processed training data now has 58 columns, including the outcome variable *classe*. 

## Building prediction model
We will split the *processed* training data into two sets, one for training the model and another for validation. Validation data set will allow us perform validation and generate an estimate of out-of-sample error.


```r
set.seed(12345)
inTrain = createDataPartition(training.data.processed$classe, p = 0.75, list = FALSE)
trData = training.data.processed[inTrain,]
valData = training.data.processed[-inTrain,]
```

We will use *Random Forest* method for the prediction model, as it generally has better accuracy than a simple classification model. Random forest method chooses important predictors at each split and ensures the de-correlation of trees. Obviously this is more computationally intensive and can be difficult for interpretation.

We will also use cross validation while generating the model. Since this is being done on a local computer, we will use a 5-fold validation (instead of default 10)


```r
mod.rf = randomForest(classe ~ ., data = trData,
                      trControl = trainControl(method = "cv", number = 5))

mod.rf
```

```
## 
## Call:
##  randomForest(formula = classe ~ ., data = trData, trControl = trainControl(method = "cv",      number = 5)) 
##                Type of random forest: classification
##                      Number of trees: 500
## No. of variables tried at each split: 7
## 
##         OOB estimate of  error rate: 0.08%
## Confusion matrix:
##      A    B    C    D    E  class.error
## A 4184    1    0    0    0 0.0002389486
## B    0 2848    0    0    0 0.0000000000
## C    0    3 2564    0    0 0.0011686794
## D    0    0    4 2406    2 0.0024875622
## E    0    0    0    2 2704 0.0007390983
```

```r
# Generate outcomes on validation data
valClasses = predict(mod.rf, valData)

# Calculate accuracy and out-of-sample error rate
confMatrix = confusionMatrix(valData$classe, valClasses)
confMatrix$overall[1]
```

```
##  Accuracy 
## 0.9989804
```

The OOB error rate for the random forest model is 0.08%. And the out-of-sample error rate (from prediction on validation data set) is 0.0010196.

## Predictions for testing data
Before we apply the prediction model to testing data, it needs to be processed similar to the training data.
Testing data also has 160 columns, with a *problem_id* column instead of the *classe* column.


```r
testing.data.processed = testing.data[, names(testing.data) %in% names(training.data.processed)]

# Randomforest prediction model requires all factor columns to have same levels in both training & testing data
for (col in names(testing.data.processed)) {
  if (class(testing.data.processed[[col]]) == "factor") {
    levels(testing.data.processed[[col]]) = levels(trData[[col]])
  }
}

dim(testing.data.processed)
```

```
## [1] 20 57
```

Processed testing data has 57 columns, same as processed training data except the outcome (*classe*) column.

We will now apply the prediction model from the section above on the processed testing data set. 

```r
resultClasses = predict(mod.rf, testing.data.processed, type = "class")
resultClasses
```

```
##  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 
##  B  A  B  A  A  E  D  B  A  A  B  C  B  A  E  E  A  B  B  B 
## Levels: A B C D E
```

The above classes represent the predicted manner of excercising based on the accelerometer data in each entry of the testing data set.
