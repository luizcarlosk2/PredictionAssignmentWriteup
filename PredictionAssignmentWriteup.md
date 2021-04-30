---
title: "Prediction Assignment Writeup"
author: "Luiz Carlos Franze"
date: "29/04/2021"
output: 
  html_document: 
    keep_md: yes
---



## Human Activity Recognition (HAR) Analysis and prediction

### Summary


Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement – a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, our goal is to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways.

More information about this study, you can check on this [link](http://web.archive.org/web/20161224072740/http:/groupware.les.inf.puc-rio.br/har).


### Preliminar analysis

Load main libraries and download the training data and test data:


```r
library(dplyr)
library(ggplot2)
library(caret)
#library(lubridate)
library(rlist)
library(tictoc)
library(rattle)
library(ggcorrplot)

set.seed(202104)
dl_list <- c("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv",
             "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv")
fl_name_list <- c("pml-training.csv",
                  "pml-testing.csv")

x <- c(1:length(dl_list))
# Download datasets if doesn't exist
for (i in 1:length(dl_list)){
    if(!file.exists(fl_name_list[i])){
        download.file(dl_list[i],fl_name_list[i])
        print(paste("Downloading file",fl_name_list[i]))
    }
    else {print(paste("File",fl_name_list[i], "already exists."))}
}
```

```
## [1] "File pml-training.csv already exists."
## [1] "File pml-testing.csv already exists."
```

```r
na_common_values <- c("", " ", "#DIV/0!", "N/A", "NA", "Nan", "")

# Loading both files to Dataset and setting NA Values as real NA
df_training <- as_tibble(read.csv(fl_name_list[1],na.strings = na_common_values))
df_testing <- as_tibble(read.csv(fl_name_list[2],na.strings = na_common_values))

#List to measure the Model training time:
time_execution <- list()
```

Lets take a look into the training dataset dimension:


```r
dim(df_training)
```

```
## [1] 19622   160
```

So there are 160 variables. Columns from 1 to 7 is relalted to index(X), user(user_name), time and date of measurement sensor (raw_timestamp_part_1, raw_timestamp_part_2, cvtd_timestamp, new_window, num_window). The Variabe 160 is our value to predict. a Factor with 5 levels: A, B, C, D and E. All other variables are our sensor values. So, We will discard the the columns 1 to 7. So we will keep only variables related to the sensors and the classe:


```r
names(df_training[c(1:7,160)])
```

```
## [1] "X"                    "user_name"            "raw_timestamp_part_1"
## [4] "raw_timestamp_part_2" "cvtd_timestamp"       "new_window"          
## [7] "num_window"           "classe"
```


```r
df_training <- df_training[-c(1:7)]
```


Now lets take a look in the dataset, if there are any variable, with relative number of N/A's:



```r
check_perc_na <- function(df){ # Function to return a table with number of N/A's in %
    df <- t(as.data.frame.list(colMeans(is.na(df))))
    df <- data.frame(df)
    colnames(df) <- c("NA_Perc")
    df$NA_Perc <- trunc(df$NA_Perc*100)/100
    result <- list("table" = table(df), "df" = df)
    return(result)
}

check_perc_na(df_training)$table
```

```
## df
##    0 0.97 0.98    1 
##   53   81   13    6
```

So, 53 variables has 0% N/A's, 94 variables has between 97% and 99% of N/A's and 6 variables has 100% N/A's.
So, we will remove from the dataset, the variables with those huge N/A's values (Percentage of N/As > 97%):


```r
index <- check_perc_na(df_training)$df$NA_Perc < 0.97
df_training <- df_training[index]
dim(df_training)
```

```
## [1] 19622    53
```



Now we remain a dataset with 52 predictors + the "classe" variable to be predicted.

Lets do a double check if we have a dataset with 100% values no N/A:



```r
check_perc_na(df_training)$table
```

```
## df
##  0 
## 53
```

---

### Split the data to Training and Test datasets:




```r
inTrain <- createDataPartition(y=df_training$classe, p=0.7, list=FALSE)
train.dataset <- df_training[inTrain,]
test.dataset <- df_training[-inTrain,]
```



Now lets take a look in the features correlation in train.dataset:



```r
test.dataset.corr <- test.dataset
test.dataset.corr$classe <- as.numeric(test.dataset.corr$classe)
corr_train <- round(cor(test.dataset.corr), 2)
ggcorrplot(corr_train,insig = "blank")
```

![](PredictionAssignmentWriteup_files/figure-html/unnamed-chunk-9-1.png)<!-- -->


As we can see, there are not so much correlated feature direct to classe.

---

### Creating the models

We are going to create predict models.

I tired to use GBM was to boost predictors but it take ' 27 minutes and take so many RAM resources after finished, so I decided to exclude it. But I keep the code in the document, but set to not to execute (`markdown chunk option: eval = FALSE`) the code for reproducible if necessary:


```r
# This code was set to not execute!
# Runtime in sec: 1625.705
# High RAM allocation
tic()
modFitGBM <- train(classe ~. ,method="gbm",data=train.dataset,verbose=FALSE)
exec_time <- toc(quiet=T)
exec_time <- exec_time$toc - exec_time$tic
exec_time <- exec_time[[1]]
time_execution[["gbm"]] <- exec_time
```



Lets Start creating a Random Forest Model:



```r
tic()
modFitRF <- train(classe~., method="rf", data=train.dataset,
                   trControl = trainControl(method="cv"),number=4) #cross validation and 4 interactions
exec_time <- toc(quiet=T)
exec_time <- exec_time$toc - exec_time$tic
exec_time <- exec_time[[1]]
time_execution[["rf"]] <- exec_time
```





```r
modFitRF$finalModel
```

```
## 
## Call:
##  randomForest(x = x, y = y, mtry = param$mtry, number = 4) 
##                Type of random forest: classification
##                      Number of trees: 500
## No. of variables tried at each split: 27
## 
##         OOB estimate of  error rate: 0.71%
## Confusion matrix:
##      A    B    C    D    E class.error
## A 3902    3    1    0    0 0.001024066
## B   18 2628   11    1    0 0.011286682
## C    0   12 2373   11    0 0.009599332
## D    0    1   26 2224    1 0.012433393
## E    0    0    7    6 2512 0.005148515
```



Decision Tree Model:



```r
tic()
modFitDT <- train(classe ~ .,method="rpart",data=train.dataset) # rpart regression and classifications trees.
exec_time <- toc(quiet=T)
exec_time <- exec_time$toc - exec_time$tic
exec_time <- exec_time[[1]]
time_execution[["rf"]] <- exec_time
```



```r
modFitDT$finalModel
```

```
## n= 13737 
## 
## node), split, n, loss, yval, (yprob)
##       * denotes terminal node
## 
##  1) root 13737 9831 A (0.28 0.19 0.17 0.16 0.18)  
##    2) roll_belt< 130.5 12575 8679 A (0.31 0.21 0.19 0.18 0.11)  
##      4) pitch_forearm< -26.7 1224   51 A (0.96 0.042 0 0 0) *
##      5) pitch_forearm>=-26.7 11351 8628 A (0.24 0.23 0.21 0.2 0.12)  
##       10) magnet_dumbbell_y< 439.5 9562 6893 A (0.28 0.18 0.24 0.19 0.11)  
##         20) roll_forearm< 122.5 5983 3585 A (0.4 0.18 0.19 0.17 0.06) *
##         21) roll_forearm>=122.5 3579 2387 C (0.076 0.17 0.33 0.23 0.19) *
##       11) magnet_dumbbell_y>=439.5 1789  875 B (0.03 0.51 0.044 0.23 0.19) *
##    3) roll_belt>=130.5 1162   10 E (0.0086 0 0 0 0.99) *
```



```r
fancyRpartPlot(modFitDT$finalModel)
```

![](PredictionAssignmentWriteup_files/figure-html/unnamed-chunk-13-1.png)<!-- -->

---

### Testing Models




```r
print("Confusion Matrix of Random Forest:")
```

```
## [1] "Confusion Matrix of Random Forest:"
```

```r
predRF <- predict(modFitRF,test.dataset); 
test.dataset$predRight <- predRF==test.dataset$classe
confusionMatrix(predRF,test.dataset$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1670    9    0    0    0
##          B    2 1129    5    0    1
##          C    1    1 1019    5    0
##          D    0    0    2  959    3
##          E    1    0    0    0 1078
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9949          
##                  95% CI : (0.9927, 0.9966)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9936          
##                                           
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9976   0.9912   0.9932   0.9948   0.9963
## Specificity            0.9979   0.9983   0.9986   0.9990   0.9998
## Pos Pred Value         0.9946   0.9930   0.9932   0.9948   0.9991
## Neg Pred Value         0.9990   0.9979   0.9986   0.9990   0.9992
## Prevalence             0.2845   0.1935   0.1743   0.1638   0.1839
## Detection Rate         0.2838   0.1918   0.1732   0.1630   0.1832
## Detection Prevalence   0.2853   0.1932   0.1743   0.1638   0.1833
## Balanced Accuracy      0.9977   0.9948   0.9959   0.9969   0.9980
```


```r
print("Confusion Matrix of Decision Tree:")
```

```
## [1] "Confusion Matrix of Decision Tree:"
```

```r
predDT <- predict(modFitDT,test.dataset); 
test.dataset$predRight <- predDT==test.dataset$classe
confusionMatrix(predDT,test.dataset$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1527  492  460  426  164
##          B   26  352   30  164  147
##          C  117  295  536  374  292
##          D    0    0    0    0    0
##          E    4    0    0    0  479
## 
## Overall Statistics
##                                           
##                Accuracy : 0.4918          
##                  95% CI : (0.4789, 0.5046)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.3357          
##                                           
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9122  0.30904  0.52242   0.0000  0.44270
## Specificity            0.6338  0.92267  0.77814   1.0000  0.99917
## Pos Pred Value         0.4976  0.48957  0.33209      NaN  0.99172
## Neg Pred Value         0.9478  0.84766  0.88527   0.8362  0.88837
## Prevalence             0.2845  0.19354  0.17434   0.1638  0.18386
## Detection Rate         0.2595  0.05981  0.09108   0.0000  0.08139
## Detection Prevalence   0.5215  0.12218  0.27426   0.0000  0.08207
## Balanced Accuracy      0.7730  0.61586  0.65028   0.5000  0.72093
```

---

### Conclusion and 20 test cases predictions

So, We can see the Accuracy of Random forest is about 99.49% and from Decision Tree is close to 49.18%.
About the time execution, the Training Random Forest take around 17 minutes. Decision Tree takes 6 seconds. But due the low accuracy of Decision Tree, and a good performance od random forest, I decided to keep RF to predict the 20 classe:




```r
predict(modFitRF,df_testing)
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```



