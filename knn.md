K Nearest Neighbors
================

## Background

The K Nearest Neighbors (KNN) algorithm predicts the labels of the test
dataset by looking at the labels of its closest neighbors in the feature
space of the training dataset. The following example will use it for a
classification problem.

## Dataset

The dataset can be found
[here](https://www.openintro.org/data/index.php?data=loans_full_schema).
It consists of 10,000 loans, and we will find whether a loan will be
paid back based on the customer’s data.

``` r
suppressPackageStartupMessages(library(tidyverse))

df <- read_csv('data/loans.csv', show_col_types = FALSE)
df <- subset(df, select = -c(loan_purpose))
head(df)
```

    ## # A tibble: 6 × 54
    ##   emp_title         emp_length state homeownership annual_income verified_income
    ##   <chr>                  <dbl> <chr> <chr>                 <dbl> <chr>          
    ## 1 global config en…          3 NJ    MORTGAGE              90000 Verified       
    ## 2 warehouse office…         10 HI    RENT                  40000 Not Verified   
    ## 3 assembly                   3 WI    RENT                  40000 Source Verified
    ## 4 customer service           1 PA    RENT                  30000 Not Verified   
    ## 5 security supervi…         10 CA    RENT                  35000 Verified       
    ## 6 <NA>                      NA KY    OWN                   34000 Not Verified   
    ## # ℹ 48 more variables: debt_to_income <dbl>, annual_income_joint <dbl>,
    ## #   verification_income_joint <chr>, debt_to_income_joint <dbl>,
    ## #   delinq_2y <dbl>, months_since_last_delinq <dbl>,
    ## #   earliest_credit_line <dbl>, inquiries_last_12m <dbl>,
    ## #   total_credit_lines <dbl>, open_credit_lines <dbl>,
    ## #   total_credit_limit <dbl>, total_credit_utilized <dbl>,
    ## #   num_collections_last_12m <dbl>, num_historical_failed_to_pay <dbl>, …

We will remove columns where there are a lot of missing (NA) or null
values

``` r
df %>%
  summarize_all(function(x){ sum(is.na(x)) })
```

    ## # A tibble: 1 × 54
    ##   emp_title emp_length state homeownership annual_income verified_income
    ##       <int>      <int> <int>         <int>         <int>           <int>
    ## 1       833        817     0             0             0               0
    ## # ℹ 48 more variables: debt_to_income <int>, annual_income_joint <int>,
    ## #   verification_income_joint <int>, debt_to_income_joint <int>,
    ## #   delinq_2y <int>, months_since_last_delinq <int>,
    ## #   earliest_credit_line <int>, inquiries_last_12m <int>,
    ## #   total_credit_lines <int>, open_credit_lines <int>,
    ## #   total_credit_limit <int>, total_credit_utilized <int>,
    ## #   num_collections_last_12m <int>, num_historical_failed_to_pay <int>, …

``` r
### Remove NA columns
df <- df %>%
  select(-c(emp_title, emp_length, annual_income_joint, debt_to_income, verification_income_joint, debt_to_income_joint, months_since_last_delinq, months_since_90d_late, months_since_last_credit_inquiry, num_accounts_120d_past_due))

### Create column to determine if fully paid or not
df$fully_paid <- ifelse(df$loan_status == 'Fully Paid', 1, 0)

### Remove non-numeric columns
df <- df %>% select_if(is.numeric)

### Change fully paid to a factor
df$fully_paid <- as.factor(df$fully_paid)
```

## Training and Test data

Now we split the data into test and training datasets

``` r
suppressPackageStartupMessages(library(caret))
set.seed(1000)

trainIndex <- createDataPartition(df$fully_paid, 
                                  times=1, 
                                  p = .8, 
                                  list = FALSE)
train <- df[trainIndex, ]
test <- df[-trainIndex, ]
```

## Feature scaling

We will now scale both the training and testing sets

``` r
preProcValues <- preProcess(train, method = c("center", "scale"))
trainTransformed <- predict(preProcValues, train)
testTransformed <- predict(preProcValues, test)
```

## Model

Now we can construct the model

``` r
knnModel <- train(
  fully_paid ~ ., 
  data = trainTransformed, 
    method = "knn", 
    trControl = trainControl(method = "cv"), 
    tuneGrid = data.frame(k = c(3,5,7))
  )
```

And choose the best k value

``` r
best_model<- knn3(fully_paid ~ .,
                  data = trainTransformed,
                  k = knnModel$bestTune$k)
```

## Model evaluation

Now we can evaluate the model using metrics given by the `caret` package

``` r
predictions <- predict(best_model, testTransformed, type = "class")
# Calculate confusion matrix
cm <- confusionMatrix(predictions, testTransformed$fully_paid)
cm
```

    ## Confusion Matrix and Statistics
    ## 
    ##           Reference
    ## Prediction    0    1
    ##          0 1901   28
    ##          1    9   61
    ##                                           
    ##                Accuracy : 0.9815          
    ##                  95% CI : (0.9746, 0.9869)
    ##     No Information Rate : 0.9555          
    ##     P-Value [Acc > NIR] : 1.802e-10       
    ##                                           
    ##                   Kappa : 0.7578          
    ##                                           
    ##  Mcnemar's Test P-Value : 0.003085        
    ##                                           
    ##             Sensitivity : 0.9953          
    ##             Specificity : 0.6854          
    ##          Pos Pred Value : 0.9855          
    ##          Neg Pred Value : 0.8714          
    ##              Prevalence : 0.9555          
    ##          Detection Rate : 0.9510          
    ##    Detection Prevalence : 0.9650          
    ##       Balanced Accuracy : 0.8403          
    ##                                           
    ##        'Positive' Class : 0               
    ## 

Our model is very good, with a high Balanced Accuracy value

Sensitivity measures the ratio of actual positive cases that are
correctly identified and our model has very high sensitivity
