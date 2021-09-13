
#------------------------------------------
## Clear  workspace
#-----------------------------------

rm(list=ls())

#------------------------------------------ 
## Set wd
#-----------------------------------

setwd("/Users/tanmaymehta/Documents")

#-----------------------------------
## Load libraries
#-----------------------------------
library(caret)
library(rpart)
library(rpart.plot)
library(randomForest)
library(caretEnsemble) # training multiple models, custom ensembles
library(parallel)
library(doParallel)
library(fastAdaboost)
library(ipred)


#-----------------------------------
## Load Data
#-----------------------------------

Cust_Churn_Initial <- read.csv(file = "CustomerChurn.csv",
                               stringsAsFactors = FALSE)

#-----------------------------------
## Data Exploration & Preparation
#-----------------------------------

# View high-level information
# about the dataframe (structure and sample of data)
str(Cust_Churn_Initial)
head(Cust_Churn_Initial)

## Prepare Target (Y) Variable
Cust_Churn_Initial$Churn <- factor(Cust_Churn_Initial$Churn) #convert target variable to factor

# Create a barplot of target variable 
plot(Cust_Churn_Initial$Churn,
     main = "Churn")

#-----------------------------------
## Prepare Predictor (X) Variables
#-----------------------------------

#-----------------------------------
## Categorical
#-----------------------------------

# Nominal (Unordered) Factor Variables
noms <- c("gender", "SeniorCitizen", "Partner", "Dependents", "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity", 
          "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies", "PaperlessBilling", "PaymentMethod")

Cust_Churn_Initial[ ,noms] <- lapply(X = Cust_Churn_Initial[ ,noms], 
                                     FUN = factor) #convert to nomical categorical factors

#-----------------------------------
# Ordinal (Ordered) Factor Variables
#-----------------------------------

Cust_Churn_Initial$Contract <- factor(x = Cust_Churn_Initial$Contract,
                                      ordered = TRUE) #convert to an ordinal categorical factor

ords <- c("Contract")

#-----------------------------------
#Numeric variables
#-----------------------------------
## Numeric and tenure (int.)
nums <- c("tenure", "MonthlyCharges", "TotalCharges")

#  combine the 3 vectors to create a 
# vector of all  
# predictor variables 
vars <- c(noms, ords, nums)

#summary informaton for
# prepared data
summary(Cust_Churn_Initial[,c(vars, "Churn")])

#-----------------------------------
#Missing Values
#-----------------------------------

any(is.na(Cust_Churn_Initial)) #check for missing values
Cust_Churn_Cleaned<- na.omit(Cust_Churn_Initial) #omit missing data
any(is.na(Cust_Churn_Cleaned)) #double-check

Cust_Churn_Cleaned<- Cust_Churn_Cleaned[-1] #remove Customer ID variable (irrelevant)

#-----------------------------------
#Spliting training & testing data
#-----------------------------------

# Initialize random seed
set.seed(527729110) 

# Create list of training indices
sub <- createDataPartition(y = Cust_Churn_Cleaned$Churn, # target variable
                           p = 0.85, # % in training
                           list = FALSE)

# Subset the transformed data
# to create the training (train)
# and testing (test) datasets
train <- Cust_Churn_Cleaned[sub, ] # create train dataframe
test <- Cust_Churn_Cleaned[-sub, ] # create test dataframe


#Check for class imbalance
summary(train$Churn) 

#bar plot to view the
# distribution of target variable
plot(train$Churn,
     main = "Churn") 

#-----------------------------------
# Base Model (with class imbalance)
# Train a Decision Tree model
#-----------------------------------

ctrl_DT <- trainControl(method = "repeatedcv",
                        number = 5, #5-fold cross validation
                        repeats = 3) #repeated 3 times

#-----------------------------------
#Base Decision Tree
#-----------------------------------


set.seed(527729110) # initialize random seed

DTFit_base <- train(x = train[,vars],
                    y = train$Churn, #training data
                    method = "rpart",
                    trControl = ctrl_DT, # control object
                    tuneLength = 5) # try 5 cp values



# View the results of  tuned
# model with class imbalance
DTFit_base

#-----------------------------------
## Class Weighting
#-----------------------------------

target_var <- train$Churn # identify target variable

#Find class weights
weights <- c(sum(table(target_var))/(nlevels(target_var)*table(target_var)))
weights

# If case-weights are needed
wghts <- weights[match(x = target_var, 
                       table = names(weights))]

# Remove the sampling argument
# of the ctrl_DT object so no resampling 
# of the data is used

ctrl_DT$sampling <- NULL


#-----------------------------------
#Weighted Decision Tree
#-----------------------------------

set.seed(527729110) # initialize random seed

weight.training.rpart <- rpart(formula = Churn ~ ., # Y ~ all other variables in dataframe
                               data = train[ ,c(vars, "Churn")], # include only relevant variables
                               method = "class", 
                               weights = wghts)

weight.training.rpart$variable.importance #variable importance
weight.training.rpart

#-----------------------------------
## Tree Plot
#-----------------------------------

rpart.plot(x= weight.training.rpart, extra=101, cex = .8, box.palette = "GnYlRd", #color code from green to red (non churn to churn)
           branch.lty = 1, split.cex = .95, 
           split.prefix = "is ", # put "is " before split text
           split.suffix = "?", # put "?" after split text
           split.box.col = "beige") #textbox color

#-----------------------------------
#Training Performance
#-----------------------------------

DTFit_cw <- train(x = train[ ,vars], 
                  y = train$Churn, #training data
                  method = "rpart", 
                  trControl = ctrl_DT, # control object
                  tuneLength = 5, # try 5 cp values
                  weights = wghts) # weighted model
DTFit_cw


#-----------------------------------
#Testing Performance
#-----------------------------------

CW.preds <- predict(object = DTFit_cw, #testing set, weighted method
                    newdata = test)

CW_conf <- confusionMatrix(data = CW.preds, # predictions
                           reference = test$Churn, # testing data
                           positive = "Yes",
                           mode = "everything")

CW_conf


#-----------------------------------
#random forest
#-----------------------------------

# Initialize random seed
set.seed(527729110) 

# Create list of training indices
sub <- createDataPartition(y = Cust_Churn_Cleaned$Churn, # target variable
                           p = 0.85, # % in training
                           list = FALSE)

# Subset the transformed data
# to create the training (train)
# and testing (test) datasets
train <- Cust_Churn_Cleaned[sub, ] # create train dataframe
test <- Cust_Churn_Cleaned[-sub, ] # create test dataframe


floor(sqrt(length(vars))) #model hyperparameter (m variable): find number of random predictors to use splits on

# Perform a grid search, searching from 2 variables to the total number of predictors
# (if mtry = length(vars), bagging)
grids = expand.grid(mtry = seq(from = 2, 
                               to = length(vars), 
                               by = 1))
grids

#5-fold cross validation, repeated 3 times and specify
# search = "grid" for a grid search
grid_ctrl <- trainControl(method = "repeatedcv",
                          number = 5, # 5-fold cross validation
                          repeats = 3, #repeat 3 times
                          search="grid")

base_RF <- train(x = train[ ,vars], 
                 y = train$Churn, #training data
                 method = "rf", #random forest
                 trControl = grid_ctrl,
                 tuneGrid = grids) 


# Initialize a random seed for cross validation
set.seed(527729110)

#weighted random forest model
weighted_fit.rf <- train(x = train[ ,vars], 
                         y = train$Churn, #training data
                         method = "rf", #random forest 
                         trControl = grid_ctrl,
                         tuneGrid = grids, 
                         weights=wghts) #weighted model


# View the results of  cross validation across  grid
# of mtry values for for Accuracy and Kappa.

weighted_fit.rf #mtry = 2, accuracy= 80.19%, kappa= 0.4490

# Use the best fitting model performance to compare to  testing performance
confusionMatrix(weighted_fit.rf)   

# Variable Importance
plot(varImp(weighted_fit.rf))

#-----------------------------------
### Training Performance
#-----------------------------------

tune.tr.preds <- predict(object = weighted_fit.rf,
                         newdata = train)

# obtain a confusion matrix and obtain performance
# measures for our model applied to the training dataset (train).
RF_trtune_conf <- confusionMatrix(data = tune.tr.preds, # predictions
                                  reference = train$Churn, # training data
                                  positive = "Yes",
                                  mode = "everything")
RF_trtune_conf


#-----------------------------------
### Testing Performance
#-----------------------------------

# generate class predictions for our testing data set
tune.te.preds <- predict(object = weighted_fit.rf,
                         newdata = test)

# Confusion matrix and  performance
# measures for  model applied to the testing dataset (test).
RF_tetune_conf <- confusionMatrix(data = tune.te.preds, # predictions
                                  reference = test$Churn, # testing data
                                  positive = "Yes",
                                  mode = "everything")
RF_tetune_conf

# Overall comparison
cbind(Training = RF_trtune_conf$overall,
      Testing = RF_tetune_conf$overall)

# Class-Level comparison
cbind(Training = RF_trtune_conf$byClass,
      Testing = RF_tetune_conf$byClass)

#-----------------------------------
######### Model comparison #########
#-----------------------------------

#Overall comparison
cbind(DT = CW_conf$overall, 
      RF = RF_tetune_conf$overall)

#Class Level comparison
cbind(DT = CW_conf$byClass,
      RF = RF_tetune_conf$byClass)


#-----------------------------------
