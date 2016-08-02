# This script trains a gradient boosting model based on the data, and use the model 
# to predict the response to the test data

library(xgboost)
library(mice)
library(caret)

train0 <- read.csv("train.csv",  na.strings = "")
test0  <- read.csv("test.csv",  na.strings = "")

## Check the pattern of missing data in the training and testing data sets 
md.pattern(train0)
md.pattern(test0)
## Cabin turns out to include a high percentage of missing data, so we drop this column. 

## Based on the patterns of the data sets, some columns are chosen to train the model
extractFeatures <- function(data) {
  features <- c("Pclass",
                "Sex",
                "Age",
                "Parch",
                "SibSp",
                "Fare",
                "Embarked"
  )
  fea <- data[,features]
  fea$Pclass<-as.factor(fea$Pclass)
  return(fea)
}
xgbTrain<-extractFeatures(train0)
#xgbTrain$Survived<-as.factor(train0$Survived)
xgbTest<-extractFeatures(test0)

## Add a column of title, extracted from name
setTitle<- function(dataset.new, dataset.old){
  dataset.new$Title<-NA
  dataset.new$Title[grep("Mr.", dataset.old$Name, fixed = TRUE)]<-"Mr"
  dataset.new$Title[grep("Miss.", dataset.old$Name, fixed = TRUE)]<-"Miss"
  dataset.new$Title[grep("Master.", dataset.old$Name, fixed = TRUE)]<-"Master"
  dataset.new$Title[grep("Mrs.", dataset.old$Name, fixed = TRUE)]<-"Mrs"
  dataset.new$Title[grep("Dr.", dataset.old$Name, fixed = TRUE)]<-"Mro"
  dataset.new$Title[grep("Rev.", dataset.old$Name, fixed = TRUE)]<-"Mro"
  dataset.new$Title[grep("Don.", dataset.old$Name, fixed = TRUE)]<-"Mro"
  dataset.new$Title[grep("Mme.", dataset.old$Name, fixed = TRUE)]<-"Mrs"
  dataset.new$Title[grep("Ms.", dataset.old$Name, fixed = TRUE)]<-"Miss"
  dataset.new$Title[grep("Major.", dataset.old$Name, fixed = TRUE)]<-"Mro"
  dataset.new$Title[grep("Mlle.", dataset.old$Name, fixed = TRUE)]<-"Miss"
  dataset.new$Title[grep("Col.", dataset.old$Name, fixed = TRUE)]<-"Mro"
  dataset.new$Title[grep("Capt.", dataset.old$Name, fixed = TRUE)]<-"Mro"
  dataset.new$Title[grep("Countess.", dataset.old$Name, fixed = TRUE)]<-"Miss"
  dataset.new$Title[grep("Jonkheer.", dataset.old$Name, fixed = TRUE)]<-"Mr"
  dataset.new$Title[grep("Lady.", dataset.old$Name, fixed = TRUE)]<-"Mrs"
  dataset.new$Title[grep("Dona.", dataset.old$Name, fixed = TRUE)]<-"Mrs"
  dataset.new$Title[grep("Sir.", dataset.old$Name, fixed = TRUE)]<-"Mro"
  dataset.new$Title<-as.factor(dataset.new$Title)
  return(dataset.new)
}

xgbTrain<- setTitle(xgbTrain, train0)
xgbTest<- setTitle(xgbTest, test0)

train.dummy<-dummyVars(~., data=xgbTrain)
train1<-predict(train.dummy, newdata = xgbTrain)
test1<-predict(train.dummy, newdata = xgbTest)

## Impute the missing data in the training set 
#train.imp<-mice(xgbTrain, m=10, print=FALSE, seed=10)
## stack the imputed data sets together
#train.com<-complete(train.imp, 'long')

## Train a gradient boost model
set.seed(100)
indx <- createFolds(train0$Survived, returnTrain = TRUE)
ctrl <- trainControl(method = "cv", index = indx)

xgbGrid <- expand.grid(nrounds = seq(10, 200, by=10),
                       eta = .1, 
                       max_depth = c(3, 4, 5, 6),
                       gamma = .01,
                       colsample_bytree = 1,
                       min_child_weight = 1)

xgbTune <- train(train1, as.factor(train0$Survived), 
            method = "xgbTree", tuneGrid = xgbGrid,
            trControl = ctrl)

plot(xgbTune, auto.key= list(columns = 2, lines = TRUE))
xgb.params<-xgbTune$bestTune
## Plot the importance of variables
varImp(xgbTune)

## Impute the missing data in the testing set
#test.imp<-mice(test, m=10, print=FALSE, seed=10)
## stack the imputed data sets together
#test.com<-complete(test.imp, 'long')

## Predict the survival of the testing set
pred.test <- data.frame(PassengerId = test0$PassengerId)
Surv_prob<-predict(xgbTune$finalModel, test1)
pred.test$Survived<-as.integer(Surv_prob<.5)
write.csv(pred.test, file = "Xgb_r_submission.csv", row.names=FALSE)
