# This script first imputes the missing data using the "mice" package, 
# then trains a Random Forest model based on the data, and use the model 
# to predict the response to the test data

library(randomForest)
library(mice)

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
                "Parch",
                "SibSp",
                "Fare",
                "Embarked")
  fea <- data[,features]
  fea$Pclass<-as.factor(fea$Pclass)
  return(fea)
}
train<-extractFeatures(train0)
test<-extractFeatures(test0)

## Age under 16 seems to be a cutoff for youngsters (particularly for males)
#SurvByAge<-tapply(as.numeric(train0$Survived)-1, train0$Age, sum)
#NumByAge<-tapply(train0$Age, train0$Age, length)
#plot(SurvByAge/NumByAge, type="l")
#males.train<-subset(train, Sex=="male")
#SurvByAge.m<-tapply(as.numeric(males.train0$Survived)-1, males.train0$Age, sum)
#NumByAge.m<-tapply(males.train0$Age, males.train0$Age, length)
#plot(SurvByAge.m/NumByAge.m, type="l")
train$Young<-as.factor(ifelse(train0$Age<16, 1, 0))
train$Survived<-as.factor(train0$Survived)
test$Young<-as.factor(ifelse(test0$Age<16, 1, 0))

## Impute the missing data in the training set 
train.imp<-mice(train, m=5, print=FALSE, seed=10)
## stack the imputed data sets together
train.com<-complete(train.imp, 'long')

## Train a Random Forest model
rf <- randomForest(train.com[,c(3:9)], train.com$Survived, ntree=100, importance=TRUE)
## Plot the importance of variables
varImpPlot(rf)

## Impute the missing data in the testing set
test.imp<-mice(test, m=5, print=FALSE, seed=10)
## stack the imputed data sets together
test.com<-complete(test.imp, 'long')

## Predict the survival of the testing set
submission <- data.frame(PassengerId = test0$PassengerId)
test.com$surv.pred<-predict(rf, test.com)
## Based on the predictions on different imputed sets, the majority is chosen as the final prediction
sum.surv.pred<-tapply(as.numeric(test.com$surv.pred)-1, test.com$.id, sum)
sum.surv.pred<-sum.surv.pred[order(as.integer(names(sum.surv.pred)))]
submission$Survived<-ifelse(sum.surv.pred>test.imp$m/2, 1, 0)
write.csv(submission, file = "mice_RandomForest_r_submission.csv", row.names=FALSE)