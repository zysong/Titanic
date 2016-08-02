# This script first imputes the missing data using the "mice" package, 
# then fulls a Random Forest model based on the data, and use the model 
# to predict the response to the test data

library(randomForest)
library(mice)

train0 <- read.csv("train.csv",  na.strings = "")
test0  <- read.csv("test.csv",  na.strings = "")
n.train<-nrow(train0)
n.test<-nrow(test0)
full0 <- rbind(train0[, -2], test0)

## Check the pattern of missing data in the fulling and testing data sets 
md.pattern(train0)
md.pattern(test0)
## Cabin turns out to include a high percentage of missing data, so we drop this column. 

## Based on the patterns of the data sets, some columns are chosen to full the model
extractFeatures <- function(data) {
  features <- c("Pclass",
                "Sex",
                "Age",
                "Parch",
                "SibSp",
                "Fare",
                "Embarked")
  fea <- data[,features]
  fea$Pclass<-as.factor(fea$Pclass)
  return(fea)
}
full<-extractFeatures(full0)
full$Fare<-full$Fare/(full$Parch+full$SibSp+1)

## Add a column of title, extracted from name
full$Title<-NA
full$Title[grep("Mr.", full0$Name, fixed = TRUE)]<-"Mr"
full$Title[grep("Miss.", full0$Name, fixed = TRUE)]<-"Miss"
full$Title[grep("Master.", full0$Name, fixed = TRUE)]<-"Master"
full$Title[grep("Mrs.", full0$Name, fixed = TRUE)]<-"Mrs"
full$Title[grep("Dr.", full0$Name, fixed = TRUE)]<-"Mro"
full$Title[grep("Rev.", full0$Name, fixed = TRUE)]<-"Mro"
full$Title[grep("Don.", full0$Name, fixed = TRUE)]<-"Mro"
full$Title[grep("Mme.", full0$Name, fixed = TRUE)]<-"Mrs"
full$Title[grep("Ms.", full0$Name, fixed = TRUE)]<-"Miss"
full$Title[grep("Major.", full0$Name, fixed = TRUE)]<-"Mro"
full$Title[grep("Mlle.", full0$Name, fixed = TRUE)]<-"Miss"
full$Title[grep("Col.", full0$Name, fixed = TRUE)]<-"Mro"
full$Title[grep("Capt.", full0$Name, fixed = TRUE)]<-"Mro"
full$Title[grep("Countess.", full0$Name, fixed = TRUE)]<-"Miss"
full$Title[grep("Jonkheer.", full0$Name, fixed = TRUE)]<-"Mr"
full$Title[grep("Lady.", full0$Name, fixed = TRUE)]<-"Mrs"
full$Title[grep("Dona.", full0$Name, fixed = TRUE)]<-"Mrs"
full$Title[grep("Sir.", full0$Name, fixed = TRUE)]<-"Mro"
full$Title<-as.factor(full$Title)

full$Young<-as.factor(ifelse(full0$Age<16, 1, 0))
full$Young[full$Title=="Mrs"]<-0
full$Young[full$Title=="Mro"]<-0
full$Young[full$Title=="Master"]<-1
full$Young[full$Parch==0&is.na(full$Young)]<-0

full$Mother<-0
full$Mother[full$Title=='Mrs'&full$Parch>0]<-1

## Impute the missing data in the fulling set 

full.imp<-mice(full, m=5, print=FALSE, seed=10)
## stack the imputed data sets together
full.com<-complete(full.imp, 'long')


##split the full data set into train and test
train.com<-subset(full.com, as.integer(as.character(.id))<=n.train)
test.com<-subset(full.com, as.integer(as.character(.id))>n.train)

## train a Random Forest model
train.com$Survived<-as.factor(rep(train0$Survived, full.imp$m))
rf <- randomForest(train.com[,c(3:12)], train.com$Survived, ntree=100, importance=TRUE)
## Plot the importance of variables
varImpPlot(rf)

## Predict the survival of the testing set
submission <- data.frame(PassengerId = test0$PassengerId)
test.com$surv.pred<-predict(rf, test.com)
## Based on the predictions on different imputed sets, the majority is chosen as the final prediction
sum.surv.pred<-tapply(as.numeric(test.com$surv.pred)-1, as.integer(as.character(test.com$.id)), sum)
#sum.surv.pred<-sum.surv.pred[order(as.integer(names(sum.surv.pred)))]
submission$Survived<-ifelse(sum.surv.pred>full.imp$m/2, 1, 0)
write.csv(submission, file = "mice_RandomForest_r_submission.csv", row.names=FALSE)
