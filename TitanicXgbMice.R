# This script trains a gradient boosting model based on the data, and use the model 
# to predict the response to the test data

library(xgboost)
library(mice)
library(caret)
library(plyr)
library(dplyr)


train0 <- read.csv("train.csv",  na.strings = "")
test0  <- read.csv("test.csv",  na.strings = "")
test0$Survived<-0
full0 <- rbind(train0, test0)
full0$Fare[full0$Fare==0]<-NA

## Check the pattern of missing data in the training and testing data sets 
#md.pattern(full0)
#md.pattern(train0)
#md.pattern(test0)
## Cabin turns out to include a high percentage of missing data, so we drop this column. 

#test with a simple tree model
#library(rpart)
#rpartTree<-rpart(Survived~Age+Sex, data=train0)

full_byTicket<-full0 %>% group_by(Ticket) 
FamilySize<-full_byTicket %>% summarise(FamilySize = n())
Survival_byFamily<- full_byTicket %>% summarise(SurvivedInFamily = sum(Survived))

full_Family<-full0 %>% left_join(Survival_byFamily) %>% left_join(FamilySize)
#count how many passengers have more family members on board than (Parch+SibSp)
#nrow(subset(full_Family, FamilySize>Parch+SibSp+1))
full_Family$SurvivedFamilyMember<-as.factor(as.numeric(full_Family$SurvivedInFamily-full_Family$Survived>0))
#Fare per person on the same ticket
full_Family$FarePP<-full_Family$Fare/full_Family$FamilySize

## Based on the patterns of the data sets, some columns are chosen to train the model
extractFeatures <- function(data) {
  features <- c("Pclass",
                "Sex",
                "Age",
                "Parch",
                "SibSp",
                "FarePP",
                "FamilySize",
                "SurvivedFamilyMember"
  )
  fea <- data[,features]
  fea$Pclass<-as.factor(fea$Pclass)
#  fea$LastName<-as.character(data$Name) %>% strsplit(",") %>% sapply(head, n=1) %>% as.factor()
  return(fea)
}

xgbFull<-extractFeatures(full_Family)
#xgbTrain<-extractFeatures(train0)
#xgbTrain$Survived<-as.factor(train0$Survived)
#xgbTest<-extractFeatures(test0)

## Add a column of title, extracted from name
getTitle<- function(dataset.new, dataset.old){
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
  dataset.new$Title[grep("Countess.", dataset.old$Name, fixed = TRUE)]<-"Mrs"
  dataset.new$Title[grep("Jonkheer.", dataset.old$Name, fixed = TRUE)]<-"Mr"
  dataset.new$Title[grep("Lady.", dataset.old$Name, fixed = TRUE)]<-"Mrs"
  dataset.new$Title[grep("Dona.", dataset.old$Name, fixed = TRUE)]<-"Mrs"
  dataset.new$Title[grep("Sir.", dataset.old$Name, fixed = TRUE)]<-"Mro"
  dataset.new$Title<-as.factor(dataset.new$Title)
  return(dataset.new)
}

xgbFull<- getTitle(xgbFull, full0)
#xgbTrain$Missing<-as.factor(is.na(xgbTrain$Age))
#xgbTest$Missing<-as.factor(is.na(xgbTest$Age))

## Impute the missing data in the training set 
xgbFull.ext<-cbind(xgbFull, Parch2 = NA)
init.mice<-mice(xgbFull.ext, max=0, print=FALSE)
meth<-init.mice$method
meth["Parch2"]<-"~I(Parch^2)"
pred<-init.mice$predictorMatrix
full.imp<-mice(xgbFull.ext, m=1, print=FALSE, seed=100, method=meth, predictorMatrix = pred)
## stack the imputed data sets together
full.com<-complete(full.imp)

full.dummy<-dummyVars(~., data=full.com, sparse=TRUE)
full1<-as.data.frame(predict(full.dummy, newdata = full.com))
train1<-full1[1:nrow(train0),]
test1<-full1[-(1:nrow(train0)),]

## Train a gradient boost model
set.seed(100)
indx <- createFolds(train0$Survived, returnTrain = TRUE)
ctrl <- trainControl(method = "repeatedcv", repeats = 5)

xgbGrid <- expand.grid(nrounds = seq(10, 50, by=5),
                       eta = .1, 
                       max_depth = seq(4, 7),
                       gamma = .1,
                       colsample_bytree = seq(.7, .9, by =.1),
                       min_child_weight = seq(.2, 1, by=.2))

xgbTune <- train(train1, as.factor(train0$Survived), 
                 method = "xgbTree", 
                 tuneGrid = xgbGrid,
                 trControl = ctrl)

plot(xgbTune, auto.key= list(columns = 2, lines = TRUE))
xgbTune$bestTune
## Plot the importance of variables
varImp(xgbTune)

## Predict the survival of the testing set
pred.test <- data.frame(PassengerId = test0$PassengerId)
Surv_prob<-predict(xgbTune$finalModel, as.matrix(test1))
pred.test$Survived<-as.integer(Surv_prob<.5)
write.csv(pred.test, file = "Xgb_r_submission.csv", row.names=FALSE)
