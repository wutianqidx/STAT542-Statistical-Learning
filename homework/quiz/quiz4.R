library(rpart)
library(randomForest)
load("/Users/wtq/Desktop/fa18/stat542/homework/BostonHousing1.rdata")
dim(Housing1)
myfit = rpart(Y~.,data = Housing1)
printcp(myfit)

myfit2 = prune(myfit,cp=0.02)
printcp(myfit2)

set.seed(100)
mrf = randomForest(Y~.,data=Housing1,ntree=500)
mrf$predicted
