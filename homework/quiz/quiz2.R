#1
data =read.csv('/Users/wtq/Desktop/train_housing_quiz2.csv')
length(data)
length(data[,1])
summary(data[,'SalePrice'])
length(which(data[,'SalePrice']>500000))
length(which(data[,'GarageCars']>3))

#2
model = lm(log(SalePrice+1)~OverallQual+log(X1stFlrSF+1)+log(GrLivArea+1)+GarageCars+log(GarageArea+1),data=data)
summary(model)
new = data.frame(OverallQual = 6, X1stFlrSF = 1087, GrLivArea = 1464,GarageCars = 2,GarageArea = 480)
exp(predict(model, new))-1

b0 = 50; b1 = 20; b2 = 0.07;
b3 = 35; b4 = 0.01;  b5= -10
IQ = 100 ; GPA = 4.0 ; Gender = 1 
x = b0 + GPA*b1 + IQ*b2 + Gender*b3 +  GPA*IQ*b4 + GPA*Gender*b5