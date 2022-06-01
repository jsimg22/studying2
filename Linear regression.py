#Linear regression

from statistics import linear_regression
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression 

perch_length = np.array([8.4, 13.7, 15.0, 16.2, 17.4, 18.0, 18.7, 19.0, 19.6, 20.0, 21.0,
       21.0, 21.0, 21.3, 22.0, 22.0, 22.0, 22.0, 22.0, 22.5, 22.5, 22.7,
       23.0, 23.5, 24.0, 24.0, 24.6, 25.0, 25.6, 26.5, 27.3, 27.5, 27.5,
       27.5, 28.0, 28.7, 30.0, 32.8, 34.5, 35.0, 36.5, 36.0, 37.0, 37.0,
       39.0, 39.0, 39.0, 40.0, 40.0, 40.0, 40.0, 42.0, 43.0, 43.0, 43.5,
       44.0])
perch_weight = np.array([5.9, 32.0, 40.0, 51.5, 70.0, 100.0, 78.0, 80.0, 85.0, 85.0, 110.0,
       115.0, 125.0, 130.0, 120.0, 120.0, 130.0, 135.0, 110.0, 130.0,
       150.0, 145.0, 150.0, 170.0, 225.0, 145.0, 188.0, 180.0, 197.0,
       218.0, 300.0, 260.0, 265.0, 250.0, 250.0, 300.0, 320.0, 514.0,
       556.0, 840.0, 685.0, 700.0, 700.0, 690.0, 900.0, 650.0, 820.0,
       850.0, 900.0, 1015.0, 820.0, 1100.0, 1000.0, 1100.0, 1000.0,
       1000.0])


train_input,test_input, train_target,test_target = train_test_split(perch_length,perch_weight, random_state=42)  #train, test set split 

train_input=train_input.reshape(-1,1)
test_input=test_input.reshape(-1,1)

lr=LinearRegression()

lr.fit(train_input,train_target)
print(lr.predict([[50]]))
print(lr.coef_,lr.intercept_)

print("train set r2 score {0} ".format(lr.score(train_input,train_target)))
print("test set r2 score {0} ".format(lr.score(test_input,test_target)))


plt.scatter(train_input,train_target)
plt.plot([15,50],[15*lr.coef_+lr.intercept_,50*lr.coef_+lr.intercept_]) #1차 방정식 그래프
plt.scatter(50,1241,marker="^")
plt.xlabel("length")
plt.ylabel("weight")
plt.show()

#다항 회귀 
train_poly=np.column_stack((train_input**2,train_input)) #first column : x^2 , second COLUMN : X 
test_poly=np.column_stack((test_input**2,test_input))
#print(train_poly)
lr2=LinearRegression()
lr2.fit(train_poly,train_target)
print("coef :{0} , intercept : {1} ".format(lr2.coef_,lr2.intercept_))
print("다항회귀 방정식 : weight = {0}x lenght^2 + {1}x length + {2}".format(lr2.coef_[0],lr2.coef_[1],lr2.intercept_))


print("train poly set r2 score {0} ".format(lr2.score(train_poly,train_target)))
print("test poly set r2 score {0} ".format(lr2.score(test_poly,test_target)))

point=np.arange(15,50)
plt.scatter(train_input,train_target)
plt.plot(point, 1.01*point**2 - 21.6*point + 116.05) 
plt.scatter(50,1574,marker="^")
plt.xlabel("length")
plt.ylabel("weight")
plt.show()

