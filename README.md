# Slips

Q1) Write a Python program to prepare Scatter Plot for Iris Dataset?
Output: -
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
data=pd.read_csv("iris.csv")
print(data)
x=data["SepalLengthCm"]
y=data["SepalWidthCm"]
print(x)
print(y)
plt.scatter(x,y,c="red")
plt.title("IRIS CSV")
plt.xlabel("SepalLengthCm")
plt.ylabel("SepalWidthCm")
plt.show()
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
data=pd.read_csv("iris.csv")
print(data)
x=data["SepalLengthCm"]
y=data["SepalWidthCm"]
print(x)
print(y)
plt.scatter(x,y,c="red")
plt.title("IRIS CSV")
plt.xlabel("SepalLengthCm")
plt.ylabel("SepalWidthCm")
plt.show()
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
data=pd.read_csv("iris.csv")
print(data)
x=data["SepalLengthCm"]
y=data["SepalWidthCm"]
print(x)
print(y)
plt.scatter(x,y,c="red")
plt.title("IRIS CSV")
plt.xlabel("SepalLengthCm")
plt.ylabel("SepalWidthCm")
plt.show()
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
data=pd.read_csv("iris.csv")
print(data)
x=data["SepalLengthCm"]
y=data["SepalWidthCm"]
print(x)
print(y)
plt.scatter(x,y,c="red")
plt.title("IRIS CSV")
plt.xlabel("SepalLengthCm")
plt.ylabel("SepalWidthCm")
plt.show()
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
data=pd.read_csv("iris.csv")
print(data)
x=data["SepalLengthCm"]
y=data["SepalWidthCm"]
print(x)
print(y)
plt.scatter(x,y,c="red")
plt.title("IRIS CSV")
plt.xlabel("SepalLengthCm")
plt.ylabel("SepalWidthCm")
plt.show()
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
data=pd.read_csv("iris.csv")
print(data)
x=data["SepalLengthCm"]
y=data["SepalWidthCm"]
print(x)
print(y)
plt.scatter(x,y,c="red")
plt.title("IRIS CSV")
plt.xlabel("SepalLengthCm")
plt.ylabel("SepalWidthCm")
plt.show()
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data=pd.read_csv(r"C:\Users\Admin\Desktop\csv_files\iris.csv")
print(data)

x=data["SepalLength"]
y=data["SepalWidth"]
print(x)
print(y)

plt.scatter(x,y,c="red")
plt.title("IRIS CSV")
plt.xlabel("SepalLength")
plt.ylabel("SepalWidth")
plt.show()

Q.2) Write a python program to find all null values in a given dataset and remove them.
Output: -
import pandas as pd
import numpy as np

data=pd.read_csv(r"C:\Users\Admin\Desktop\csv_files\employees.csv")
print(data)

print(data.isnull())

print(data.notnull())

data1=data.dropna(axis=0,how="any")
print(data1)

data["m1"]=data["m1"].replace(np.NaN,data["m1"].mean())
data["m2"]=data["m2"].replace(np.NaN,data["m2"].mean())
data["m3"]=data["m3"].replace(np.NaN,data["m3"].mean())
print(data)
Q.3) Write a python program to make Categorical values in numeric format for a given dataset
Output: -
import pandas as pd
import numpy as np

data=pd.read_csv(r"C:\Users\Admin\Desktop\csv_files\iris.csv")
print(data)
x=data.iloc[:,3].values
print(x)
from sklearn.preprocessing import LabelEncoder

le=LabelEncoder()
x1=le.fit_transform(x)
print(x1)
from sklearn.preprocessing import OneHotEncoder
ohe=OneHotEncoder()
xn=ohe.fit_transform(x).toarray()
print(xn)


Q.4) Write a python program to Implement Simple Linear Regression for predicting house price.
Output: -
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

data=pd.read_csv(r"C:\Users\Admin\Desktop\csv_files\houseprice.csv")
print(data)
x=data[["bedrooms","sqft_living"]]
y=data.price

print(x)
print(y)

xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.2)
print(xtrain)
print(xtest)
print(ytrain)
print(ytest)

lr=LinearRegression()
lr.fit(xtrain,ytrain)

print(lr.intercept_)
print(lr.coef_)

print(lr.predict([[2,1000]]))

ypred=lr.predict(xtest)
cm=mean_absolute_error(ytest,ypred)
print(cm)

Q.5 Write a python program to implement Multiple Linear Regression for given dataset
Output: -
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

data=pd.read_csv(r"C:\Users\Admin\Desktop\csv_files\houseprice.csv")
print(data)
x=data[["bedrooms","sqft_living"]]
y=data.price

print(x)
print(y)

xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.2)
print(xtrain)
print(xtest)
print(ytrain)
print(ytest)

lr=LinearRegression()
lr.fit(xtrain,ytrain)

print(lr.intercept_)
print(lr.coef_)

print(lr.predict([[2,1000]]))

ypred=lr.predict(xtest)
cm=mean_absolute_error(ytest,ypred)
print(cm)

Q.6) Write a python program to implement Polynomial Linear Regression for given dataset
Output: -
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
  
dataset = pd.read_csv(r"C:\Users\Admin\Desktop\csv_files\Position_Salaries.csv")
dataset
 
X = dataset.iloc[:,1:2].values  
y = dataset.iloc[:,2].values
 
# fitting the linear regression model
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X,y)
 
# visualising the linear regression model
plt.scatter(X,y, color='red')
plt.plot(X, lin_reg.predict(X),color='blue')
plt.title("Truth or Bluff(Linear)")
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()
 
# polynomial regression model
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=2)
X_poly = poly_reg.fit_transform(X)
  
X_poly     # prints X_poly
 
lin_reg2 = LinearRegression()
lin_reg2.fit(X_poly,y)
 
 
# visualising polynomial regression
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=4)
X_poly = poly_reg.fit_transform(X)
lin_reg2 = LinearRegression()
lin_reg2.fit(X_poly,y)
  
X_grid = np.arange(min(X),max(X),0.1)
X_grid = X_grid.reshape(len(X_grid),1) 
plt.scatter(X,y, color='red') 
  
plt.plot(X_grid, lin_reg2.predict(poly_reg.fit_transform(X_grid)),color='blue') 
  
plt.title("Truth or Bluff(Polynomial)")
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

Q.7) Write a python program to implement Naive Bayes.
Output: -
# importing libraries  
import numpy as nm
import matplotlib.pyplot as mtp
import pandas as pd

#importing datasets  
data_set=pd.read_csv(r'C:\Users\Admin\Desktop\CSVfile\suvdata.csv')
#Extracting Independent and dependent Variable  
x=data_set.iloc[:,[2,3]].values
y=data_set.iloc[:,4].values
print(x)
# Splitting the dataset into training and test set.  
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=0)
#feature Scaling  
from sklearn.preprocessing import StandardScaler
st_x= StandardScaler()
x_train=st_x.fit_transform(x_train)
x_test=st_x.transform(x_test)

from sklearn.svm import SVC # "Support vector classifier"  
classifier =SVC(kernel='linear',random_state=0)
classifier.fit(x_train,y_train)
#Predicting the test set result  
y_pred=classifier.predict(x_test)
print(y_pred)
#Creating the Confusion matrix  
from sklearn.metrics import confusion_matrix  
cm= confusion_matrix(y_test, y_pred)  
print(cm)
print(classifier.score(x_test,y_test))

Q8. Write a python program to implement Multiple Linear Regression for given dataset.
Output: -
#Predicting car price MLR
#importing libraries  
import numpy as nm  
import matplotlib.pyplot as mtp  
import pandas as pd
#importing datasets
df= pd.read_csv(r"C:\Users\Admin\Desktop\csv_files\CarPriceMultiR.csv")
x= df.iloc[:,9:13].values  # these are the 
y= df.iloc[:,25].values  #this is price colm
# Splitting the dataset into training and test set.  
from sklearn.model_selection import train_test_split  
x_train, x_test, y_train, y_test= train_test_split(x, y, test_size= 0.2, random_state=0)
#Fitting the MLR model to the training set:  
from sklearn.linear_model import LinearRegression  
regressor= LinearRegression()  
regressor.fit(x_train, y_train)
#Predicting all the Test set result;  
y_pred= regressor.predict(x_test)
print(x)
#To predict the result of first row in xtest
regressor.predict([x_test[0]])
#To predict with features in x
regressor.predict([[88.6, 168.8,  64.1,  48.8]])
#output is array([13153.12031111]
regressor.predict([[105.8, 192.7,71.4,55.9]])
regressor.predict([[109,188,68,55]])
regressor.predict([[109,189,69,56]])


Q.9. Write a python program to implement Decision Tree whether or not to play Tennis.
Output: -

