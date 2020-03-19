import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

dataset=pd.read_csv('dataset.csv')

X=dataset.iloc[:,0:1].values
Y=dataset.iloc[:,1].values

lin_reg=LinearRegression()
lin_reg.fit(X,Y)

poly_reg= PolynomialFeatures(degree= 3)
X_poly = poly_reg.fit_transform(X)

lin_reg2=LinearRegression()
lin_reg2.fit(X_poly,Y)

X_grid=np.arange(min(X),max(X),0.1)
X_grid=X_grid.reshape((len(X_grid),1))
plt.scatter(X,Y, color='red')
plt.scatter(50,lin_reg.predict([[50]]),s=150)
plt.plot(X_grid,lin_reg2.predict(poly_reg.fit_transform(X_grid)))
plt.title('Total Cases')
plt.xlabel('Days')
plt.ylabel('Number of patients')
plt.show()

lin_reg2.predict(poly_reg.transform([[1000]]))
print('Predicted Number of Cases in next 50 days would be : ',lin_reg.predict([[50]]))
