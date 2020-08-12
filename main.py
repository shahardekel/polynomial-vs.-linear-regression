import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from numpy.linalg import inv


# A
x_train = np.array([[0], [0.5], [2]])
y_train = [-1.25, -0.6, -4.85]

x_test = np.array([[-1], [1], [3]])
y_test = [-5.2, -0.9, -13]

# B
lr_model = LinearRegression()
lr_model.fit(x_train, y_train)
w1 = lr_model.coef_
w0 = lr_model.intercept_
print('intercept w0=', w0)
print('coefficient w1=', w1)

y_test_pred = lr_model.predict(x_test)
y_train_pred = lr_model.predict(x_train)
print('predicted values for x_test',y_test_pred)

# C
regular_train_MSE = mean_squared_error(y_train, y_train_pred)
regular_test_MSE = mean_squared_error(y_test, y_test_pred)

print('Train MSE:', regular_train_MSE)
print('Test MSE:', regular_test_MSE)

"""
*What can you conclude from these values?
The average distance between the data values and the values that we predicted in the training group is small,
so the error of straightness will be small.
In contrast, in the test group, the mean distance is large,
so the error will be greater between the line we draw and the actual values.
"""


# D
def least_squares_estimator(explanatoryX, responsesY):
    rowsX = len(explanatoryX)
    x1 = np.ones((rowsX, 1))
    Xnew = np.hstack((x1, explanatoryX))
    XnewT = np.transpose(Xnew)
    xTx = np.matmul(XnewT, Xnew)

    xTx_inverse = inv(xTx)
    xTy = np.matmul(XnewT, responsesY)

    result = np.matmul(xTx_inverse, xTy)

    return result


# E
x = np.linspace(-3, 5,100)
y = w1 * x + w0
plt.plot(x, y, '--', color='black', label='Regression Line')
plt.scatter(x_train, y_train, marker='*', label='Train Data')
plt.scatter(x_test, y_test, marker='o', label='Test Data')
plt.xlabel('X axis')
plt.ylabel('Y axis')

"""
The training group data seems to fit the regression line, but not the test group
"""

# F
# we will assume now that 𝑦𝑖 ≈ 𝛾0 + 𝛾1 * 𝑥𝑖 + 𝛾2 * 𝑥𝑖^2
# 1.  we would mark 𝑧𝑖 = transpose([𝑥𝑖, 𝑥𝑖^2]), so 𝑦𝑖 = 𝛾0+[𝛾1,𝛾2]*𝑧𝑖 (𝑦𝑖 is now linear)
# 2.
z_train=np.hstack((x_train,np.power(x_train,2)))
print()
print(z_train)
z_test=np.hstack((x_test,np.power(x_test,2)))
print()
print(z_test)
# 3.
LS_estimators=least_squares_estimator(z_train,y_train)
print()
print('the LS estimators gama are:', LS_estimators)
gama0=LS_estimators[0]
gama1=LS_estimators[1]
gama2=LS_estimators[2]

pr_model = LinearRegression()
pr_model.fit(z_train, y_train)
yz_train_pred = pr_model.predict(z_train)
yz_test_pred = pr_model.predict(z_test)
z_train_MSE = mean_squared_error(y_train, yz_train_pred)
z_test_MSE = mean_squared_error(y_test, yz_test_pred)

print('z_Train MSE:', z_train_MSE)
print('z_Test MSE:', z_test_MSE)

# 4.
z_x1=np.linspace(-3,5,100)
z_y=gama0+gama1*z_x1+gama2*(z_x1**2)
plt.plot(z_x1,z_y,'--',color='red', label='Polynomial Regression')
plt.title('Polynomial Regression vs. Linear Regression')
plt.legend()
plt.show()

"""the polynomial regression seems to perform better on the data"""

#G
"""
We did not assume anything about the linearity of y as a function of x,
we only tried to fit a line to the points - so the linear regression failed.
In contrast, in the polynomial model we assumed this and therefore the polynomial model
was more in line with the data we collected.
"""