# polynomial-vs.-linear-regression
How to perform polynomial regression with linear regression tools using python


Given train data (𝑥1 = 0, 𝑦1 = −1.25), (𝑥2 = 0.5, 𝑦2 = −0.6), (𝑥3 = 2, 𝑦3 = −4.85)
and test data (𝑥4 = −1, 𝑦4 = −5.2), (𝑥5 = 1, 𝑦5 = −0.9), (𝑥6 = 3, 𝑦6 = −13):

a. Define a 3 × 1 two-dimensional matrix called X_train in which each row is an observation from the training data, and define a (one-dimensional) vector called y_train which contains the responses of these observations (in the same order). 
Repeat this process with the test data (X_test, y_test).

b. Calculate the regular LS estimators 𝑤̂0, 𝑤̂1 using only the training data with
sklearn built-in functions.
What are the predicted values for X_test?

c. What is the MSE of the regression on the train data?
What is the MSE of the regression on the test data?
What can you conclude from these values?

d. Write a function which receives a np-array of explanatory variables X and a np-array of responses Y and returns the least squares estimator using the closedform expression we saw in class.
There is no need to check that the input is valid. (don’t forget to add ones!)

e. Plot the regression line in a dashed (--) black line. Scatter (with plt.scatter()) the points in the train data with marker='*' and scatter the points in the test data with marker='o'. You should use legend (for train and test data) and label the axes. The range in the x axis should be np.arange(-3, 5).
Does it look like the regression fit the data?

f. We will now try to perform a 2nd degree polynomial regression, meaning we will assume now that 𝑦𝑖 ≈ 𝛾0 + 𝛾1 ∙ 𝑥𝑖 + 𝛾2 ∙ 𝑥𝑖^2

  1. If we would mark 𝑧𝑖 = (𝑥𝑖, 𝑥𝑖^2)^𝑇 , how can we write 𝑦𝑖 in a linear form?
  2. Define a 3 × 2 matrix called Z_train in which the first column corresponds to 𝑥𝑖 and the second column corresponds to 𝑥𝑖^2 for each    𝑥𝑖 in the train data (in the same order as in section a.). Repeat this process with the test data (Z_test).
  3. Use the function you wrote in section d. to calculate the LS estimators 𝛾̂ =(𝛾̂0, 𝛾̂1, 𝛾̂2)^𝑇 using only the training data (Z_train and y_train). What is the MSE of the regression on the train data? What is the MSE on the test data?
  4. Plot the corresponding 2nd degree polynomial function in a red dashed line, alongside the original regression line in a black dashed line. Scatter the points in the training data (with marker='*'), as well as the points in the testing data (with marker='o'). You should use legend for both data type (train/test) and regression type (linear/polynomial) and label the axes. The range in the x axis should be np.arange(-3, 5). Name the plot 'Polynomial Regression vs. Linear Regression'.
 Which regression seems to perform better?
 
 g. Which assumption did not hold, thus making the linear regression to fail?
