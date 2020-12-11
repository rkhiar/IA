## Project structure :
the project is structured in two main parts :

### analytics : 

- Answering the queries corresponding to the basic question 1, 2 and 4.
- In order to demonstrate different data processing skills, the queries are written using (basic) Pandas DataFrames and SQL.
- The required visuals are produced using Matplotlib.

### Predictive modeling  : 
The work has been split into 3 parts:

#### Data set construction :
Row data is provided in a transaction format. in order to perform the monthly sales predictive analysis, it has been aggregated based on this time dimension.
The logical approach of the data set construction is the following :

To predict the transactions for the next three months  per customer at any time in 2019 for , the training will be based on a sequece of the last sliding year of data.

ex : 
Prediction of 201901--201903 sales will be based on  201801--201812
Prediction of 201903--201906 sales will be based on  201803--201902......

Using Sql queries, data has been formatted this way.

Because of this transaction format of the row data, the monthly aggregation will produce data only for months with sales. (For instance, if a customer doesn't have any transaction in 201804, we will not have data for this month/customer). These missing months have been added with a number of transactions equal to 0. The fact that a customer doesn't buy anything in a particular month is as important as if he does. 
This way, we'll produce a full twelve months sequence.

Data is finally split in train and test.


#### model development : 
The problem faced is a sequential regression one.
Since it is a univariate case, it can be solved using basic regression machine learning models or using a Recurrent Neural Network.

Model applied are : 
- Linear Regression.
- Support Vector Regression.
- Gradient Boosting Regression.
- Recurrent Neural Network.

These models were scored using Rsquared metric (appropriate one for regression case).
Gradient Boosting Regressor gives the best results with test R2 = 0.79. It has been saved for an evaluation script to load it and make prediction.

