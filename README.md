# Final Project
CSCI 297A

## Authors:
- Virginia Weston
- Tina Jin
- Jeff Bradley
- Taylor Tucker

## Data Cleaning
The data we are using comes from the National Center for Education Statistics (https://nces.ed.gov/ipeds/use-the-data). We narrowed down our data search to 
relatively small baccalaureate schools which focus on arts and sciences. We chose most of the features by hand, such as 
admissions rate, number of instructional faculty, etc. We chose our target variable to be the continuous Total Price of 
attendance for an out-of-state student living on campus. 

To clean the data, we first imported it as a Pandas DataFrame. We then went through and looked at the number of NaN values
that occured in the target feature. Since we did not want to impute the values in those 13 examples to keep the soundness 
of our eventual model, we threw out those 13 examples, which was roughly 5.4% of our dataset. 

We then looked at the other features and the prevalence of NaN values. These came up only sporadically and unpredictably,
therefore, we felt comfortable taking the band-aid approach and imputing the values using the median value of the feature. 
Along the way, we dropped extraneous columns that arose from Pandas concatenation and importing the CSV. We also dropped
the "Percent Database" feature, which, while originally could have been interesting to include, was simply a vector of 0s 
for the subset of schools we selected. Therefore, we threw it out to save on computation time and with the knowledge it 
would not have provided us with any information gain as the value was the same for each example.

This left us with a data set with more than 20 features and 225 examples.

## Feature Scaling
For feature scaling, we decided to use both Sci-Kit Learn scaling techniques as well as L2 regularization. As seen in the EDA, the effect of scalers on outliers can already be captured by standard and min-max scalers, so we apply these two scalers to the data first. L2 
regularization is implemented in the feature_scaling.ipynb notebook. 


## Dimensionality Reduction
We decided to try out both RF feature elimination and PCA. For feature selection using random forest, we first applied the standard and min-max scalers to normalize and standardize the dataset. Then we used a random tree regressor on the data with 80-20 train-test split, and printed out the feature importances obtained by the RF. We ranked the 20 features by feature importance, and applied the threshold of 0.01 to the features to get the 8 most relevant ones.
