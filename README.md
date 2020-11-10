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
For feature scaling, we decided to use both Sci-Kit Learn scaling techniques as well as L2 regularization. L2 
regularization is implemented in the feature_scaling.ipynb notebook. 


## Dimensionality Reduction
We decided to try out both RF feature elimination and PCA. 