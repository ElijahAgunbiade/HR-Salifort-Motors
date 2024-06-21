#Predictive Models Salifort Motor   
In this project, we are creating a predictive model for a fictional Motor company 'Salifort Motors' to ancipate whether or not an employee in said company is likely to quit. This is to figure out the factors that contribute to people quiting to figure out what the company needs to focus on to minimize turnover. We will implement feature engineering, model development, and evaluation to figure out what factors the HR deparment at Salifort Motors should focus on.

##Background on the Automatidata scenario
The HR department at Salifort Motors wants to take some initiatives to improve employee satisfaction levels at the company. They collected data from employees, but now they don’t know what to do with it. They refer to you as a data analytics professional and ask you to provide data-driven suggestions based on your understanding of the data. They have the following question: what’s likely to make the employee leave the company?

Your goals in this project are to analyze the data collected by the HR department and to build a model that predicts whether or not an employee will leave the company.

If we can predict employees likely to quit, it might be possible to identify factors that contribute to their leaving. Because it is time-consuming and expensive to find, interview, and hire new employees, increasing employee retention will be beneficial to the company.

##Assignment
We will create a regression model for the TLC data to predict whether or not an employee will leave the company, what features affect that, and calculate the machine learning model's accuracy.

In this project, we will complete the following deliverables:

Data Clean and Organize 

Design and implement a machine learning model

Analyze important factors

Draft an executive summary of your results

## Data Cleaning and Organizing
We managed to merge 2 dataframes, create new columns from main columns and delete unnecessary columns to create columns we needed for the research and delete the ones we didn't. This coding process is shown the Data Cleaning Code. This resulted in us having a column 'generous' which depicted whether a customer was considered to be generous by the percentage of tip they gave compared to the total cost of the trip. This would be very useful in the proceeding on the research.

Design and implement a machine learning model
We created a Random Forest Classifier and an XGB Classifier and compare the F1 Score to identity the most accurate model in predicting the kind of customers that would be the most generous tippers.

We first began by testing the balance of the data based on the amount of data that had generous customers and the amount that didn't. This was to make sure our future model would have enough data of generous and ungenerous tippers to be able to accurately understand and predict what factors make a generous tipper without reporting false or unrepresentative statements. Our outcome fortunately came out to our generous tippers being 52% of the data, as seen in the image below, which is about even; therefore, there would be no misinterpretation when creating the model.

balance

Knowing the data wasn't biased, we decided to build a Random Forest Classifier with our test data as well as an XGB Classifier to test their F1 scores. We also fitted training set on both Classifiers for better results.

random forest

XGB classifier

F1 Score overall

Seeing that our F1 scores in both classifiers are over .50, this means using either models gives us a better chance of predicting who will be a generous tipper than, say a coin flip by about a 40% increase. However, given that our Rainforest Classifier has a higher F1 score than our XGB model, we opted for the Rain Forest Classifier as a higher F1 score means this model is better for prediction in this dataframe than the XGB model.

We then created a Confusion Matrix to understand the characteristics of the Rain Forest Classifyer. We found that the model is almost twice as likely to predict a false positive than it is to predict a false negative. This means the model is far more likely to predict a generous tipper and be wrong, than to predict a normal tipper and be wrong. This is less desirable, because it's better for a driver to be pleasantly surprised by a generous tip when they weren't expecting one than to be disappointed by a low tip when they were expecting a generous one. However, the overall performance of this model was satisfactory.

confustion matrix

Lastly, we created a bar graph to plot the most influencial features that affects whether a customer will be a generous tipper or not in a descending order. We found that vendor id(who the driver is), predicted fare(how much the trip itself is expected to cost), average duration (how long the trip last), and average distance plays a heavy impact whether a customer would be a generous tipper or not

graph

##Executive Summary This model performs acceptably. Its F1 score was 0.7235 and it had an overall accuracy of 0.6865. It correctly identified ~78% of the actual responders in the test set, which is 48% better than a random guess.

Unfortunately, random forest is not the most transparent machine learning algorithm. We know that VendorID, predicted_fare, mean_duration, and mean_distance are the most important features, but we don't know how they influence tipping. This would require further exploration. It is interesting that VendorID is the most predictive feature. This seems to indicate that one of the two vendors tends to attract more generous customers. It may be worth performing statistical tests on the different vendors to examine this further.

In our case, the ways we could improve the models would be to try creating three new columns that indicate if the trip distance is short, medium, or far. We could also engineer a column that gives a ratio that represents (the amount of money from the fare amount to the nearest higher multiple of $5) / fare amount. For example, if the fare were $12, the value in this column would be 0.25, because $12 to the nearest higher multiple of $5 ($15) is $3, and $3 divided by $12 is 0.25. The intuition for this feature is that people might be likely to simply round up their tip, so journeys with fares with values just under a multiple of $5 may have lower tip percentages than those with fare values just over a multiple of $5. We could also do the same thing for fares to the nearest $10. In addition to this, it would be very helpful to have past tipping behavior for each customer. It would also be valuable to have accurate tip values for customers who pay with cash, as well as to have a lot more data. With enough data, we could create a unique feature for each pickup/dropoff combination.
