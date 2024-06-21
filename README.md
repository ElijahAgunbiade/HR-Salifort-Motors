# Predictive Models Salifort Motor   
In this project, we are creating a predictive model for a fictional Motor company 'Salifort Motors' to ancipate whether or not an employee in said company is likely to quit. This is to figure out the factors that contribute to people quiting to figure out what the company needs to focus on to minimize turnover. We will implement feature engineering, model development, and evaluation to figure out what factors the HR deparment at Salifort Motors should focus on.

## Background on the Automatidata scenario
The HR department at Salifort Motors wants to take some initiatives to improve employee satisfaction levels at the company. They collected data from employees, but now they don’t know what to do with it. They refer to you as a data analytics professional and ask you to provide data-driven suggestions based on your understanding of the data. They have the following question: what’s likely to make the employee leave the company?

Your goals in this project are to analyze the data collected by the HR department and to build a model that predicts whether or not an employee will leave the company.

If we can predict employees likely to quit, it might be possible to identify factors that contribute to their leaving. Because it is time-consuming and expensive to find, interview, and hire new employees, increasing employee retention will be beneficial to the company.

## Assignment
We will create a regression model for the TLC data to predict whether or not an employee will leave the company, what features affect that, and calculate the machine learning model's accuracy.

In this project, we will complete the following deliverables:

Data Clean and Organize 

Analyze important factors

Design and implement a machine learning model

Re-analyze important factors

Draft an executive summary of your results

## Data Cleaning and Organizing
Given the size of the data, and the research we were doing, we felt the dataframe provided had sufficient data to derive valuable information from. Therefore, all we focused on were missing data, outliers, and duplicates to decide what actions to take against them, as seen in the Data Cleaning Code file. 

## Data Analysis
We first decided to analyze the relationship with numbers of hours worked to the people who quit, and number of projects worked to people quiting. This is because we felt, realistically, these two factors would/could be very big factors on the quality of life for the employees and whether they decided to stay in the company or not. 

![projects and hours](https://github.com/ElijahAgunbiade/HR-Salifort-Motors/assets/173221971/10cb3c2b-32a0-4869-b2f6-3350318742bf)

As seen in the data, those who worked more hours as well has more project had higher likelihood of quitting. For example, with those working 7 projects and consequently more hours, every single one of them had quit. Looking one step lower, about 50% of those working 6 projects had quit. With more hours and more projects, came higher likelihood of employees quiting. 
There are outliers, however, with those working the lowest amount of projects and hours having high exit rate as well. This could data possibly more so be due to quite quitter who were soon leaving the company and decided to have less work or those not doing much work and being fired than it does have to do with people actually working those hours and deciding to exit themselves. This would be an information that we would pointed out to Salifort Motors and ask for more informations about to dive deeper into said subject. 

We also ran a plot on the relationship between satisfaction level and the amount of hours worked a month, while simultanously comparing the data for those who stayed and those who left. 

![image](https://github.com/ElijahAgunbiade/HR-Salifort-Motors/assets/173221971/3b6cd9e0-6238-4c53-b171-d7b2f0cb5504)

This plot showed that there were masses of those working over 240 hours a month (about 75 hours a week) with very low satisfaction rate as well as those working normal hours with low satisfaction rate. While the reason for low satisfaction rate with the groups working over 240 hours a month is self-explanatory, those working normal hours with low satisfaction rate is a little bit trickier. As explained before, this could be those who decided to work less hours as they were exiting, or it could also be those who didn't like the lifestyle of working less than their peers and feeling the pressure to work more. 
Yes, there are those with high hours and high satisfaction rate who did leave, but that could simply be explained by better job offers, or requiring a temporary hold on their career for personal reasons. 

We also made a visualization on how tenure affected satisfaction rate.

![tenure](https://github.com/ElijahAgunbiade/HR-Salifort-Motors/assets/173221971/17aa2def-ee47-4ff5-86ac-03cd8a698da0)

This showed a higher average of satisfaction rate with employees with over 6 years tenure, and those under 3 years tenure. Employees that presented lower satisfaction rate and were more likely to leave the company where those with mid numbered tenure years from 3 to 6 years tenure. 
This is very interesting data in regards to who to focus on when trying to minimize turnover rate.

We then compared salary, which showed that long-tenured employees compromised of more higher-paid employees than the other groups.

![salary](https://github.com/ElijahAgunbiade/HR-Salifort-Motors/assets/173221971/d3c75182-f1a1-4fce-b884-cce8efbc6138)

Comparing hours worked and evaluation score with those who left vs. those who stayed, with a scatterplot, we figured employeed who worked the most hours and had very high evaluation scores were far more likely to quit. We also saw those who worked less hours with lower evaluation scores left. This second observation could be due to these individuals getting fired more than them quitting. 
We also noticed that having more hours doesn't give you a better evaluation, as the evaluation rating was spread across the board in regards to the various amount of hours worked. 

![evaluation](https://github.com/ElijahAgunbiade/HR-Salifort-Motors/assets/173221971/961a39a1-4c31-4412-b1e2-9e7a9cd24bc2)

We then plotted the relationship with amount of hours worked, and whether they got a promotion or not. We came to find that very few employees who were promoted in the last five years left, all of the employees who left were working the longest hours, very few employees working extensive hours were promoted. 
This creates a very clear image of why those with mid ranged tenures with the most hours are far more likely to leave than those just starting out or those who had been there over 6 years. 
Regardless, whether working a lot of hours or not, employees noticing the amount of work they put in has nothing to do with promotion does prompt people to consider leaving especially after staying multiple years.

![promotion](https://github.com/ElijahAgunbiade/HR-Salifort-Motors/assets/173221971/b1f80ebc-2f21-4689-9f0f-c73973979007)

We also plotted to see whether the departments employees worked in had any effect on whether they'd quit or not. It did not.

![no effect](https://github.com/ElijahAgunbiade/HR-Salifort-Motors/assets/173221971/68a0085c-c8c6-47a1-8ff4-90a18d542598)

### Creating a heatmap
Creating a heatmap, we found that that the number of projects, monthly hours, and evaluation scores all have some positive correlation with each other, and whether an employee leaves is negatively correlated with their satisfaction level. Meaning higher satisfaction results to lower chance of leaving.

![heatmap2](https://github.com/ElijahAgunbiade/HR-Salifort-Motors/assets/173221971/17c45d09-4b26-40e7-85bd-226245222775)

### Insights
It appears that employees are leaving the company as a result of poor management. Leaving is tied to longer working hours, many projects, and generally lower satisfaction levels. It can be ungratifying to work long hours and not receive promotions or good evaluation scores. There's a sizeable group of employees at this company who are probably burned out. It also appears that if an employee has spent more than six years at the company, they tend not to leave.


## Design and implement a machine learning model


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
