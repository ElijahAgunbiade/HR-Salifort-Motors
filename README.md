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
Upon testing Logistic Regression model with an F1 score of .80, we also tested a Random Forest Classifier whose F1 score out to .87 after training. 

![logistic regression](https://github.com/ElijahAgunbiade/HR-Salifort-Motors/assets/173221971/e05191d3-ae98-44e0-ac2c-02498c61d72f)

![random forest 3](https://github.com/ElijahAgunbiade/HR-Salifort-Motors/assets/173221971/87832252-3e72-4f93-b5b0-5006bdd11ac6)

With implementing all variables as features, we created a Random Forest Classifier that was trained and has the ability to predict whether an employee will quit with an F1 score of .87/87% success rate. 

![random forest 2](https://github.com/ElijahAgunbiade/HR-Salifort-Motors/assets/173221971/1a7c50a3-2095-4d30-96b3-2c5c6a48e690)

![f1 score 2](https://github.com/ElijahAgunbiade/HR-Salifort-Motors/assets/173221971/e51de885-8dc9-45f7-ac02-8384c3adb6eb)


Creating a Confusion Matrix on the Random Forest Classifier, we are able to see that the model predicts more false positives than false negatives, which means that some employees may be identified as at risk of quitting or getting fired, when that's actually not the case. Regardless, this is still a strong model with a 87% F1 score.

![confusion matrix](https://github.com/ElijahAgunbiade/HR-Salifort-Motors/assets/173221971/d5bb787e-0885-4f39-bf08-8ddbf5b35695)

## Executive Summary


