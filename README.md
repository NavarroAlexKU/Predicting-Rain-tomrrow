
# Predicting Rain Tomorrow Using Classification Modeling

Using various classification methods, the objective of the project is to develop a model that will predict the variable "RainTomorrow" which equals "Yes" if it rained the next day and "No" if it did not rain the next day.


![App Screenshot](https://www.news8000.com/content/uploads/2020/05/WKBT-Master-2.png?raw=True)

## Authors

- [@NavarroAlexKU](https://github.com/NavarroAlexKU/Predicting-Housing-Price.git)

## 🔗 Social Media Links
[![linkedin](https://img.shields.io/badge/linkedin-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/alexnavarro2/)

## Documentation
You can get the dataset used in the analysis by downloading it from my GitHub website.

[Data](https://github.com/NavarroAlexKU/Predicting-Rain-tomrrow)

## Installation & Packages:
![App Screenshot](https://github.com/NavarroAlexKU/Predicting-Rain-tomrrow/blob/main/R%20Logo.png?raw=True)
The analysis was done using R, you will need the following packages to run the code.
* library(ROCR)
* library(pROC)
* library(rpart)
* library(rpart.plot)
* library(lattice)
* library(naivebayes)
* library(nnet)
* library(NeuralNetTools)
```
install.packages("ROCR")

install.packages("rpart")
```
## Modeling:
I'm going to build four separate classification models using the following methods:
* Logistic Regression
* Decision Tree Classification
* Naive Bayes Classification
* Neural Network Classification

### Logistic Regression:
The first model I will build is a logistic regression model. I'm going to use forward stepwise regression to determine the features most important for my model. Our final model will be based off the best AIC value. We want to choose the model that has the lowest AIC value. For more information on forward stepwise and AIC, please see the following link.
[Stepwise Regression](https://en.wikipedia.org/wiki/Stepwise_regression)

The following is the optimal model output for the logistic regression model:

Optimal Model Based On Best AIC:

![App Screenshot](https://github.com/NavarroAlexKU/Predicting-Rain-tomrrow/blob/main/Screen%20Shot%202021-11-03%20at%204.40.06%20PM.png?raw=True)

Logsitic Regression Model Output

![App Screenshot](https://github.com/NavarroAlexKU/Predicting-Rain-tomrrow/blob/main/Screen%20Shot%202021-11-03%20at%204.41.15%20PM.png?raw=True)

Confusion Matrix:

![App Screenshot](https://github.com/NavarroAlexKU/Predicting-Rain-tomrrow/blob/main/Screen%20Shot%202021-11-03%20at%205.24.07%20PM.png?raw=True)

ROC Plot:

![App Screenshot](https://github.com/NavarroAlexKU/Predicting-Rain-tomrrow/blob/main/Screen%20Shot%202021-11-03%20at%205.24.20%20PM.png?raw=True)

False Positive Plot:

![App Screenshot](https://github.com/NavarroAlexKU/Predicting-Rain-tomrrow/blob/main/Screen%20Shot%202021-11-03%20at%205.24.28%20PM.png?raw=True)

False Negative:

![App Screenshot](https://github.com/NavarroAlexKU/Predicting-Rain-tomrrow/blob/main/Screen%20Shot%202021-11-03%20at%205.24.45%20PM.png?raw=True)


![App Screenshot](https://github.com/NavarroAlexKU/Predicting-Rain-tomrrow/blob/main/Screen%20Shot%202021-11-03%20at%206.06.15%20PM.png?raw=True)

Calculating:
* False Positive Rate
* False Negative Rate
* Overall Error Rate
* Sensitivity
* Specificity

FPR = 397/(1420+397)*100
* False Positive Rate = 21%

FNR = 109/(109+452)*100
* False Negative Rate = 22%

Total = 1420+109+397+452
Overall_Error_Rate = ((397+109)/Total)*100
* Overall_Error_Rate = 21%

Sensitivity = 1 - 0.22 = 0.78%
So a little over 78% of the time these weather stations are correctly predicting rain tomorrow.

Specificity = 1 - 0.21 = 0.79%
So a little over 79% of the time these weather stations are correctly predicting when it will not rain tomorrow.

ROC Curve and Area Under Curve:


![App Screenshot](https://github.com/NavarroAlexKU/Predicting-Rain-tomrrow/blob/main/Screen%20Shot%202021-11-03%20at%207.31.37%20PM.png?raw=True)


Based on the above plots, I believe that a good cut off point is 0.02 as we start to see the curves flatten around 0.02

Next we will want to check the AUC (area under curve) for our model. We want the ROC curve as far as possible towards the left. Good rule of thumb is if our AUC > 0.7 means our model is good enough. Our AUC is 88% so we have a good enough model.

![App Screenshot](https://github.com/NavarroAlexKU/Predicting-Rain-tomrrow/blob/main/Screen%20Shot%202021-11-03%20at%208.19.33%20PM.png?raw=True)

### Decision Tree Classifier:

Now I will build the second classification model using decision trees.

First Tree:

![App Screenshot](https://github.com/NavarroAlexKU/Predicting-Rain-tomrrow/blob/main/Screen%20Shot%202021-11-03%20at%209.18.55%20PM.png?raw=True)

Model CP Error Output:

![App Screenshot](https://github.com/NavarroAlexKU/Predicting-Rain-tomrrow/blob/main/Screen%20Shot%202021-11-03%20at%209.20.10%20PM.png?raw=True)

The column labeled “xerror” is the cross-validation error for different subsets of the tree. What we are looking for is a low cross validation error. The column labeled “xstd” gives the standard errors of the cross validation errors. This column can be used to get an idea of how much the values in the “xerror” column could reasonably vary.

Looking at the xerror column, we can see that xerror starts to level out with the third value 0.76981 and isn't much different in error rate vs the 0.70513 value. Thus, the tree corresponding to the third row is the tree that gives us nearly the lowest cross validation error and least complicated model using a cp of 0.017337. We will use this to prune our tree as we do not want to overfit the model.

Tree Model 1 Summary Statistics:

![App Screenshot](https://github.com/NavarroAlexKU/Predicting-Rain-tomrrow/blob/main/Screen%20Shot%202021-11-03%20at%209.20.03%20PM.png?raw=True)

Prune Tree:

![App Screenshot](https://github.com/NavarroAlexKU/Predicting-Rain-tomrrow/blob/main/Screen%20Shot%202021-11-04%20at%205.08.51%20PM.png?raw=True)

# Error Rate Prune Tree:

![App Screenshot](https://github.com/NavarroAlexKU/Predicting-Rain-tomrrow/blob/main/Screen%20Shot%202021-11-04%20at%205.08.57%20PM.png?raw=True)