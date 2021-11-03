
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