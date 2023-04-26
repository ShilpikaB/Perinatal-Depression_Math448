# 1 Background

Perinatal depression is a serious mood disorder that can affect women during pregnancy and after childbirth.
- Perinatal Depression = Prenatal Depression + Postpartum Depression 
<div align="center">
  <img src="https://user-images.githubusercontent.com/82466266/234414707-049a1582-9d35-4c58-b720-54ecaf00fe00.JPG" width=80% height=50%>
</div>

### What happens? 
Feelings of extreme sadness, anxiety, fatigue, difficulty in carrying out daily tasks, including caring for themselves, for the newborn, and/or for others.

### Impacts? 
Episodes of constant mental stress and psychosis arising from perinatal depression are detrimental for women health and can also be fatal for 
the fetus or the new-born.

### Caused by?
Combination of genetic and environmental factors. 
- Life stress (for example, experiences of past trauma).
- Physical and emotional demands of childbearing and caring for a new baby.
- Hormonal changes during and after pregnancy.
- Personal or family history of depression or bipolar disorder.
- Experienced perinatal depression with a previous pregnancy.



# 2 Project Plan
### 2.2 Contribution
- Create a predictive model whether a given subject is likely to have perinatal depression or not.
- Look closely to read into the different physiological and environmental features of the subjects.
- Design and analyze how accurately can environmental features alone predict the onset of perinatal depression 
in subjects.

### 2.2 Design Classification Models using:
- Logistic Regression: apply feature selection techniques, Ridge Regression, and Lasso
- Linear Discriminant Analysis (LDA)
- Quadratic Discriminant Analysis (QDA)
- Naïve Bayes
- K-Nearest Neighbors (KNN)
- Classification Decision Tree
- Ensemble methods: Bagging, Random Forest


# 3 Dataset Description
- 
- 

# 4 Data Visualization
<div align="center">
  <img src="" width=80% height=50%>
</div>

# 5 Methods & Implementation


# 6 Code (in R)
R Code: 


# 7 Results
<div align="center">
  <img src="https://user-images.githubusercontent.com/82466266/234496512-8e34de04-d0c5-488a-be02-71ff1a2d17c6.JPG" width=80% height=10%>
  <img src="https://user-images.githubusercontent.com/82466266/234496544-d1330ecc-eede-46eb-93f1-1909056d68fa.JPG" width=80% height=20%>
</div>


# 8 Discussion


### 8.1 Inference: 


### 8.2 Conclusion 
- When using all predictors, best performance accuracy is observed for Logistic Regression, Ridge Regression and Linear Discriminant Analysis (LDA). All 3 models have 68.57% prediction accuracy.
- When using only environmental predictors, KNN (for k=1) gives 97.14% prediction accuracy. Using k=1 implies high flexibility causing low bias but very high variance. The next best prediction accuracy in KNN is observed for k = 4 and 6 and gives accuracy of 91.43%. If prediction accuracy is the goal, then KNN (k=4, 6) would be the suggested model. However, note that from KNN we cannot deduce which predictors are the most influential.
- Prediction accuracy of 71.43% is achieved using classification decision tree (Rpart). The tree structure generated uses the following predictors suggesting INCOME_CAT, DESIRED_PREG_CAT, INSTRUCTION_CAT, EMPLOYED_CAT are the most significant predictors. 

From the dataset used for this experiment we observe the following predictors do contribute toward causing depression:
- Family income (INCOME_CAT): Higher Family Income less chances of depression; implies financial stability
- Currently Employed (EMPLOYED_CAT): Employment assures financial independence to certain extent, awareness.
- Desired pregnancy (DESIRED_PREG_CAT): Contributes toward mental and emotional health both during and after pregnancy.
- Education level of subjects (INSTRUCTION_CAT): Higher education level may imply more awareness, self-care, self-sustained, independence.

### 8.3 Future Direction
The Kaggle data set has prepopulated the fields with certain numerical values for text-fields or categorial fields that were missing values.
- Future work could include using max frequency method or prediction models to generate data for missing or unspecified data fields rather.
- Using neural network to achieve higher accuracy.

