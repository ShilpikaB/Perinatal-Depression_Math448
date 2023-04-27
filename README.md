# 1 Background

Perinatal depression is a serious mood disorder that can affect women during pregnancy and after childbirth.
- Perinatal Depression = Prenatal Depression + Postpartum Depression 
<div align="center">
  <img src="https://user-images.githubusercontent.com/82466266/234414707-049a1582-9d35-4c58-b720-54ecaf00fe00.JPG" width=50% height=30%>
</div>

### 1.1 What happens? 
Feelings of extreme sadness, anxiety, fatigue, difficulty in carrying out daily tasks, including caring for themselves, for the newborn, and/or for others.

### 1.2 Impacts? 
Episodes of constant mental stress and psychosis arising from perinatal depression are detrimental for women health and can also be fatal for 
the fetus or the new-born.

### 1.3 Caused by?
Combination of genetic and environmental factors.
<div align="center">
  <img src="https://user-images.githubusercontent.com/82466266/234967661-f71afc78-4505-4a5e-8927-5f48265cd618.JPG" width=30% height=15%>
</div>

# 2 Project Plan
- Create a predictive model whether a given subject is likely to have perinatal depression or not.
- Look closely to read into the different physiological and environmental features of the subjects.
- Design and analyze how accurately can environmental features alone predict the onset of perinatal depression 
in subjects.

# 3 Dataset Description
- Number of Observations: 107 (masked subject data from Brazil)
- Number of Predictors/Features: 9
- Response variable: Depression [Positive(1), Negative(2)]
- Type of predictors/features: Quantitative, Categorical, Free texts

### 3.1 Sample dataset
<div align="center">
  <img src="https://user-images.githubusercontent.com/82466266/234967131-7bf16d62-7fd8-4242-ac5f-cf3c33cfa37a.JPG" width=40% height=10%>
</div>

# 4 Data Preprocessing
- Removed rows with missing data
- Modified Predictors: Created new predictors from 'Health Problem' data.
<div align="center">
  <img src="https://user-images.githubusercontent.com/82466266/234968398-e68b2d75-c7ac-486c-99bb-47088274f696.JPG" width=30% height=20%>
</div>

- Random split of data into Training : Test = 70:30 ratio
- Imbalanced data: ~65% subjects no depression; only 35% subjects have depression. Used upSample() function on training dataset to balance the data for model training.
- DEPRESSION (response variable) is factorized as [Positive(1), Negative(0)].
- Dataset views: Comprising all 10 modified predictors (physiological + environmental predictors); Comprising of 6 predictors (environmental predictors) {DESIRED_PREG_CAT, EMPLOYED_CAT, INSTRUCTION_CAT, INCOME_CAT, MARITAL_STATUS_CAT, MENTAL_CAT(history)}

# 5 Methods & Implementation
### 5.1 Design Classification Models using:
- Logistic Regression: apply feature selection techniques, Ridge Regression, and Lasso
- Linear Discriminant Analysis (LDA)
- Quadratic Discriminant Analysis (QDA)
- Naïve Bayes
- K-Nearest Neighbors (KNN)
- Classification Decision Tree
- Ensemble methods: Bagging, Random Forest
<div align="center">
  <img src="https://user-images.githubusercontent.com/82466266/234970765-90268c0d-9c4b-42d4-9587-c35623b93218.JPG" width=90% height=20%>
</div>

# 6 Code (in R)
R Code: https://github.com/ShilpikaB/Perinatal-Depression_Math448/blob/main/Math448_TermProjectScript.R

# 7 Results
Accuracy and Error rates are calculated for each model to assess their performances 
<div align="center">
  <img src="https://user-images.githubusercontent.com/82466266/234496512-8e34de04-d0c5-488a-be02-71ff1a2d17c6.JPG" width=50% height=60%>
  <img src="https://user-images.githubusercontent.com/82466266/234496544-d1330ecc-eede-46eb-93f1-1909056d68fa.JPG" width=50% height=50%>
</div>

# 8 Discussion
- When using all predictors, best performance accuracy is observed for Logistic Regression, Ridge Regression and Linear Discriminant Analysis (LDA). All 3 models have 68.57% prediction accuracy.
- When using only environmental predictors, KNN (for k=1) gives 97.14% prediction accuracy. Using k=1 implies high flexibility causing low bias but very high variance. The next best prediction accuracy in KNN is observed for k = 4 and 6 and gives accuracy of 91.43%. If prediction accuracy is the goal, then KNN (k=4, 6) would be the suggested model. However, note that from KNN we cannot deduce which predictors are the most influential.
- Prediction accuracy of 71.43% is achieved using classification decision tree (Rpart). The tree structure generated uses the following predictors suggesting INCOME_CAT, DESIRED_PREG_CAT, INSTRUCTION_CAT, EMPLOYED_CAT are the most significant predictors. 

### 8.1 Conclusion
We observe the following predictors do contribute toward causing depression. This finding strongly suggests that __environmental factors play a signifcant role in causing perinatal depression than health/genetical factors.__
- Family income (INCOME_CAT): Higher Family Income less chances of depression; implies financial stability
- Currently Employed (EMPLOYED_CAT): Employment assures financial independence to certain extent, awareness.
- Desired pregnancy (DESIRED_PREG_CAT): Contributes toward mental and emotional health both during and after pregnancy.
- Education level of subjects (INSTRUCTION_CAT): Higher education level may imply more awareness, self-care, self-sustained, independence.

### 8.2 Future Direction
The Kaggle data set has prepopulated the fields with certain numerical values for text-fields or categorial fields that were missing values.
- Future work could include using max frequency method or prediction models to generate data for missing or unspecified data fields rather.
- Using neural network to achieve higher accuracy.

