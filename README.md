
# Titanic Survival Prediction
This project involves exploratory data analysis (EDA) and machine learning modeling on the Titanic dataset to predict passenger survival. The goal is to analyze the dataset, preprocess the data, and build various machine learning models to predict survival rates.

## Table of Contents
- [Project Overview](#project-overview)
- [Installation](#installation)
- [Usage](#usage)
  - [Data Loading](#data-loading)
  - [Data Preprocessing](#data-preprocessing)
  - [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
  - [Model Building and Evaluation](#model-building-and-evaluation)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)
## Project Overview

The Titanic dataset provides information about the passengers on the Titanic and whether they survived or not. This project aims to:

1. Perform exploratory data analysis to understand the dataset.
2. Preprocess the data for machine learning models.
3. Build and evaluate multiple classification models to predict survival.
## Installation

Clone this repository and install the necessary packages using pip:

```bash
git clone https://github.com/your-username/Titanic_Final.git
pip install -r requirements.txt
```

#### Requirements
The required Python packages are listed in `requirements.txt`:

```bash
pandas
numpy
scikit-learn
matplotlib
seaborn
xgboost
```
## Usage/Examples

### Data Loading
The Titanic dataset can be loaded from a CSV file. Ensure that you have the dataset file named `titanic.csv` in the project directory.

```python 
import pandas as pd

# Load the dataset
titanic_data = pd.read_csv('train.csv')
```

### Data Preprocessing

**1. Handling Missing Values:**
```python
titanic_data['Age'].fillna(titanic_data['Age'].mean(),inplace=True)
titanic_data['Embarked'].fillna(titanic_data['Embarked'].mode()[0], inplace=True)
titanic_data = titanic_data.drop(columns='Cabin', axis=1)
```

**2. Feature Engineering:**
```python 
titanic_data['FamilySize'] = titanic_data['SibSp'] + titanic_data['Parch']
titanic_data['Title'] = titanic_data['Name'].apply(lambda x: x.split(',')[1].split('.')[0].strip())
titanic_data['Title'] = titanic_data['Title'].replace('Mlle', 'Miss')
titanic_data['Title'] = titanic_data['Title'].replace('Mme', 'Mrs')
titanic_data['Title'] = titanic_data['Title'].replace('Ms', 'Mrs')
titanic_data.loc[(~titanic_data['Title'].isin(['Mr', 'Mrs', 'Miss', 'Master'])), 'Title'] = 'Rare Title'
```

**3. Encoding:**
```python
titanic_data['Sex'] = titanic_data['Sex'].map({'male': 0, 'female': 1})
titanic_data['Embarked'] = titanic_data['Embarked'].map({'S':0,'C':1,'Q':2})
titanic_data['Title'] = titanic_data['Title'].map({'Mr':0,'Mrs':1,'Miss':2,'Master':3,'Rare Title':4})
```

**4. Splitting Data:**
```python 
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'FamilySize', 'Title']
X = titanic_data[features]
y = titanic_data['Survived']  # Assuming 'Survived' is the target column
# Splitting data into train and test

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train.shape,y_train.shape,X_test.shape,y_test.shape
```

### Exploratory Data Analysis (EDA)
Perform EDA to understand the dataset and visualize various aspects.

```python 
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(8, 6))
sns.barplot(x=survived_counts.index, y=survived_counts.values, palette='viridis')
plt.title('Survival Count')
plt.xlabel('Survived (1 = Yes, 0 = No)')
plt.ylabel('Number of Passengers')
plt.xticks([0, 1], ['No', 'Yes'])
plt.show()

# Number of survivors by Gender wise
sns.countplot(x="Sex", hue="Survived", data=titanic_data)

sns.catplot(x='Sex', col='Pclass', hue='Survived', data=titanic_data, kind='count');

sns.countplot(data=titanic_data,x='Title', hue='Survived');

# Correlation Matrix after Encoding
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'FamilySize', 'Title']
plt.figure(figsize=(12, 8))
sns.heatmap(titanic_data[features + ['Survived']].corr(), annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation Matrix')
plt.show()

```

### Model Building and Evaluation
```python 
# Classification Algorithms

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier

# Define and train models
models = {
    'GaussianNB': GaussianNB(),
    'LogisticRegression': LogisticRegression(max_iter=500, random_state=42),
    'KNN': KNeighborsClassifier(),
    'DecisionTree': DecisionTreeClassifier(random_state=42),
    'RandomForest': RandomForestClassifier(random_state=42),
    'GradientBoosting': GradientBoostingClassifier(random_state=42),
    'XGBClassifier': XGBClassifier()
}

# Train and evaluate models
for name, model in models.items():
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    predictions = [round(value) for value in y_pred]


    # evaluate predictions
    accuracy = accuracy_score(y_test, predictions)
    print("Accuracy: %.2f%%" % (accuracy * 100.0),name)
```
## Results
- **GaussianNB:** Accuracy: 76.54%
- **LogisticRegression:** Accuracy: 78.77%
- **KNN:** Accuracy: 74.30%
- **DecisionTreeClassifier:** Accuracy: 78.77%
- **RandomForestClassifier:** Accuracy: 83.80%
- **GradientBoostingClassifier:** Accuracy: 81.01%
- **XGBClassifier:** Accuracy: 83.80%
## Contributing

Contributions are always welcome!
Feel free to fork this repository and submit pull requests. Contributions to improve the code, analysis, or documentation are welcome.

## License

This project is licensed under the MIT License - see the [LICENSE](https://choosealicense.com/licenses/mit/) file for details.


For more detailed analysis or specific questions, please refer to the code comments and documentation within this repository.

### Key Points
- **Data Loading**: Specifies how to load the data.
- **Data Preprocessing**: Details handling missing values, encoding categorical variables, and feature engineering.
- **EDA**: Provides examples of exploratory data analysis.
- **Model Building and Evaluation**: Shows how to train and evaluate multiple models.
- **Results**: Summarizes the performance of each model.
- **Contributing and License**: Includes information on contributing to the project and licensing.

Feel free to customize the README file further based on your project specifics and preferences.
