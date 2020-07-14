import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

titanic = pd.read_csv('titanic.csv')
print(titanic.isnull().sum())
# Missing parameters are: Age, Cabin and Embarked.
titanic['Age'].fillna(titanic['Age'].mean, inplace=True)

for i, col in enumerate(['SibSp', 'Parch']):
    plt.figure(i)
    sns.catplot(x=col, y='Survived', data=titanic, kind='point', aspect=2, )
    plt.show()
# We can see that passengers with higher number of siblings or spouses the are less likely to survive,
# so I decided to add them together.

titanic['Family_cnt'] = titanic['SibSp'] + titanic['Parch']
titanic.drop(['PassengerId', 'SibSp', 'Parch'], axis=1, inplace=True)
print(titanic.head())

print(titanic.groupby(titanic['Cabin'].isnull())['Survived'].mean())
# We can see that the survival rate is lower when cabin is missing,
# means that people without cabin are less likely to survive.

# === Converting cabins missing/not missing to numerical values ===
titanic['Cabin_ind'] = np.where(titanic['Cabin'].isnull(), 0, 1)
# === Converting Sex to numerical values ===
gender_dict = {'male': 0, 'female': 1}
titanic['Sex'] = titanic['Sex'].map(gender_dict)
# === Cleaning the DataSet === (Embarked isn't cause relationship after inspection)
titanic.drop(['Cabin', 'Name', 'Ticket', 'Embarked'], axis=1, inplace=True)

titanic.to_csv('titanic_cleaned.csv')
