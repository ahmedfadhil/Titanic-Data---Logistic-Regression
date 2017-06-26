import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cufflinks as cf

train = pd.read_csv('titanic_train.csv')
train.head()
# sns.heatmap(train.isnull(), yticklabels=False, cbar=False, cmap='viridis')
sns.set_style('whitegrid')
# sns.countplot(data=train, x='Survived',hue='Sex',palette='RdBu_r')
# sns.countplot(data=train, x='Survived', hue='Pclass')
# sns.distplot(train['Age'].dropna(), kde=False, bins=30)
# train['Age'].plot.hist(bins=35)
# sns.countplot(data=train, x='SibSp')
# train['Fare'].hist(bins=45, figsize=(10, 4))

cf.go_offline()
train['Fare'].iplot(kind='hist', bins=45)
plt.figure(figsize=(10, 7))


# sns.boxplot(data=train, x='Pclass', y='Age')


def impute_age(cols):
    Age = cols[0]
    Pclass = cols[1]

    if pd.isnull(Age):
        if Pclass == 1:
            return 37
        elif Pclass == 2:
            return 29
        else:
            return 24
    else:
        return Age


train['Age'] = train[['Age', 'Pclass']].apply(impute_age, axis=1)

# sns.heatmap(train.isnull(), yticklabels=False, cbar=False, cmap='viridis')

train.drop('Cabin', axis=1, inplace=True)
train.head()

train.dropna(inplace=True)

sex = pd.get_dummies(train['Sex'], drop_first=True)
sex.head()

embark = pd.get_dummies(train['Embarked'], drop_first=True)
embark.head()
train = pd.concat([train, sex, embark], axis=1)
# train.head(2)

train.drop(['Sex', 'Embarked', 'Name', 'Ticket'], axis=1, inplace=True)

train.head()
train.tail()

train.drop(['PassengerId'], axis=1, inplace=True)
