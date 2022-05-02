from random import random
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn import tree
from sklearn import linear_model
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
import plotly.express as px
from yellowbrick.regressor import ResidualsPlot

#Part 1
df = pd.read_csv("adult.csv")

df = df.drop(['fnlwgt', 'capital-gain', 'capital-loss', 'education'], axis=1)

df = df.loc[df['hours-per-week'] >= 40]

# PAIR PLOT
# plot = sns.pairplot(df, hue="income")
# plt.show()

# COUNT BY RACE
# plot = sns.countplot(x='race', hue='income', data=df)
# plt.show()

# COUNT BY SEX
# plot = sns.countplot(x='sex', hue='income', data=df)
# plt.show()


# AGE BY PERCENTAGE
# df['age-count'] = df.groupby(by=['age'])['income'].transform('count')
# df['age-count-by-income'] = df.groupby(by=['age', 'income'])['income'].transform('count')
# df['percent-age'] = df['age-count-by-income'] / df['age-count']

# plot = sns.lineplot(x='age', y='percent-age', data=df.loc[df['income'] == '>50K'])
# plt.show()




# PART 2

# WORK CLASS
# plot = sns.countplot(x='workclass', data=df)
# plt.show()

df['workclass'] = df['workclass'].apply(lambda x: 'Private' if x == '?' else x)

# Education Num
# plot = sns.countplot(x='education-num', data=df)
# plt.show()

# Marital Status / Relationship
# plot = sns.countplot(x='marital-status', data=df)
# plt.show()

# plot = sns.countplot(x='relationship', data=df)
# plt.show()

# plot = sns.countplot(x='marital-status', hue='relationship', data=df)
# plt.show()

marriedList = ['Husband', 'Wife']
df['married'] = df['relationship'].apply(lambda x: 1 if x in marriedList else 0)

# Occiupation
# plot = sns.countplot(x='occupation', data=df)
# plt.show()

df['occupation'] = df['occupation'].apply(lambda x: 'Other-service' if x == '?' else x)


# Race
# plot = sns.countplot(x='race', data=df)
# plt.show()


# Sex
le = preprocessing.LabelEncoder()
df['sex'] = le.fit_transform(df['sex'])


# Native Country
# df['isUS'] = df['native-country'].apply(lambda x: 'US' if x == 'United-States' else 'Not US')
# plot = sns.countplot(x='isUS', data=df)
# plt.show()

df = df.loc[df['native-country'] == 'United-States']

# Income
df['More Than 50K'] = df['income'].map({ '<=50K': 0, '>50K': 1} ).astype(int)



# HEAT MAP
# plot = sns.heatmap(df.corr(method='pearson')[['More Than 50K']].sort_values(by='More Than 50K', ascending=False), annot=True)
# plt.show()

# 3D Scatter
# fig = px.scatter_3d(df.sample(frac=0.1, random_state=1), x='age', y='education-num', z='hours-per-week', color='income')
# fig.update_traces(marker=dict(size=3))
# fig.show(renderer="notebook")


# One hot encode a few columns
temp = pd.get_dummies(df['workclass'])
df = df.join(temp)

temp = pd.get_dummies(df['occupation'])
df = df.join(temp)

temp = pd.get_dummies(df['race'])
df = df.join(temp)

# Drop one hot encoded columns
df = df.drop(['workclass', 'occupation', 'race'], axis=1)

# Drop a few other columns that were taken care of before.
df = df.drop(['marital-status', 'relationship', 'native-country', 'income'], axis=1)


# Model Training

# Get the data and scale it
y_data = df['More Than 50K']
x_data = df.drop('More Than 50K', axis=1)

y_columns = ['More Than 50K']
x_columns = x_data.columns.tolist()

print(y_columns)
print(x_columns)

scaler = preprocessing.StandardScaler()
x_data = scaler.fit_transform(x_data)

percentages = np.array([])
linearResults = np.array([])
treeResults = np.array([])

for num in range(1, 10):

    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=num * 0.1, random_state=1)

    percentages = np.append(percentages, num * 0.1)

    reg = linear_model.LinearRegression()
    reg = reg.fit(x_train, y_train)
    linearResults = np.append(linearResults, reg.score(x_test, y_test))

    # Plot residuals for one of the iterations
    # if (num == 5): 
    #     visual = ResidualsPlot(reg)
    #     visual.fit(x_train, y_train)
    #     visual.score(x_test, y_test)
    #     visual.poof()

    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(x_train, y_train)
    treeResults = np.append(treeResults, clf.score(x_test, y_test))

    # Show Decision Tree Visual for one of the iterations
    # if (num == 5): 
    #     plot = tree.plot_tree(clf, 
    #                feature_names=x_columns,  
    #                class_names=['<=50K', '>50K'],
    #                filled=True)
    #     plt.show()

resultDF = pd.DataFrame()
resultDF['% Test'] = percentages
resultDF['Linear Regression Score'] = linearResults
resultDF['Decision Tree Score'] = treeResults

resultDF = resultDF.loc[resultDF['Linear Regression Score'] > 0]
print(resultDF)

# Result Comparison Plot
# plot = sns.lineplot(x='% Test', y='value', hue='variable', 
#              data=pd.melt(resultDF, ['% Test']))
# plt.show()

# Decisions Tree
