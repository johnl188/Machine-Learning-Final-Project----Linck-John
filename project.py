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


# plot = sns.pairplot(df, hue="income")
# plt.show()


# plot = sns.countplot(x='sex', hue='income', data=df)
# plt.show()


# df['age-count'] = df.groupby(by=['age'])['income'].transform('count')
# df['age-count-by-income'] = df.groupby(by=['age', 'income'])['income'].transform('count')
# df['percent-age'] = df['age-count-by-income'] / df['age-count']


#plot = sns.lineplot(x='age', y='percent-age', data=df.loc[df['income'] == '>50K'])
#plt.show()

df = df.loc[df['native-country'] == 'United-States']

df['More Than 50K'] = df['income'].map({ '<=50K': 0, '>50K': 1} ).astype(int)

marriedList = ['Husband', 'Wife']
df['married'] = df['relationship'].apply(lambda x: 1 if x in marriedList else 0)

df['workclass'] = df['workclass'].apply(lambda x: 'Private' if x == '?' else x)

df['occupation'] = df['occupation'].apply(lambda x: 'Other-service' if x == '?' else x)


le = preprocessing.LabelEncoder()
df['sex'] = le.fit_transform(df['sex'])


#print(df.corr(method='pearson')[['More Than 50K']])

#plot = sns.heatmap(df.corr(method='pearson')[['More Than 50K']].sort_values(by='More Than 50K', ascending=False), annot=True)
#plt.show()

# fig = px.scatter_3d(df, x='age', y='education-num', z='hours-per-week', color='income')
# fig.update_traces(marker=dict(size=3))
# fig.show(renderer="notebook")


temp = pd.get_dummies(df['workclass'])
df = df.join(temp)

temp = pd.get_dummies(df['occupation'])
df = df.join(temp)

temp = pd.get_dummies(df['race'])
df = df.join(temp)

#drop encoded
df = df.drop(['workclass', 'occupation', 'race'], axis=1)

df = df.drop(['marital-status', 'relationship', 'native-country', 'income'], axis=1)

y_data = df['More Than 50K']
x_data = df.drop('More Than 50K', axis=1)

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
    #print(reg.score(x_test, y_test))

    # visual = ResidualsPlot(reg)
    # visual.fit(x_train, y_train)
    # visual.score(x_test, y_test)
    # visual.poof()

    clf = tree.DecisionTreeClassifier(max_depth=2)
    clf = clf.fit(x_train, y_train)
    treeResults = np.append(treeResults, clf.score(x_test, y_test))

    #print(clf.score(x_test, y_test))

resultDF = pd.DataFrame()
resultDF['% Test'] = percentages
resultDF['Linear Regression Score'] = linearResults
resultDF['Decision Tree Score'] = treeResults

resultDF = resultDF.loc[resultDF['Linear Regression Score'] > 0]
print(resultDF)

#plot = sns.lineplot(data=resultDF, x='% Test')

plot = sns.lineplot(x='% Test', y='value', hue='variable', 
             data=pd.melt(resultDF, ['% Test']))
plt.show()



