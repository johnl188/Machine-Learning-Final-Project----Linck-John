import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


#Part 1
df = pd.read_csv("adult.csv")

df = df.drop(['fnlwgt', 'capital-gain', 'capital-loss', 'education'], axis=1)

df = df.loc[df['hours-per-week'] >= 40]


# plot = sns.pairplot(df, hue="income")
# plt.show()


# plot = sns.countplot(x='sex', hue='income', data=df)
# plt.show()


df['age-count'] = df.groupby(by=['age'])['income'].transform('count')
df['age-count-by-income'] = df.groupby(by=['age', 'income'])['income'].transform('count')
df['percent-age'] = df['age-count-by-income'] / df['age-count']


plot = sns.lineplot(x='age', y='percent-age', data=df.loc[df['income'] == '>50K'])
plt.show()

