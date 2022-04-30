# Machine-Learning-Final-Project----Linck-John

I found a dataset with Census Data from [this archive](http://archive.ics.uci.edu/ml/datasets/Adult). The dataset has a column for a person's income being more than or less than 50k. I will attempt to create a model to predict this value with the constraint that the person works at least 40 hours.

I took a few preprocessing steps to prepare the dataset. The file downloaded from the archive did not have a header row and had spaces after each column. I added a header row and removed all spaces from the data set and saved the file as adult.csv.

I only want to look at people who work at least 40 hours so I filtered the dataset on that condition. 

Some columns that I do not think will aid in the prediction are fnlwgt, capital-gain, and capital-loss so I remove those from the dataset. I also remove education because this column is encapsulated numerically in education-num.
I hope to use a combination of the remaining columns (age, workclass, education-num, marital-status, occupation, relationship, race, sex, hours-per-week, and native-country) to predict the income attribute.

To start analysis, I created a pairplot of my trimmed down dataset.

![Pair Plot](pairplot.png)

After reviewing the plots, it appears that both age and education-num could be good indicators of the income attribute. In both cases, at the value goes up, the chance that the individual makes more than 50k increases. 

I also created count plots to get the counts by race and by sex for the number of people of each income attribute.

![Count By Race](countbyrace.png)
![Count By Sex](countbysex.png)

Both plots demonstrate that these columns could be used in a model to predict the income attribute.

Finally I added columns to the dataset to obtain the percentage of people that have the respective income attributes at a certain age. I plotted the >50k percentage by age in a lineplot.

![Age by Percentage](agepercent.png)

This plot indicates that as people get older, it is more likely they make more than 50k.


My plan for prediction will be to use a subset of columns to train a model and use the test data in the archive to see how accurate the model is.

