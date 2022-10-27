import pandas as pd
import seaborn as sns

teams = pd.read_csv("teams.csv")

# Using the relevant data from teams.csv
teams = teams[["team", "country", "year", "athletes", "events", "age", "prev_medals", "medals"]]
teams

# Identifying which columns (variables) have a strong correlation with medals
# In this ML model, we will use prev_medals, athletes, and events to predict the number of medals
print(teams.corr()["medals"])

# Plotting the linear regression to see the relationship between the selected variables (x-axis) and medals (y-axis)
sns.lmplot(x="athletes", y="medals", data=teams, fit_reg=True, ci=None)
sns.lmplot(x="events", y="medals", data=teams, fit_reg=True, ci=None)
sns.lmplot(x="prev_medals", y="medals", data=teams, fit_reg=True, ci=None)

# Removing missing values from dataset
teams = teams.dropna()
teams.shape

# Data splitting - train: (1609/2014 or 79.9%) - test: (405/2014 or 20.1%)
train = teams[teams["year"] < 2012].copy()
test = teams[teams["year"] >= 2012].copy()

# Using mean absolute error metric to evaluate the model
from sklearn.linear_model import LinearRegression
regression = LinearRegression()

# Using data on athletes, events, and previous medals to predict medals
predictors = ["athletes", "events", "prev_medals"]
target = ["medals"]

regression.fit(train[predictors], train["medals"])
LinearRegression()

medal_prediction = regression.predict(test[predictors])

# Assign predictions to a column in test dataframe
test["medal_prediction"] = medal_prediction

# If medal_prediction is less than 0, change to 0
test.loc[test["medal_prediction"] < 0, "medal_prediction"] = 0

# Round predictions to nearest whole number
test["medal_prediction"] = test["medal_prediction"].round()

# Looking at mean absolute error...
from sklearn.metrics import mean_absolute_error
error = mean_absolute_error(test["medals"], test["medal_prediction"])

# Looking at errors by country
errors = (test["medals"] - test["medal_prediction"]).abs()
errors

# Grouping errors by team, medals by team
errors_by_team = errors.groupby(test["team"]).mean()
medals_by_team = test["medals"].groupby(test["team"]).mean()

# Finding ratio between errors...
error_ratio = errors_by_team / medals_by_team

import numpy as np
error_ratio = error_ratio[np.isfinite(error_ratio)]
error_ratio.sort_values()

# Looking at 2012 and 2016 medal predictions for USA, China, Russia, Lithuania, and Morocco
print(test[test["team"] == "USA"])
print(test[test["team"] == "CHN"])
print(test[test["team"] == "RUS"])
print(test[test["team"] == "LTU"])
print(test[test["team"] == "MAR"])
