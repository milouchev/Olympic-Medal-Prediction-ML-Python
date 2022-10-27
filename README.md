# Olympic-Medal-Prediction-ML-Python

A simple machine learning model that predicts a team's medal count during the Olympic Games.

Using historical data from 1964 to 2016, the model uses the athlete count, event count, and previous medals count variables as predictors for team medal counts, as these variables have strong correlations, ranging from 0.77 to 0.92.

<img width="1020" alt="variable_relationships" src="https://user-images.githubusercontent.com/98411949/198403571-a055af73-56d9-44f5-a13f-93dab5e1fd5a.png">

The model uses data from 1964-2008 for training and data from 2012-2016 for testing. Making predictions for the 2012 and 2016 games allows the user to assess the accuracy of the model, as the actual medal count is present in the data.

This simple ML model provides fairly accurate predictions for teams that typically receive many medals, and less accurate predictions for teams that
do not receive many medals.
