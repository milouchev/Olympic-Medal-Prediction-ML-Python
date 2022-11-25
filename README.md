# Project Overview

A simple machine learning model that predicts a team's medal count during the Olympic Games.

Using historical data from 1964 to 2016, the model uses the athlete count, event count, and previous medals count as predictive variables for team medal counts, as these variables have strong correlations, ranging from 0.77 to 0.92.

<img width="1020" alt="variable_relationships" src="https://user-images.githubusercontent.com/98411949/198403571-a055af73-56d9-44f5-a13f-93dab5e1fd5a.png">

The model uses data from 1964-2008 for training and data from 2012-2016 for testing. Making predictions for the 2012 and 2016 games allows the user to assess the accuracy of the model, as the actual medal count is present in the data.

This simple ML model provides fairly accurate predictions for teams that typically receive many medals, and less accurate predictions for teams that
do not receive many medals.
This can be shown with the results below, where the model predicted 280 and 240 medals for USA in 2012 and 2016, vs the actual 248 and 264 amounts. For Lithuania, the model predicted 0 and 3 medals in 2012 and 2016, vs the actual 5 and 7 amounts.

![predictions](https://user-images.githubusercontent.com/98411949/198413756-dd23c93d-a5d1-4ac5-b7d2-cd52d6e10da5.jpg)

# Requirements & Setup:

<b>Python Packages:</b>
- pandas
- seaborn
- scikit-learn (sklearn)
- numpy
