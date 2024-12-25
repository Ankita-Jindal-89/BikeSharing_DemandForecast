# Bike-sharing Demand Forecast
A US bike-sharing provider BoomBikes aspires to understand the demand for shared bikes among the people after this ongoing quarantine situation ends across the nation due to Covid-19. They have planned this to prepare themselves to cater to the people's needs once the situation gets better all around and stand out from other service providers and make huge profits.



## Table of Contents
* [General Info](#general-information)
* [Technologies Used](#technologies-used)
* [Conclusions](#conclusions)


## General Information
BoomBikes is trying to understand and analyse the factors that can have an impact on its revenue and hence growth
The company wants to know:
Which variables are significant in predicting the demand for shared bikes.
How well those variables describe the bike demands

**Dataset** : day.csv
Dataset contains 16 columns that will be used for training and testing the Linear Regression model.



## Conclusions
- **"atemp" and "temp"** (We removed "temp" from the final model because the two were highly correlated and could lead to multi-colinearity)
have a very high postive correlation value [0.630685348953104] with the number of shared bike rides.
Higher the temerature, higher is the demand

- **"yr"**
The year 2019 seems to have seen much higher number of shared bike rides as compared to the year 2018.
Positive correlation value being [0.5697284652110435]

- **"windspeed"**
It seems to be highly negatively correlated with the demand for shared bike rides.
More the wind, lesser the demand and vice-versa.
Negative correlation value being [-0.2351324951410363]	

- **"weathersit"**
A value of 1 (Clear, Few clouds, Partly cloudy, Partly cloudy) sees maximum number of shared bike riders followed by value 2(Mist + Cloudy, Mist + Broken clouds, Mist + Few clouds, Mist). Value 3 (Light Snow, Light Rain + Thunderstorm + Scattered clouds, Light Rain + Scattered clouds) sees very few bike rides and there are no bike rides reported for value 4(Heavy Rain + Ice Pallets + Thunderstorm + Mist, Snow + Fog)

- **"holiday"**
Non holiday days(value=0) have a higher number of shared bike rides as compared to the holidays.

- **"season"**
Summer and fall season see a higher number of shared bike rides as compared to spring and winters.

- **"month"**
Months June to October see a higher number of shared bike rides as compared to the rest of the year.



## Technologies Used
- **Languages:** Python
- **Libraries:** Pandas, NumPy, Matplotlib, Seaborn, sklearn, scipy
- **Tools:** Jupyter Notebook, word, pdf



## Contact
Created by [@Ankita-Jindal-89] - feel free to contact me!

