# Databricks notebook source
# MAGIC %md
# MAGIC # Step 1 - Importing the required libraries

# COMMAND ----------

# Importing the required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler

# COMMAND ----------

# MAGIC %md
# MAGIC # Step 2 - Importing the raw data

# COMMAND ----------

#Importing the raw data
raw_data=pd.read_csv("/dbfs/FileStore/day.csv")
raw_data.info()

# COMMAND ----------

# MAGIC %md
# MAGIC # Step 3 - Data Pre-processing

# COMMAND ----------

# Data Preprocessing based on the data definition

# Removing the instant column as it only shows the record sequence number
clean_data=raw_data.copy()
clean_data.drop(["instant"],axis=1,inplace=True)

# Removing the dteday column as it gives the date information which is already present in other columns
clean_data.drop(["dteday"],axis=1,inplace=True)

# Manipulating the season column based on the data definition
clean_data["season"]=clean_data["season"].map({1:"spring",2:"summer",3:"fall",4:"winter"})

# Manipulating the yr column based on the data definition
clean_data["yr"]=clean_data["yr"].map({0:"2018",1:"2019"})

# Removing casual and registered columns as they will not be available while predicting and their sum is already given by the cnt column
clean_data.drop(["casual","registered"],axis=1,inplace=True)


# COMMAND ----------

# Checking the details after cleaning the data
clean_data.info()

# COMMAND ----------

# MAGIC %md
# MAGIC # Step 4 - EDA (Exploratory Data Analysis)

# COMMAND ----------

## Data exploration and finding co-relations of the independent variables with the target column "cnt"

# COMMAND ----------

# MAGIC %md
# MAGIC ### Continuous Variables

# COMMAND ----------

# Checking data distribution for the continuous variables
clean_data[["temp","atemp","hum","windspeed","cnt"]].hist()

# COMMAND ----------

# checking correlations of the target with the continuos variables
clean_data[["temp","atemp","hum","windspeed","cnt"]].corr()

# COMMAND ----------

# checking correlations of the target with the continuos variables
cols=["temp","atemp","hum","windspeed"]

for a in cols:
    plt.figure()
    sns.scatterplot(x=a,y="cnt",data=clean_data)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Categorical Variables
# MAGIC

# COMMAND ----------

# checking correlations of the target with the continuos variables
cols=["season","yr","mnth","holiday","weekday","workingday","weathersit"]

for a in cols:
    plt.figure()
    sns.boxplot(x=a,y="cnt",data=clean_data)

# COMMAND ----------

# MAGIC %md
# MAGIC ### EDA - Observations
# MAGIC - "temp" and "atemp" have a positive correlation with the target "cnt". It means higher the temperature, more are the number of bike rides.
# MAGIC - "hum" and "windspeed" have a slight negative correlation with the target "cnt". It means higher the humidity or windspeed, lesser are the number of bike rides.
# MAGIC - "weathersit" of value 1 (Clear, Few clouds, Partly cloudy, Partly cloudy) sees a higher number of bike riders as compared to cloudy/rainy/snowy days.
# MAGIC - Non holiday days have a higher number of bike rides as compared to the holidays.
# MAGIC - Summer and fall season see a higher number of bike rides as compared to spring and winters.
# MAGIC - Months June to October see a higher number of bike rides as compared to the rest of the year.
# MAGIC - "weekday" and "workingday" do not have much impact on the bike rides and they also seem to represent the similar meanings to the data

# COMMAND ----------

# Removing the "weekday" column as it does not have much correlation with the target and also, its essence is covered by the 'workingday' column

clean_data=clean_data.drop(["weekday"],axis=1)

# COMMAND ----------

# MAGIC %md
# MAGIC # Step 5 - One Hot Encoding the categorical columns

# COMMAND ----------

# Using OHE(One Hot Encoding) to convert the categorical columns to numerical columns

categories="auto"
handle_unknown="ignore"

data_categorial = clean_data[["season","yr","mnth","weathersit"]]

encoder = OneHotEncoder(categories = categories, handle_unknown = handle_unknown,drop="first")
encoded_array=encoder.fit_transform(data_categorial)
encoded_data=pd.DataFrame(encoded_array.toarray(),columns=encoder.get_feature_names(input_features=["season","yr","mnth","weathersit"]))

# COMMAND ----------

# Concatinating the encoded data with clean data to get the final_data
final_data=pd.concat([clean_data.drop(["season","yr","mnth","weathersit"],axis=1),encoded_data],axis=1)
final_data.info()

# COMMAND ----------

# MAGIC %md
# MAGIC # Step 6 - pvalue analysis

# COMMAND ----------

import scipy
from scipy.stats import chi2
from scipy.stats import chi2_contingency
from scipy.stats import pearsonr, spearmanr

# Checking the p-values of the independent variables
x=final_data.drop(["cnt"],axis=1)
y=pd.DataFrame(final_data['cnt'])

pd.DataFrame(
    [scipy.stats.pearsonr(x[col], 
    y) for col in x.columns], 
    columns=["Pearson Corr.", "p-value"], 
    index=x.columns,
).round(4)

# COMMAND ----------

# MAGIC %md
# MAGIC ###Observations
# MAGIC Most of the independent variables have a low p-value and hence seem to be highly correlated with the dependent variable except for "month_4" and "month_11"

# COMMAND ----------

# MAGIC %md
# MAGIC # Step 7 - Dealing with multicolinearity

# COMMAND ----------

from statsmodels.stats.outliers_influence import variance_inflation_factor

# COMMAND ----------

# Checking the VIF (Variance Inflation Factor) of the independent variables
def calc_vif(X):

    # Calculating VIF
    vif = pd.DataFrame()
    vif["variables"] = X.columns
    vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

    return(vif)

X = final_data.drop("cnt",axis=1)
calc_vif(X)

# COMMAND ----------

# Checking the VIF (Variance Inflation Factor) of the independent variables after removing "temp" variable which has a high VIF

def calc_vif(X):

    # Calculating VIF
    vif = pd.DataFrame()
    vif["variables"] = X.columns
    vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

    return(vif)

X = final_data.drop(["temp","cnt"],axis=1)
calc_vif(X)

# COMMAND ----------

# Finally removing columns from the input data to handle multicolinearity
final_data=final_data.drop(["temp"],axis=1)

# COMMAND ----------

# MAGIC %md
# MAGIC # Step 8 - Using Sklearn to train the Linear Regression Model

# COMMAND ----------

# Importing the required libraries
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
from sklearn.preprocessing import MinMaxScaler

# COMMAND ----------

# Scaling the data using MinMaxScaler

x=final_data.drop(["cnt"],axis=1)
y=final_data['cnt']

x_scaled=MinMaxScaler().fit_transform(x)

# COMMAND ----------

# Splitting the data into training and testing sets
x_train,x_test,y_train,y_test=train_test_split(x_scaled,y,test_size=0.3,random_state=101)  

# COMMAND ----------

# Applying the Linear Regression Model, training it on the test set and checking the predictions.

model=LinearRegression()
model.fit(x_train,y_train)

predictions=model.predict(x_test)

score=r2_score(y_test,predictions)
print("R2 score is :",score)

# COMMAND ----------

# Trying to check the feature importance by seeing the coefficients. Since the data is scaled, coefficients can represent the feature importance.

df1=pd.DataFrame(model.coef_,index=x.columns,columns=['Coefficient'])
df1.sort_values(by='Coefficient',ascending=False)

# COMMAND ----------

# MAGIC %md
# MAGIC # Step 9 - Residual Analysis

# COMMAND ----------

# Checking the distribution of the error terms

residuals=y_test-predictions
sns.distplot(residuals)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Observation
# MAGIC The plot seems to be a normal distribution as expected and centred around 0

# COMMAND ----------

# MAGIC %md
# MAGIC # Conclusion:
# MAGIC
# MAGIC - **"atemp" and "temp"** (We removed "temp" from the final model because the two were highly correlated and could lead to multi-colinearity)
# MAGIC have a very high postive correlation value [0.630685348953104] with the number of shared bike rides.
# MAGIC Higher the temerature, higher is the demand
# MAGIC
# MAGIC - **"yr"**
# MAGIC The year 2019 seems to have seen much higher number of shared bike rides as compared to the year 2018.
# MAGIC Positive correlation value being [0.5697284652110435]
# MAGIC
# MAGIC - **"windspeed"**
# MAGIC It seems to be highly negatively correlated with the demand for shared bike rides.
# MAGIC More the wind, lesser the demand and vice-versa.
# MAGIC Negative correlation value being [-0.2351324951410363]	
# MAGIC
# MAGIC - **"weathersit"**
# MAGIC A value of 1 (Clear, Few clouds, Partly cloudy, Partly cloudy) sees maximum number of shared bike riders followed by value 2(Mist + Cloudy, Mist + Broken clouds, Mist + Few clouds, Mist). Value 3 (Light Snow, Light Rain + Thunderstorm + Scattered clouds, Light Rain + Scattered clouds) sees very few bike rides and there are no bike rides reported for value 4(Heavy Rain + Ice Pallets + Thunderstorm + Mist, Snow + Fog)
# MAGIC
# MAGIC - **"holiday"**
# MAGIC Non holiday days(value=0) have a higher number of shared bike rides as compared to the holidays.
# MAGIC
# MAGIC - **"season"**
# MAGIC Summer and fall season see a higher number of shared bike rides as compared to spring and winters.
# MAGIC
# MAGIC - **"month"**
# MAGIC Months June to October see a higher number of shared bike rides as compared to the rest of the year.