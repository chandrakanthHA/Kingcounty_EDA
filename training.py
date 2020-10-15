{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import statsmodels.api as sm\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.metrics import mean_squared_error\n",
    "import numpy as np"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Reading the cleaned data\n",
    "df = pd.read_csv('model_dataset.csv')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Dropping unwanted column\n",
    "df = df.drop(['Unnamed: 0'], axis=1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Dummy variables creation\n",
    "bed_room_dummies = pd.get_dummies(df['number_of_bedrooms'], prefix='bed_rm', drop_first=True)\n",
    "bath_room_dummies = pd.get_dummies(df['numberofbathrooms_per_house'], prefix='bath_rm', drop_first=True)\n",
    "floor_dummies = pd.get_dummies(df['floors'], prefix='flr', drop_first=True)\n",
    "house_condition_dummies = pd.get_dummies(df['house_condition'], prefix='cond', drop_first=True)\n",
    "housing_grade_dummies = pd.get_dummies(df['housing_grade'], prefix='grd', drop_first=True)\n",
    "yr_built_dummies = pd.get_dummies(df['yr_built'], prefix='yr_b', drop_first=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Joining the dataframe and dummyvariable\n",
    "df = pd.concat([df, bath_room_dummies, housing_grade_dummies], axis=1)\n",
    "df = df.drop(['numberofbathrooms_per_house','housing_grade','number_of_bedrooms','floors',\n",
    "             'house_condition','yr_built'], axis=1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [],
   "source": [
    "# feature engineering\n",
    "X = df.iloc[:,1:]\n",
    "Y = df.iloc[:, 0]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "-----  Splitting the data in train and test ----\n"
     ]
    }
   ],
   "source": [
    "#splitting data\n",
    "print(\"-----  Splitting the data in train and test ----\")\n",
    "X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=42)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [],
   "source": [
    "#adding the constant\n",
    "\n",
    "X_train = sm.add_constant(X_train) # adding a constant\n",
    "X_test = sm.add_constant(X_test) # adding a constant"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "-----  Training the model ----\n"
     ]
    }
   ],
   "source": [
    "#training the model\n",
    "print(\"-----  Training the model ----\")\n",
    "model = sm.OLS(y_train, X_train).fit()\n",
    "print_model = model.summary()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "-----  Evaluating the model ----\n"
     ]
    }
   ],
   "source": [
    "#predictions to check the model\n",
    "print(\"-----  Evaluating the model ----\")\n",
    "predictions = model.predict(X_train)\n",
    "err_train = np.sqrt(mean_squared_error(y_train, predictions))\n",
    "predictions_test = model.predict(X_test)\n",
    "err_test = np.sqrt(mean_squared_error(y_test, predictions_test))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "                            OLS Regression Results                            \n",
      "==============================================================================\n",
      "Dep. Variable:                  price   R-squared:                       0.661\n",
      "Model:                            OLS   Adj. R-squared:                  0.660\n",
      "Method:                 Least Squares   F-statistic:                     707.3\n",
      "Date:                Thu, 15 Oct 2020   Prob (F-statistic):               0.00\n",
      "Time:                        09:55:51   Log-Likelihood:            -1.8942e+05\n",
      "No. Observations:               14197   AIC:                         3.789e+05\n",
      "Df Residuals:                   14157   BIC:                         3.792e+05\n",
      "Df Model:                          39                                         \n",
      "Covariance Type:            nonrobust                                         \n",
      "==========================================================================================\n",
      "                             coef    std err          t      P>|t|      [0.025      0.975]\n",
      "------------------------------------------------------------------------------------------\n",
      "const                  -4.659e+07   1.18e+06    -39.451      0.000   -4.89e+07   -4.43e+07\n",
      "total_living_area_sqft    86.6193      3.400     25.474      0.000      79.954      93.284\n",
      "lot_area_sqft              0.1469      0.034      4.384      0.000       0.081       0.213\n",
      "basement_area_sqft        28.7294      3.728      7.706      0.000      21.421      36.038\n",
      "latitude                5.774e+05   9389.192     61.493      0.000    5.59e+05    5.96e+05\n",
      "longitude              -2.005e+05   1.04e+04    -19.239      0.000   -2.21e+05    -1.8e+05\n",
      "sqft_living15             61.8464      3.269     18.917      0.000      55.438      68.255\n",
      "bath_rm_0.75            8.805e+04   8.99e+04      0.979      0.328   -8.83e+04    2.64e+05\n",
      "bath_rm_1.0             9.657e+04   8.73e+04      1.106      0.269   -7.45e+04    2.68e+05\n",
      "bath_rm_1.25            3.445e+04   1.07e+05      0.322      0.747   -1.75e+05    2.44e+05\n",
      "bath_rm_1.5             9.186e+04   8.74e+04      1.051      0.293   -7.95e+04    2.63e+05\n",
      "bath_rm_1.75             9.16e+04   8.73e+04      1.049      0.294   -7.96e+04    2.63e+05\n",
      "bath_rm_2.0             1.033e+05   8.74e+04      1.182      0.237    -6.8e+04    2.75e+05\n",
      "bath_rm_2.25            8.772e+04   8.74e+04      1.004      0.315   -8.36e+04    2.59e+05\n",
      "bath_rm_2.5             5.842e+04   8.73e+04      0.669      0.504   -1.13e+05     2.3e+05\n",
      "bath_rm_2.75            9.496e+04   8.75e+04      1.085      0.278   -7.65e+04    2.66e+05\n",
      "bath_rm_3.0             8.495e+04   8.76e+04      0.970      0.332   -8.67e+04    2.57e+05\n",
      "bath_rm_3.25            1.285e+05   8.77e+04      1.465      0.143   -4.34e+04    3.01e+05\n",
      "bath_rm_3.5             1.113e+05   8.77e+04      1.270      0.204   -6.05e+04    2.83e+05\n",
      "bath_rm_3.75            1.935e+05   8.91e+04      2.172      0.030    1.89e+04    3.68e+05\n",
      "bath_rm_4.0             1.237e+05   8.92e+04      1.387      0.166   -5.12e+04    2.99e+05\n",
      "bath_rm_4.25            2.052e+05   9.08e+04      2.260      0.024    2.72e+04    3.83e+05\n",
      "bath_rm_4.5             1.279e+05   9.02e+04      1.417      0.156    -4.9e+04    3.05e+05\n",
      "bath_rm_4.75           -1.249e+04   1.16e+05     -0.108      0.914   -2.39e+05    2.14e+05\n",
      "bath_rm_5.0             1.171e+05   1.04e+05      1.121      0.262   -8.77e+04    3.22e+05\n",
      "bath_rm_5.25            9.481e+04   1.16e+05      0.816      0.414   -1.33e+05    3.22e+05\n",
      "bath_rm_5.5             4.008e+05   1.77e+05      2.258      0.024    5.29e+04    7.49e+05\n",
      "bath_rm_5.75           -1.822e+05   1.75e+05     -1.044      0.297   -5.24e+05     1.6e+05\n",
      "bath_rm_6.0            -5.527e+05   1.79e+05     -3.093      0.002   -9.03e+05   -2.02e+05\n",
      "bath_rm_6.5             9052.3944   1.75e+05      0.052      0.959   -3.34e+05    3.52e+05\n",
      "bath_rm_6.75           -5.062e+05   1.75e+05     -2.887      0.004    -8.5e+05   -1.62e+05\n",
      "bath_rm_7.5            -7.994e+04   1.75e+05     -0.458      0.647   -4.22e+05    2.62e+05\n",
      "grd_4                  -5.363e+06   1.34e+05    -40.102      0.000   -5.63e+06    -5.1e+06\n",
      "grd_5                  -5.364e+06   1.31e+05    -41.022      0.000   -5.62e+06   -5.11e+06\n",
      "grd_6                  -5.364e+06   1.31e+05    -40.974      0.000   -5.62e+06   -5.11e+06\n",
      "grd_7                  -5.329e+06   1.31e+05    -40.653      0.000   -5.59e+06   -5.07e+06\n",
      "grd_8                  -5.255e+06   1.31e+05    -39.977      0.000   -5.51e+06      -5e+06\n",
      "grd_9                  -5.135e+06   1.32e+05    -38.972      0.000   -5.39e+06   -4.88e+06\n",
      "grd_10                 -5.045e+06   1.32e+05    -38.213      0.000    -5.3e+06   -4.79e+06\n",
      "grd_11                 -4.948e+06   1.33e+05    -37.273      0.000   -5.21e+06   -4.69e+06\n",
      "grd_12                 -4.783e+06   1.37e+05    -34.874      0.000   -5.05e+06   -4.51e+06\n",
      "==============================================================================\n",
      "Omnibus:                     3360.694   Durbin-Watson:                   2.003\n",
      "Prob(Omnibus):                  0.000   Jarque-Bera (JB):            11748.725\n",
      "Skew:                           1.171   Prob(JB):                         0.00\n",
      "Kurtosis:                       6.792   Cond. No.                     2.64e+15\n",
      "==============================================================================\n",
      "\n",
      "Warnings:\n",
      "[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.\n",
      "[2] The smallest eigenvalue is 3.74e-18. This might indicate that there are\n",
      "strong multicollinearity problems or that the design matrix is singular.\n",
      "-------------\n",
      "RMSE on train data: 150728.23819367346\n",
      "RMSE on test data: 163005.87024575213\n"
     ]
    }
   ],
   "source": [
    "print(print_model)\n",
    "print (\"-------------\")\n",
    "print (f\"RMSE on train data: {err_train}\")\n",
    "print (f\"RMSE on test data: {err_test}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
