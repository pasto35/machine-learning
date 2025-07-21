import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

if __name__ == '__main__':
    companies = pd.read_csv("./../datasets/1000_Companies.csv")
    # it takes all lines, all columns except last column (indexes from 0 to 3)
    x = companies.iloc[:, :-1].values
    # it takes all lines, last column (index 4)
    y = companies.iloc[:, 4].values
    # print(companies.head())

    # # dark - positive correlation. If item 'a' is growing and item 'b' is growing to
    # #        R&D - Profit = 0.9
    # #        Administration - Profit = 0.4
    # sns.heatmap(companies.corr(numeric_only=True), cmap="Greys")
    # plt.show()

    # transformation of State column (categorical values) to numbers
    column_transformer = ColumnTransformer(
        transformers=[
            ('onehot', OneHotEncoder(), [3]) # index 3 = 'State'
        ],
        remainder='passthrough'  # ignore others columns
    )
    x = column_transformer.fit_transform(x)
    # print(x)

    # Avoiding Dummy Variable Trap:
    # After one-hot encoding, new columns are created for the categorical variable 'State'.
    # For example, for States: New York, California, Florida,
    # the one-hot encoding creates three columns:
    # New York | California | Florida
    #    1     |     0      |    0
    #    0     |     1      |    0
    #    0     |     0      |    1
    #
    # These columns are linearly dependent because the value of one can be
    # calculated from the others:
    # New York = 1 - (California + Florida)
    #
    # This linear dependency can cause problems in regression models,
    # leading to multicollinearity and unstable coefficient estimates.
    #
    # To avoid this dummy variable trap, we drop one of the dummy variables,
    # typically the first one.
    #
    # This can be done automatically using OneHotEncoder with the parameter drop='first':
    # OneHotEncoder(drop='first')
    # or manually

    x = x[:, 1:]

    # Model
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
    regressor = LinearRegression()
    regressor.fit(x_train, y_train)

    # Use it
    y_pred = regressor.predict(x_test)
    # print(y_pred)
    # print(regressor.coef_)
    # print(regressor.intercept_)
    print(r2_score(y_test, y_pred))