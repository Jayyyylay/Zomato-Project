import pandas as pd
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import train_test_split

# Import the cleaned csv
df = pd.read_csv('Zomato_df1.csv')


# Drop the Unnecessary column
df.drop('Unnamed: 0', axis=1, inplace=True)

# Create the x and y
x = df.drop('rate',axis=1)
y= df['rate']

# Create the train and test set
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.33,random_state=42)

# Create the model
model_etr = ExtraTreesRegressor(n_estimators=500)

# Fit the model
model_etr.fit(x_train,y_train)

# Predict the model
etr_pred = model_etr.predict(x_test)

# evaluate the model
from sklearn.metrics import r2_score
print(f'{round(r2_score(y_test,etr_pred)*100,2)}%')

import pickle
# # Saving model to disk
pickle.dump(model_etr, open('model_etr2.pkl','wb'))