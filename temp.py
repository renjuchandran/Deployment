import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

sales_data1 = pd.read_csv('train1.csv')

sales_data1.rename({"Unnamed: 0":"a"}, axis="columns", inplace=True)

sales_data1.drop(["a"], axis=1, inplace=True)

sales_data1['TARGET_IN_EA'] =sales_data1['TARGET_IN_EA'].astype(str)
sales_data1['TARGET_IN_EA'] = sales_data1['TARGET_IN_EA'].str.replace(',', '')
sales_data1['TARGET_IN_EA'] = sales_data1['TARGET_IN_EA'].astype(int)

sales_data1['ACH_IN_EA'] =sales_data1['ACH_IN_EA'].astype(str)
sales_data1['ACH_IN_EA'] = sales_data1['ACH_IN_EA'].str.replace(',', '')
sales_data1['ACH_IN_EA'] = sales_data1['ACH_IN_EA'].astype(int)
sales_data1['SLSMAN_CD'].replace(regex=True,inplace=True,to_replace=r'\D',value=r'')
sales_data1['SLSMAN_CD'] = sales_data1['SLSMAN_CD'].astype(int)
sales_data1['PROD_CD'].replace(regex=True,inplace=True,to_replace=r'\D',value=r'')
sales_data1['PROD_CD'] = sales_data1['PROD_CD'].astype(int)
feature_cols=sales_data1.columns[:5]
  
X=sales_data1[feature_cols]

result_cols= sales_data1.columns[5:]

Y=sales_data1[result_cols]
from xgboost import XGBRegressor
regressor=XGBRegressor(max_depth=5)

#from sklearn.linear_model import LinearRegression
#regressor = LinearRegression()
regressor.fit(X,Y)





pickle.dump(regressor, open('model.pkl','wb'))

model = pickle.load(open('model.pkl','rb'))
#print(model.predict([[4, 300, 500]]))