import sys
sys.path.insert(0, sys.path[0]+"/../") # add parent directory to path
sys.path.insert(0, sys.path[0]+"Algorithm")
import numpy as np
import random
import pandas as pd
from Algorithm.alg_method_1 import method_1
from Algorithm.plot_par_sen import plot_fig_par
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler,MinMaxScaler
import warnings
import os

# problems parameters
seed_value = 42
num_iters = 100
d_list = [2,4,8,16] 
num_experiments = 10
map = 5
np.random.seed(seed_value)
random.seed(seed_value)
folder_path = f'RegionalSalesData/result_small/'
os.makedirs(folder_path, exist_ok=True)

data=pd.read_csv('RegionalSalesData/US_Regional_Sales_Data.csv')
data= data.drop(columns = 'OrderNumber')
data = data.drop(columns = 'CurrencyCode')
df=data.copy()
#converting dates into numerical
df['ProcuredDate'] = df['ProcuredDate'].str.replace('[/-]', '', regex=True).astype(np.int64)
df['OrderDate'] = df['OrderDate'].str.replace('[/-]', '', regex=True).astype(np.int64)
df['ShipDate'] = df['ShipDate'].str.replace('[/-]', '', regex=True).astype(np.int64)
df['DeliveryDate'] = df['DeliveryDate'].str.replace('[/-]', '', regex=True).astype(np.int64)
df['Unit Cost'] = df['Unit Cost'].str.replace(',', '')
df['Unit Price'] = df['Unit Price'].str.replace(',', '')
df['Unit Cost'] = pd.to_numeric( df['Unit Cost'] )
df['Unit Price'] =pd.to_numeric(df['Unit Price'] )
numerical_features = df.select_dtypes(include=['int', 'float']).columns
categorical_features = df.select_dtypes(include=['object']).columns
numerical_data = df[numerical_features]
categorical_data = df[categorical_features]
df['OrderDate'] = pd.to_datetime(df['OrderDate'])
df['OrderDate_Year'] = df['OrderDate'].dt.year
df['OrderDate_Month'] = df['OrderDate'].dt.month
df['OrderDate_Day'] = df['OrderDate'].dt.day
df = df.drop(columns = 'OrderDate')
features_with_multiple_categories  = ['Sales Channel' , 'WarehouseCode']
encoder = LabelEncoder()
for feature in features_with_multiple_categories:
    df[feature] = encoder.fit_transform(categorical_data[feature])
X = df.drop(columns = 'Unit Price').values
y = df['Unit Price'].values.reshape(-1, 1)
# X[:,11] Unit cost
# X[:,9] Order Quantity

scaler = MinMaxScaler()
y_ini = scaler.fit_transform(y)*0.9
X_ini = scaler.fit_transform(X)*0.9 #avoid gradient disappear in gradient methods

ns = [0.05,0.1,0.2,0.4]

# for n_p in ns:
#     print('n_p = ',n_p)
#     n = int(n_p*100)#X_ini.shape[0])
#     random_indices = np.random.choice(X_ini.shape[0], size=n, replace=False)
#     X = X_ini[random_indices, :]
#     y = y_ini[random_indices, :]
#     # method 1
#     model_gaps_avg,model_gaps_std,mse_list_start_avg,mse_list_start_std,mse_list_end_avg,mse_list_end_std,method_name = \
#     method_1(X,y,num_iters,d_list,map = map,num_experiments = num_experiments,seed_value = seed_value,alpha_1 = 2.1)
#     file_name_npy = f"{folder_path}{n_p}.npz"
#     np.savez(file_name_npy, model_gaps_avg = model_gaps_avg, model_gaps_std = model_gaps_std,\
#                 mse_list_start_avg = mse_list_start_avg, mse_list_start_std = mse_list_start_std,\
#                 mse_list_end_avg = mse_list_end_avg, mse_list_end_std = mse_list_end_std)
#     print(f"Data saved to {file_name_npy}")

rmse_min = 0.1
rmse_max = 1.5
gap_min = 1e-7
gap_max = 2

plot_fig_par(num_iters,d_list,folder_path,ns,latex_text = "n",rmse_min=rmse_min, rmse_max=rmse_max, gap_min=gap_min, gap_max=gap_max)