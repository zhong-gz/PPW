import sys
sys.path.insert(0, sys.path[0]+"/../") # add parent directory to path
sys.path.insert(0, sys.path[0]+"Algorithm")
import numpy as np
import random
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from Algorithm.plot import plot_fig
from Algorithm.alg_method_1 import method_1
from Algorithm.alg_RRM import RRM
from Algorithm.alg_RGD import RGD
from Algorithm.alg_PPNN import PPNN
from Algorithm.alg_Outside import TSA
from Algorithm.alg_DFO import DFO
from Algorithm.alg_PerGD import PerformativeGD

# problems parameters
seed_value = 42
num_iters = 100
d_list = [2,4,8,16] 
num_experiments = 10
map = 5
np.random.seed(seed_value)
random.seed(seed_value)
folder_path = 'RegionalSalesData/result/'

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
y = scaler.fit_transform(y)*0.9
X = scaler.fit_transform(X)*0.9 #avoid gradient disappear in gradient methods

# split_idx = int(len(X) * 0.02)
# X = X[:split_idx]
# y = y[:split_idx]

methods = ['RRM Linear Regression','RGD Linear Regression','Two-Stage Approach','PerfGD','RRM Neural Networks','DFO','PPW']

# RGD
model_gaps_avg,model_gaps_std,mse_list_start_avg,mse_list_start_std,mse_list_end_avg,mse_list_end_std,method_name = \
    RGD(X,y,num_iters,d_list,map = map,num_experiments = num_experiments,seed_value = seed_value)
file_name_npy = f"{folder_path}{method_name}.npz"
np.savez(file_name_npy, model_gaps_avg = model_gaps_avg, model_gaps_std = model_gaps_std,\
            mse_list_start_avg = mse_list_start_avg, mse_list_start_std = mse_list_start_std,\
            mse_list_end_avg = mse_list_end_avg, mse_list_end_std = mse_list_end_std)
print(f"Data saved to {file_name_npy}")

# method 1
model_gaps_avg,model_gaps_std,mse_list_start_avg,mse_list_start_std,mse_list_end_avg,mse_list_end_std,method_name = \
    method_1(X,y,num_iters,d_list,map = map,num_experiments = num_experiments,seed_value = seed_value)
file_name_npy = f"{folder_path}{method_name}.npz"
print(file_name_npy)
np.savez(file_name_npy, model_gaps_avg = model_gaps_avg, model_gaps_std = model_gaps_std,\
            mse_list_start_avg = mse_list_start_avg, mse_list_start_std = mse_list_start_std,\
            mse_list_end_avg = mse_list_end_avg, mse_list_end_std = mse_list_end_std)
print(f"Data saved to {file_name_npy}")

# RRM
model_gaps_avg,model_gaps_std,mse_list_start_avg,mse_list_start_std,mse_list_end_avg,mse_list_end_std,method_name = \
    RRM(X,y,num_iters,d_list,map = map,num_experiments = num_experiments,seed_value = seed_value)
file_name_npy = f"{folder_path}{method_name}.npz"
np.savez(file_name_npy, model_gaps_avg = model_gaps_avg, model_gaps_std = model_gaps_std,\
            mse_list_start_avg = mse_list_start_avg, mse_list_start_std = mse_list_start_std,\
            mse_list_end_avg = mse_list_end_avg, mse_list_end_std = mse_list_end_std)
print(f"Data saved to {file_name_npy}")

# ppnn
model_gaps_avg,model_gaps_std,mse_list_start_avg,mse_list_start_std,mse_list_end_avg,mse_list_end_std,method_name = \
    PPNN(X,y,num_iters,d_list,map = map,num_experiments = num_experiments,seed_value = seed_value)
file_name_npy = f"{folder_path}{method_name}.npz"
np.savez(file_name_npy, model_gaps_avg = model_gaps_avg, model_gaps_std = model_gaps_std,\
            mse_list_start_avg = mse_list_start_avg, mse_list_start_std = mse_list_start_std,\
            mse_list_end_avg = mse_list_end_avg, mse_list_end_std = mse_list_end_std)
print(f"Data saved to {file_name_npy}")

# outside the echo chamber
model_gaps_avg,model_gaps_std,mse_list_start_avg,mse_list_start_std,mse_list_end_avg,mse_list_end_std,method_name = \
    TSA(X,y,num_iters,d_list,map = map,num_experiments = num_experiments,seed_value = seed_value)
file_name_npy = f"{folder_path}{method_name}.npz"
np.savez(file_name_npy, model_gaps_avg = model_gaps_avg, model_gaps_std = model_gaps_std,\
            mse_list_start_avg = mse_list_start_avg, mse_list_start_std = mse_list_start_std,\
            mse_list_end_avg = mse_list_end_avg, mse_list_end_std = mse_list_end_std)
print(f"Data saved to {file_name_npy}")

# PerformativeGD
model_gaps_avg,model_gaps_std,mse_list_start_avg,mse_list_start_std,mse_list_end_avg,mse_list_end_std,method_name = \
    PerformativeGD(X,y,num_iters,d_list,map = map,num_experiments = num_experiments,seed_value = seed_value)
file_name_npy = f"{folder_path}{method_name}.npz"
np.savez(file_name_npy, model_gaps_avg = model_gaps_avg, model_gaps_std = model_gaps_std,\
            mse_list_start_avg = mse_list_start_avg, mse_list_start_std = mse_list_start_std,\
            mse_list_end_avg = mse_list_end_avg, mse_list_end_std = mse_list_end_std)
print(f"Data saved to {file_name_npy}")

# DFO (Two-timescale Derivative Free Optimization for Performative Prediction with Markovian Data)
model_gaps_avg,model_gaps_std,mse_list_start_avg,mse_list_start_std,mse_list_end_avg,mse_list_end_std,method_name = \
    DFO(X,y,num_iters,d_list,map = map,num_experiments = num_experiments,seed_value = seed_value)
file_name_npy = f"{folder_path}{method_name}.npz"
np.savez(file_name_npy, model_gaps_avg = model_gaps_avg, model_gaps_std = model_gaps_std,\
            mse_list_start_avg = mse_list_start_avg, mse_list_start_std = mse_list_start_std,\
            mse_list_end_avg = mse_list_end_avg, mse_list_end_std = mse_list_end_std)
print(f"Data saved to {file_name_npy}")

plot_fig(num_iters,d_list,folder_path,methods)
