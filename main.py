# from fastapi import FastAPI, Depends, HTTPException, status
# from fastapi.middleware.cors import CORSMiddleware
# from fastapi.security import OAuth2PasswordRequestForm
# from datetime import timedelta

# from config import ACCESS_TOKEN_EXPIRE_MINUTES
# from database_conn import database
# from helper import authenticate_user, create_access_token, get_current_active_user
# from custom_data_type import Token, User

# import pandas as pd
# from io import StringIO
# import json
# import numpy as np
# import time

# import torch
# from AE_ import data_load_preprocess, Transform_data, AE, recreation_loss



# import warnings
# warnings.filterwarnings('ignore')


# #time series model
# from TS_model import pdm_ts_model
# #ml model
# from ML_model import pdm_ml_model
# #fault mapping
# from Fault_Mapping import Faults_Mapping
# import yaml
# with open('cbm_yaml.yml','r') as file:
#     utility_dict = yaml.safe_load(file)

# app = FastAPI()


# # #model weight loading
# model = AE()
# model.load_state_dict(torch.load('./utils/AE/model/model_02082023.pt', map_location=torch.device('cpu')))
# model.eval()

# #TS model loading
# ts_features_file = pd.read_csv(utility_dict['imp_feats_path'])
# ts_model =  utility_dict['TS_model_path']#load ts model here
# x_scale = utility_dict['TS_scale_x']#load X_scaler model here
# y_scale = utility_dict['TS_scale_y']#load Y_scaler model here
# engine_normalized = utility_dict['bool']
# engine_number = utility_dict['engine_number']
# ts_res_loc = utility_dict['ts_res_loc']
# TS = pdm_ts_model(ts_features_file, ts_model,x_scale,y_scale,engine_normalized,engine_number,ts_res_loc)


# #ML model loading
# Efd_features = utility_dict['efd_features']
# ml_res_loc = utility_dict['ml_res_loc']
# ML = pdm_ml_model(Efd_features,engine_normalized,ts_res_loc,engine_number,ml_res_loc)



# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# @app.on_event("startup")
# async def database_connect():
#     await database.connect()
# @app.on_event("shutdown")
# async def database_disconnect():
#     await database.disconnect()


# #test endpoint
# @app.get("/test")
# async def test(User = Depends(get_current_active_user)):
#     return {"data": User.dict()}

# # Authentication
# @app.post("/token")
# async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
#     user = await authenticate_user(form_data.username, form_data.password)
#     if not user:
#         raise HTTPException(
#             status_code=status.HTTP_401_UNAUTHORIZED,
#             detail="Incorrect username or password",
#             headers={"WWW-Authenticate": "Bearer"},
#         )
#     access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
#     access_token = create_access_token(
#         data={"sub": user.username}, expires_delta=access_token_expires
#     )
#     return {"access_token": access_token, "token_type": "bearer"}


# async def preprocess_dataset(data):
#     pass


# def data_preprocess(raw_data_path):
#     df = pd.read_csv(raw_data_path, index_col='signaldate')
#     df = df[(df['Estimated engine load'] >= 30) & (df['Estimated engine load'] <= 100)]
#     #last 336 datapoints
#     df = df.iloc[-utility_dict['look_back']:,:]
#     return df

# @app.post("/forecast-14days")
# async def forecast_14days(current_user: User = Depends(get_current_active_user)):
#     start_time = time.time()
#     data = pd.read_csv(utility_dict['forecast_data_path'], index_col=utility_dict['index'])
    

#     #-------------------> TS_model calling
#     TS_result = TS.Timeseries(data)


#     #-------------------> AUTOENCODE
#     #data preprocessing
#     df = data_load_preprocess(utility_dict['preprocess_data_path'])
#     df_norm_obj_test = Transform_data(df)
#     df_norm_test = df_norm_obj_test.normalize()
#     #final data for model
#     df_tensor_test = torch.tensor(df_norm_test.values, dtype=torch.float32)
#     recreated_df_test = model(df_tensor_test).cpu().detach().numpy()
#     base_data_test = df_tensor_test.cpu().detach().numpy()
#     faulty_date = recreation_loss(base_data_test, recreated_df_test)
#     if len(faulty_date) > 0:
#         #data_from_AE = data.iloc[faulty_date,:]
#         data_from_AE = data.drop(index=faulty_date)
#     else:
#         data_from_AE = data
#     print('data_from_AE-----------',data_from_AE.shape)
    

#     #-------------------> ML_model calling
#     ML_result = ML.ML_models(data_from_AE)


#     #-------------------> Fault mapping
#     final_indx = [pd.Timestamp(TS_result)+pd.Timedelta(tim,'h') for tim in range(1,TS.forecast_horizon+1)] #for getting timestamps for forecast period, here delta is hourly based
#     fault_mat_loc = utility_dict['f_mat_path']
#     p=utility_dict['p_value'] #weight value for kpi calculations
#     mapping_loc = utility_dict['map_path']
#     output_dict = {}
#     for i in range(1,ML.cyl_count+1):#------------ML.cyl_count+1
#         ml_ress = pd.read_csv(ml_res_loc+'ENG_2_TS_ML_res_Cyl_{}.csv'.format(str(i)),index_col=False)
#         ff = Faults_Mapping(ml_ress,fault_mat_loc,Efd_features,p)
#         ff1,fault_ids = ff.Mapping()
#         ml_ress = pd.concat([ml_ress,ff1[fault_ids]],axis=1)
#         ml_ress['Date Time'] = final_indx
#         #for ordering columns
#         ml_ress = ml_ress[utility_dict['end_res_colorder']]
#         ml_ress.to_excel(mapping_loc+'mapping_res_cyl{}.xlsx'.format(i),index=False)
#         output_dict['Cyl_'+str(i)] = ml_ress.to_dict(orient='list')
#     # df_res_cyl_1 = pd.read_excel(mapping_loc+'mapping_res_cyl1.xlsx')
#     # df_res_cyl_2 = pd.read_excel(mapping_loc+'mapping_res_cyl2.xlsx')
#     # df_res_cyl_3 = pd.read_excel(mapping_loc+'mapping_res_cyl3.xlsx')
#     # df_res_cyl_4 = pd.read_excel(mapping_loc+'mapping_res_cyl4.xlsx')
#     # df_res_cyl_5 = pd.read_excel(mapping_loc+'mapping_res_cyl5.xlsx')
#     # df_res_cyl_6 = pd.read_excel(mapping_loc+'mapping_res_cyl6.xlsx')
#     # #return "success"
#     # return {'cyl_1':df_res_cyl_1.to_dict(orient='list')}

#     end_time = time.time()
#     print('total time taken for 14 days forecast',end_time-start_time)
#     return output_dict
    
    
























































































# # @app.post("/recommend_vendor")
# # async def fetch_data(userinput: UserInput, current_user: User = Depends(get_current_active_user)):
# #     #print(userinput.dict())
# #     user_inp =  userinput.dict()
# #     print(user_inp)
#     # try:
#     #     data =  query_database(query)
#     # except Exception as e:
#     #     try:
#     #         data = read_data_from_blob("vendor_dataset.csv")
#     #     except Exception as e:
#     #         print(e)
#     #         raise HTTPException(status_code=404, detail='No data found for the given input')

from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordRequestForm
from datetime import timedelta

from config import ACCESS_TOKEN_EXPIRE_MINUTES
from database_conn import database
from helper import authenticate_user, create_access_token, get_current_active_user
from custom_data_type import Token, User

import pandas as pd
from io import StringIO
import json
import numpy as np
import time

import torch
from AE_ import data_load_preprocess, Transform_data, AE, recreation_loss



import warnings
warnings.filterwarnings('ignore')


#time series model
from TS_model import pdm_ts_model
#ml model
from ML_model import pdm_ml_model
#fault mapping
from Fault_Mapping import Faults_Mapping
import yaml

with open('cbm_yaml.yml','r') as file:
    utility_dict = yaml.safe_load(file)


app = FastAPI()


# #model weight loading
model = AE()
model.load_state_dict(torch.load(utility_dict['AE_path'], map_location=torch.device('cpu')))
model.eval()

#TS model loading
ts_features_file = pd.read_csv(utility_dict['imp_feats_path'])
ts_model =  utility_dict['TS_model_path']#load ts model here
x_scale = utility_dict['TS_scale_x']#load X_scaler model here
y_scale = utility_dict['TS_scale_y']#load Y_scaler model here
engine_normalized = utility_dict['bool']
engine_number = utility_dict['engine_number']
ts_res_loc = utility_dict['ts_res_loc']
TS = pdm_ts_model(ts_features_file, ts_model,x_scale,y_scale,engine_normalized,engine_number,ts_res_loc)


#ML model loading
Efd_features = utility_dict['efd_features']
ml_res_loc = utility_dict['ml_res_loc']
ML = pdm_ml_model(Efd_features,engine_normalized,ts_res_loc,engine_number,ml_res_loc)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def database_connect():
    await database.connect()
@app.on_event("shutdown")
async def database_disconnect():
    await database.disconnect()


#test endpoint
@app.get("/test")
async def test(User = Depends(get_current_active_user)):
    return {"data": User.dict()}

# Authentication
@app.post("/token")
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    user = await authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}


async def preprocess_dataset(data):
    pass


def data_preprocess(raw_data_path):
    df = pd.read_csv(raw_data_path, index_col=utility_dict['index'])
    df = df[(df[utility_dict['engine_load']] >= 30) & (df[utility_dict['engine_load']] <= 100)]
    #last 336 datapoints
    df = df.iloc[-utility_dict['look_back']:,:]
    return df

@app.post("/forecast-14days")
async def forecast_14days(current_user: User = Depends(get_current_active_user)):
    start_time = time.time()
    data = pd.read_csv(utility_dict['forecast_data_path'], index_col=utility_dict['index'])
    

    #-------------------> TS_model calling
    TS_result = TS.Timeseries(data)


    #-------------------> AUTOENCODE
    #data preprocessing
    df = data_load_preprocess(utility_dict['preprocess_data_path'])
    df_norm_obj_test = Transform_data(df)
    df_norm_test = df_norm_obj_test.normalize()
    #final data for model
    df_tensor_test = torch.tensor(df_norm_test.values, dtype=torch.float32)
    recreated_df_test = model(df_tensor_test).cpu().detach().numpy()
    base_data_test = df_tensor_test.cpu().detach().numpy()
    faulty_date = recreation_loss(base_data_test, recreated_df_test)
    if len(faulty_date) > 0:
        #data_from_AE = data.iloc[faulty_date,:]
        data_from_AE = data.drop(index=faulty_date)
    else:
        data_from_AE = data
    print('data_from_AE-----------',data_from_AE.shape)
    

    #-------------------> ML_model calling
    ML_result = ML.ML_models(data_from_AE)


    #-------------------> Fault mapping
    final_indx = [pd.Timestamp(TS_result)+pd.Timedelta(tim,'h') for tim in range(1,TS.forecast_horizon+1)] #for getting timestamps for forecast period, here delta is hourly based
    fault_mat_loc = utility_dict['f_mat_path']
    p=utility_dict['p_value'] #weight value for kpi calculations
    mapping_loc = utility_dict['map_path']
    output_dict = {}
    for i in range(1,ML.cyl_count+1):#------------ML.cyl_count+1
        ml_ress = pd.read_csv(ml_res_loc+'ENG_2_TS_ML_res_Cyl_{}.csv'.format(str(i)),index_col=False)
        ff = Faults_Mapping(ml_ress,fault_mat_loc,Efd_features,p)
        ff1,fault_ids = ff.Mapping()
        ml_ress = pd.concat([ml_ress,ff1[fault_ids]],axis=1)
        ml_ress[utility_dict['index2']] = final_indx
        #for ordering columns
        ml_ress = ml_ress[utility_dict['end_res_colorder']]
        ml_ress.to_excel(mapping_loc+'mapping_res_cyl{}.xlsx'.format(i),index=False)
        output_dict['Cyl_'+str(i)] = ml_ress.to_dict(orient='list')
    end_time = time.time()
    print('total time taken for 14 days forecast',end_time-start_time)
    return output_dict
