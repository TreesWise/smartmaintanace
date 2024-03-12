from datetime import datetime,timedelta
import re
import pandas as pd
import requests
import pickle
import yaml   
from azure.storage.blob import ContainerClient


# class api_data_collection():
#     def __init__(self,number_days):
#         self.number_days = number_days
#     def data_collect(self,login,pass1,col,col_dict,table,table1,table2):
#         #col - column ids for data collection
#         #col_dict - column names and ids dict for mapping
#         cur_date = datetime.now()-timedelta(days=1)
#         full_df = pd.DataFrame()
#         for dayy in range(self.number_days):
#             chck_date = cur_date-timedelta(days=dayy)   
#             # from_date = str(chck_date.year)+'-'+str(chck_date.month)+'-'+str(chck_date.day)+' 00:00:00'
#             from_date = chck_date.strftime('%Y-%m-%d')+' 00:00:00'
#             # to_date = str(chck_date.year)+'-'+str(chck_date.month)+'-'+str(chck_date.day)+' 23:59:59'
#             to_date = chck_date.strftime('%Y-%m-%d')+' 23:59:59'
#             payload = {'login':login,'password':pass1}
#             response = requests.post('https://wingd-api.e-vesseltracker.com/auth/login',headers=payload)
#             data = requests.get('https://wingd-api.e-vesseltracker.com/api/ship/9878876/'+table+'?',params={'column':col['EMS'],'from':from_date,'to':to_date},headers={'Token':response.json()['token']})
#             data1 = requests.get('https://wingd-api.e-vesseltracker.com/api/ship/9878876/'+table1+'?',params={'column':col['ME'],'from':from_date,'to':to_date},headers={'Token':response.json()['token']})
#             data2 = requests.get('https://wingd-api.e-vesseltracker.com/api/ship/9878876/'+table2+'?',params={'column':col['EMS_Failures'],'from':from_date,'to':to_date},headers={'Token':response.json()['token']})
#             try:
#                 # EMS table
#                 df = pd.DataFrame(data.json()['response']).T
#                 df = df.rename(columns=col_dict) 
#                 df.index = pd.to_datetime(df.index) 
#                 df = df.loc[df.index<=to_date]
#                 # ME table  
#                 df1 = pd.DataFrame(data1.json()['response']).T
#                 df1 = df1.rename(columns=col_dict)
#                 df1.index = pd.to_datetime(df1.index)
#                 df1 = df1.loc[df1.index<=to_date]
#                 # Diesel mode
#                 df2 = pd.DataFrame(data2.json()['response']).T
#                 df2 = df2.rename(columns={'P669':'Diesel Model Active'})
#                 df2.index = pd.to_datetime(df2.index)
#                 df2 = df2.loc[df2.index<=to_date]
#                 #combining all tables data
#                 df = df.resample('1Min').bfill().ffill()
#                 df = pd.concat([df,df1,df2],axis=1)
#                 df = df[df['Diesel Model Active']==1]
#                 df = df.resample('1H').bfill().ffill()
#                 # df = df.bfill()
#                 #calculating mean,min,max for features
#                 for pat in [re.compile(r'Exh\. valve opening angle Cyl'),re.compile(r'Firing Pr\. Balancing Injection Offset Cyl'),
#                             re.compile(r'Start of Injection Cyl')]:
#                     matches = [item for item in list(df.columns) if re.search(pat, item)]
#                     for i in df.index:
#                         df.loc[i,pat.pattern.replace('\\','')+'_mean'] = df.loc[i,matches].mean()
#                         df.loc[i,pat.pattern.replace('\\','')+'_min'] = df.loc[i,matches].min()
#                         df.loc[i,pat.pattern.replace('\\','')+'_max'] = df.loc[i,matches].max()
#                     # df.drop(columns=)  
            
#                 print(df.shape)  
#             except:
#                 df = pd.DataFrame()
#                 print('Got error while data collection')
#             # time.sleep(70)       
#             full_df = pd.concat([full_df,df]) 
#         full_df.reset_index(names='signaldate',inplace=True)
#         full_df.sort_values(by='signaldate',ascending=True,inplace=True)
#         full_df.set_index(full_df['signaldate'],inplace=True)
#         full_df.drop(columns=['signaldate'],inplace=True)    
#         full_df = full_df.loc[full_df.index<=to_date]   
#         return full_df #this will be hourly data data with all engine loads          
class api_data_collection():
    def __init__(self,number_days):
        self.number_days = number_days
    def data_collect(self,login,pass1,col,col_dict,table,table1,table2):
        #col - column ids for data collection
        #col_dict - column names and ids dict for mapping
        cur_date = datetime.now()-timedelta(days=1)
        full_df = pd.DataFrame()
        with open('cbm_yaml.yml','r') as file:
            utility_dict = yaml.safe_load(file)
        container_client = ContainerClient.from_connection_string(
        utility_dict['connection_string'], container_name=utility_dict['container_name'])
        for dayy in range(1,self.number_days+1):
            chck_date = cur_date-timedelta(days=self.number_days-dayy)   
            # from_date = str(chck_date.year)+'-'+str(chck_date.month)+'-'+str(chck_date.day)+' 00:00:00'
            from_date = chck_date.strftime('%Y-%m-%d')+' 00:00:00'
            # to_date = str(chck_date.year)+'-'+str(chck_date.month)+'-'+str(chck_date.day)+' 23:59:59'
            to_date = chck_date.strftime('%Y-%m-%d')+' 23:59:59'
            payload = {'login':login,'password':pass1}
            try:
                response = requests.post('https://wingd-api.e-vesseltracker.com/auth/login',headers=payload)
                data = requests.get('https://wingd-api.e-vesseltracker.com/api/ship/9878876/'+table+'?',params={'column':col['EMS'],'from':from_date,'to':to_date},headers={'Token':response.json()['token']})
                data1 = requests.get('https://wingd-api.e-vesseltracker.com/api/ship/9878876/'+table1+'?',params={'column':col['ME'],'from':from_date,'to':to_date},headers={'Token':response.json()['token']})
                data2 = requests.get('https://wingd-api.e-vesseltracker.com/api/ship/9878876/'+table2+'?',params={'column':col['EMS_Failures'],'from':from_date,'to':to_date},headers={'Token':response.json()['token']})
            except:
                print('Wingd API is not accessible')
                continue
            try:
                # EMS table
                df = pd.DataFrame(data.json()['response']).T
                df = df.rename(columns=col_dict) 
                df.index = pd.to_datetime(df.index) 
                df = df.loc[df.index<=to_date]
                # ME table  
                df1 = pd.DataFrame(data1.json()['response']).T
                df1 = df1.rename(columns=col_dict)
                df1.index = pd.to_datetime(df1.index)
                df1 = df1.loc[df1.index<=to_date]
                # Diesel mode
                df2 = pd.DataFrame(data2.json()['response']).T
                df2 = df2.rename(columns={'P669':'Diesel Model Active'})
                df2.index = pd.to_datetime(df2.index)
                df2 = df2.loc[df2.index<=to_date]
                #combining all tables data
                df = df.resample('1Min').bfill().ffill()
                df = pd.concat([df,df1,df2],axis=1)
                df = df[df['Diesel Model Active']==1]
                if len(df)!=0:
                    df = df.resample('1H').bfill().ffill()
                    # df = df.bfill()
                    #calculating mean,min,max for features
                    for pat in [re.compile(r'Exh\. valve opening angle Cyl'),re.compile(r'Firing Pr\. Balancing Injection Offset Cyl'),
                                re.compile(r'Start of Injection Cyl')]:
                        matches = [item for item in list(df.columns) if re.search(pat, item)]
                        for i in df.index:
                            df.loc[i,pat.pattern.replace('\\','')+'_mean'] = df.loc[i,matches].mean()
                            df.loc[i,pat.pattern.replace('\\','')+'_min'] = df.loc[i,matches].min()
                            df.loc[i,pat.pattern.replace('\\','')+'_max'] = df.loc[i,matches].max()
                    
                    blob_client = container_client.get_blob_client('Data/'+utility_dict['Vessel_name']+'/Results/data_collection_history.pickle')
                    pickled_data = blob_client.download_blob().readall()
                    data_collection_hist = pickle.loads(pickled_data) 
                    data_collection_hist['Last_run_date_eng_'+table.split('_')[1]] = to_date.split(' ')[0]     
                    blob_client = container_client.get_blob_client('Data/'+utility_dict['Vessel_name']+'/Results/data_collection_history.pickle')
                    output_format_mapping_bytes = pickle.dumps(data_collection_hist)
                    blob_client.upload_blob(output_format_mapping_bytes,overwrite=True)   
                
                    print(df.shape)  
                else:
                    print('No Diesel data',to_date)
                    df = pd.DataFrame()    
                    blob_client = container_client.get_blob_client('Data/'+utility_dict['Vessel_name']+'/Results/data_collection_history.pickle')
                    pickled_data = blob_client.download_blob().readall()
                    data_collection_hist = pickle.loads(pickled_data) 
                    data_collection_hist['Last_run_date_eng_'+table.split('_')[1]] = to_date.split(' ')[0]     
                    blob_client = container_client.get_blob_client('Data/'+utility_dict['Vessel_name']+'/Results/data_collection_history.pickle')
                    output_format_mapping_bytes = pickle.dumps(data_collection_hist)
                    blob_client.upload_blob(output_format_mapping_bytes,overwrite=True)  
                       
            except:
                df = pd.DataFrame()
                print('Got error while data preprocessing while data collection')
            # time.sleep(70)       
            full_df = pd.concat([full_df,df]) 
        full_df.reset_index(names='signaldate',inplace=True)
        full_df.sort_values(by='signaldate',ascending=True,inplace=True)
        full_df.set_index(full_df['signaldate'],inplace=True)
        full_df.drop(columns=['signaldate'],inplace=True)    
        # full_df = full_df.loc[full_df.index<=to_date]  
        return full_df #this will be hourly data data with all engine loads  