from datetime import datetime,timedelta
import re
import time
import pandas as pd
import numpy as np
import requests
class api_data_collection():
    def __init__(self,number_days):
        self.number_days = number_days
    def data_collect(self,login,pass1,col,col_dict,table):
        #col - column ids for data collection
        #col_dict - column names and ids dict for mapping
        cur_date = datetime.now()-timedelta(days=1)
        full_df = pd.DataFrame()
        for dayy in range(self.number_days):
            chck_date = cur_date-timedelta(days=dayy)   
            from_date = str(chck_date.year)+'-'+str(chck_date.month)+'-'+str(chck_date.day)+' 00:00:00'
            to_date = str(chck_date.year)+'-'+str(chck_date.month)+'-'+str(chck_date.day)+' 23:59:59'
            payload = {'login':login,'password':pass1}
            response = requests.post('https://wingd-api.e-vesseltracker.com/auth/login',headers=payload)
            data = requests.get('https://wingd-api.e-vesseltracker.com/api/ship/9878876/'+table+'?',params={'column':col,'from':from_date,'to':to_date},headers={'Token':response.json()['token']})
            try:
                df = pd.DataFrame(data.json()['response']).T
                df = df.rename(columns=col_dict) 
                df.index = pd.to_datetime(df.index) 
                df = df.resample('1H').bfill()
                df = df.bfill()
                #calculating mean,min,max for features
                for pat in [re.compile(r'Exh\. valve opening angle Cyl'),re.compile(r'Firing Pr\. Balancing Injection Offset Cyl'),
                            re.compile(r'Start of Injection Cyl')]:
                    matches = [item for item in list(df.columns) if re.search(pat, item)]
                    for i in df.index:
                        df.loc[i,pat.pattern.replace('\\','')+'_mean'] = df.loc[i,matches].mean()
                        df.loc[i,pat.pattern.replace('\\','')+'_min'] = df.loc[i,matches].min()
                        df.loc[i,pat.pattern.replace('\\','')+'_max'] = df.loc[i,matches].max()
                    # df.drop(columns=)  
            
                print(df.shape)  
            except:
                df = pd.DataFrame()
                print('Got error while data collection')
            # time.sleep(70)       
            full_df = pd.concat([full_df,df]) 
        full_df.reset_index(names='signaldate',inplace=True)
        full_df.sort_values(by='signaldate',ascending=True,inplace=True)
        full_df.set_index(full_df['signaldate'],inplace=True)
        full_df.drop(columns=['signaldate'],inplace=True)       
        return full_df #this will be hourly data data with all engine loads          