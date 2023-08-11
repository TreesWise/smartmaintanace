
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, LSTM, Reshape
from joblib import load
import time

def Build_model(n_steps_in,n_steps_out,X_feat,Y_feat):
    class AttentionLayer(tf.keras.layers.Layer):
            def __init__(self,name=None):
                super(AttentionLayer, self).__init__(name=name)

            def build(self, input_shape):
                self.W = self.add_weight(shape=(input_shape[-1], 1), initializer="random_normal", trainable=True)
                super(AttentionLayer, self).build(input_shape)

            def call(self, inputs):
                attention_weights = tf.nn.softmax(tf.matmul(inputs, self.W), axis=1)
                weighted_inputs = inputs * attention_weights
                return tf.reduce_sum(weighted_inputs, axis=1)

    inputs = tf.keras.Input(shape=(n_steps_in, X_feat))
    x = LSTM(335, return_sequences=True)(inputs)
    x = LSTM(188, return_sequences=True, dropout=.06586127019804049)(x)
    x = LSTM(172, return_sequences=True, dropout=.1494075127203595)(x) 
    x = AttentionLayer(name='AttentionLayer')(x)    

    out = Dense(64)(x)
    out = Dense(128)(out)
    out = Dense(64)(out)

    out = Dense(n_steps_out*Y_feat)(out)
    out = Reshape((n_steps_out,Y_feat))(out)   
    model = Model(inputs = inputs, outputs = out)
    return model

class pdm_ts_model():
 
    cyl_count = 6 #may change depending upon vessel
    max_load = 84.19076333333332 #max engine load from 1 yr data for normalization
    look_back = 336
    forecast_horizon = 336

    def __init__(self,ts_features_file,ts_model,x_scale,y_scale,engine_normalized:bool,engine_number,res_loc): #raw_data,anomaly_path,Efd_features,feature,input_data
        self.ts_features_file = ts_features_file #input features list for TS
        self.ts_model_path = ts_model #saved ts model with path
        # self.Efd_features = Efd_features #list of all EFD features for iter on ml models ['Pscav','Pcomp','Pmax','Texh','Ntc','Ntc/Pscav','Pcomp/Pscav','PR']
        self.scaler_inp_train_m = load(x_scale) #saved inputs scaler model with path
        self.scaler_out_train_m = load(y_scale) #saved output scaler model with path
        self.engine_normalized = engine_normalized #True or False for applying engine based normalization
        self.engine_number = engine_number # '1' or '2' for corr engine
        self.res_loc = res_loc #loc where TS results save 'E:/python_codes1/ML_reference/1hr/TS_models_engineinout/TS_res/'
        #loading model
        self.ts_model = Build_model(self.look_back,self.forecast_horizon,52,9)
        self.ts_model.load_weights(self.ts_model_path)
    
    def Timeseries(self, new_data):
        t1 = time.time()
        #part for test data
        df = new_data 
        df = df[(df['Estimated engine load']>=30)&(df['Estimated engine load']<=100)]    
        df['PR'] = df['Firing Pressure Average'] - df['Compression Pressure Average']
        df['Ntc_Pscav'] = df['Turbocharger 1 speed'] / df['Scav. Air Press. Mean Value']
        df['Pcomp_Pscav'] = df['Compression Pressure Average'] / df['Scav. Air Press. Mean Value']
        #Normalizing based on engine load
        if self.engine_normalized == True:
            max_load = self.max_load
            print(max_load)
            df['Estimated engine load'] = df['Estimated engine load']/max_load
            for col in df.columns:
                if col == 'Estimated engine load':
                    df[col] = df['Estimated engine load']
                else:
                    df[col] = df[col]*df['Estimated engine load']     
                
        # df = df.iloc[-self.look_back:,] #taking last look_back period from new dataset
        print(df.index.min())
        print(df.index.max())
        print(df.shape)

        
        #predicting for individual cylinders
        ts_res_cyl = {}
        df_pred_load_wise = pd.DataFrame()
        for cyl in range(1,self.cyl_count+1):
            df['Exh. valve opening angle Cyl AVG'] = df['Exh. valve opening angle Cyl #0'+str(cyl)]
            df['GAV Timing Set Point Cyl AVG'] = df['GAV Timing Set Point Cyl #0'+str(cyl)]
            df['Exhaust Valve Closing Angle Setpoint Cyl AVG'] = df['Exhaust Valve Closing Angle Setpoint Cyl #0'+str(cyl)]
            df['PFI Timing Set Point Cyl AVG'] = df['PFI Timing Set Point Cyl #0'+str(cyl)]
            df['PFI Duration Set Point Cyl AVG'] = df['PFI Duration Set Point Cyl #0'+str(cyl)]
            df['Cyl. lub. distribution share below_PERC'] = (df['Cyl. lub. distribution share below piston']/df['Cyl. lub. distribution share into piston'])*100
            df['Cyl. lub. distribution share above_PERC'] = (df['Cyl. lub. distribution share above piston']/df['Cyl. lub. distribution share into piston'])*100
            df['Fuel Rail Pressure_diff'] = df['Mean Fuel Rail Pressure (display)'] - df['Main Fuel Rail Pressure']
            df['Firing Pr. Balancing Injection Offset Cyl_AVG'] = df['Firing Pr. Balancing Injection Offset Cyl #0'+str(cyl)]
            df['Fuel Pressure Actuator Setpoint_AVG'] = (df['Fuel Pressure Actuator Setpoint 1']+df['Fuel Pressure Actuator Setpoint 2']+df['Fuel Pressure Actuator Setpoint 3'])/3
            df['Fuel Pump Setpoint_AVG'] = (df['Fuel Pump Setpoint Master Controller']+df['Fuel Pump Setpoint Slave Controller'])/2
            df['Lubrication Oil Feed Rate Cyl AVG'] = df['Lubrication Oil Feed Rate Cyl #0'+str(cyl)]
            df['Lubrication Deadtime Feedback Cyl AVG'] = df['Lubrication Deadtime Feedback Cyl #0'+str(cyl)]
            df['Start of Injection Cyl_AVG'] = df['Start of Injection Cyl #0'+str(cyl)]
            df['Pilot Fuel Pressure diff'] = df['Pilot Fuel Pressure A']-df['Pilot Fuel Pressure B']
            df['Scavenge Air Temp. Piston Underside Cyl_AVG'] = df['Scavenge Air Temp. Piston Underside Cyl #0'+str(cyl)+'.1']
            #scaling X
            model_inputs = df[list(self.scaler_inp_train_m.feature_names_in_)]
            scaled_inp_test = self.scaler_inp_train_m.transform(model_inputs)
            pos = np.array(range(1,self.look_back+1))/self.look_back
            scaled_inp_test = np.append(scaled_inp_test,pos.reshape(-1,1),axis=1)
 
            X_new = scaled_inp_test.reshape((1, scaled_inp_test.shape[0], scaled_inp_test.shape[1]))
            predictons = self.ts_model.predict(X_new)
            #inverse scaling y
            y_pred_real = self.scaler_out_train_m.inverse_transform(predictons[-1].reshape(-1,len(self.scaler_out_train_m.feature_names_in_)))
            df_pred = pd.DataFrame(y_pred_real, columns=['TS_Pcomp', 'TS_Pscav','TS_Texh', 'TS_Ntc','TS_Pmax', 'TS_PR','TS_Ntc_Pscav','TS_Pcomp_Pscav','Estimated engine load'])
            df_pred.to_csv(self.res_loc+'ENG_{}_TS_res_Cyl_{}.csv'.format(self.engine_number,cyl), index=False)
            #print('Cylinder_'+str(cyl)+' timeseries prediction completed!!!')

        t2 = time.time()
        print('Time taken for TS prediction: ',t2-t1)
        return df.index.max()