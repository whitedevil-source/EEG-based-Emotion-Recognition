# -*- coding: utf-8 -*-
"""
Created on Thu Mar 24 22:07:09 2022

@author: SUVAM PAUL
"""
import mne
import numpy as np
import matplotlib as ml
import pywt
from scipy.signal import butter, sosfilt, sosfreqz

event_list = [1536, 1537, 1538, 1539, 1540, 1541, 1542]
f_s = 512
t_s = -2.0 #Trials of 5 seconds or Full length trial
t_f = 3.0
sample_s = int(t_s*f_s)
sample_f = int(t_f*f_s)

sub01_mi_run = ['motorimagination_subject1_run1.gdf','motorimagination_subject1_run2.gdf',
                'motorimagination_subject1_run3.gdf','motorimagination_subject1_run4.gdf',
                'motorimagination_subject1_run5.gdf','motorimagination_subject1_run6.gdf',
                'motorimagination_subject1_run7.gdf','motorimagination_subject1_run8.gdf',
                'motorimagination_subject1_run9.gdf','motorimagination_subject1_run10.gdf']
x = []
y = []
#have to run code below this for all 10 runs for subject1
for r in range(len(sub01_mi_run)):
    data = mne.io.read_raw_gdf(sub01_mi_run[r]) 
    eeg_data = data.get_data()
    info = data.info
    
    event_info = data._annotations.description
    event_onset = data._annotations.onset
    time_values = data.times
    print(eeg_data.shape)
    
    for e in range(len(event_info)):
      channel_data=[]
      if (int(event_info[e]) in event_list):
        event_sample = int(event_onset[e]*f_s)
        for ch in range(eeg_data.shape[0]):
          channel_data.append(eeg_data[ch, event_sample+sample_s:event_sample+sample_f])
        x.append(np.reshape(channel_data,(96,2560)))
        y.append(event_list.index(int(event_info[e]))+1)

def butter_bandpass(lowcut, highcut, fs, order=6):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        sos = butter(order, [low, high], analog=False, btype='band', output='sos')
        return sos

def butter_bandpass_filter(data, lowcut, highcut, fs, order=6):
        sos = butter_bandpass(lowcut, highcut, fs, order=order)
        y = sosfilt(sos, data)
        return y

t1 = np.arange(0, 5, 1/512)

def plotF(ar):
    t=t1.copy()
    #t = np.arange(0, 5, 1/512)
    
    wp = pywt.WaveletPacket(data=ar, wavelet='db4', maxlevel=6)
    #arr=wp.data
    #plt.plot(t,x[0][0], label='Original') #original
    
    #plt.plot(t[0:326],wp['aaa'].data, label='Beta Unfiltered') #beta unfiltered
    beta_filtered=butter_bandpass_filter(wp['aaa'].data, 13, 30, 512, order=6)
    plt.plot(t[0:326], beta_filtered, label='Beta') #beta
    
    #plt.plot(t[80:166],wp['aaaad'].data, label='Alpha Unfiltered') #alpha unfiltered
    alpha_filtered=butter_bandpass_filter(wp['aaaad'].data, 8, 13, 512, order=5)
    plt.plot(t[80:166],alpha_filtered, label='Alpha') #alpha
    
    plt.plot(t[40:86],wp['aaaaad'].data, label='Theta') #theta
    
    plt.plot(t[0:46],wp['aaaaaa'].data, label='Delta') #delta
    
    plt.legend()
    fig, axs = plt.subplots(nrows=1, ncols=1)
    #print(axs.shape)
    #axs

def fn(ar1):
    v=[]
    v.append(ar1['aaaaaa'].data)
    v.append(ar1['aaaaad'].data)
    v.append(ar1['aaaada'].data)
    v.append(ar1['aaaadd'].data)
    v.append(ar1['aaadaa'].data)
    v.append(ar1['aaadad'].data)
    v.append(ar1['aaadda'].data)
    v.append(ar1['aaaddd'].data)
    v.append(ar1['aadaaa'].data)
    v.append(ar1['aadaad'].data)
    v.append(ar1['aadada'].data)
    v.append(ar1['aadadd'].data)
    v.append(ar1['aaddaa'].data)
    v.append(ar1['aaddad'].data)
    v.append(ar1['aaddda'].data)
    v.append(ar1['aadddd'].data)
    v.append(ar1['adaaaa'].data)
    v.append(ar1['adaaad'].data)
    v.append(ar1['adaada'].data)
    v.append(ar1['adaadd'].data)
    v.append(ar1['adadaa'].data)
    v.append(ar1['adadad'].data)
    v.append(ar1['adadda'].data)
    v.append(ar1['adaddd'].data)
    v.append(ar1['addaaa'].data)
    v.append(ar1['addaad'].data)
    v.append(ar1['addada'].data)
    v.append(ar1['addadd'].data)
    v.append(ar1['adddaa'].data)
    v.append(ar1['adddad'].data)
    v.append(ar1['adddda'].data)
    v.append(ar1['addddd'].data)
    v.append(ar1['daaaaa'].data)
    v.append(ar1['daaaad'].data)
    v.append(ar1['daaada'].data)
    v.append(ar1['daaadd'].data)
    v.append(ar1['daadaa'].data)
    v.append(ar1['daadad'].data)
    v.append(ar1['daadda'].data)
    v.append(ar1['daaddd'].data)
    v.append(ar1['dadaaa'].data)
    v.append(ar1['dadaad'].data)
    v.append(ar1['dadada'].data)
    v.append(ar1['dadadd'].data)
    v.append(ar1['daddaa'].data)
    v.append(ar1['daddad'].data)
    v.append(ar1['daddda'].data)
    v.append(ar1['dadddd'].data)
    v.append(ar1['ddaaaa'].data)
    v.append(ar1['ddaaad'].data)
    v.append(ar1['ddaada'].data)
    v.append(ar1['ddaadd'].data)
    v.append(ar1['ddadaa'].data)
    v.append(ar1['ddadad'].data)
    v.append(ar1['ddadda'].data)
    v.append(ar1['ddaddd'].data)
    v.append(ar1['dddaaa'].data)
    v.append(ar1['dddaad'].data)
    v.append(ar1['dddada'].data)
    v.append(ar1['dddadd'].data)
    v.append(ar1['ddddaa'].data)
    v.append(ar1['ddddad'].data)
    v.append(ar1['ddddda'].data)
    v.append(ar1['dddddd'].data)
    return v

super_delta=[]
super_theta=[]
super_alpha=[]
super_beta=[]

for j in range(len(x)):
    
    storage_delta=[]
    storage_theta=[]
    storage_alpha=[]
    storage_beta=[]
    
    for i in range(61):
        delta=butter_bandpass_filter(x[j][i], 0.5, 4, 512, order=6)
        theta=butter_bandpass_filter(x[j][i], 4, 8, 512, order=6)
        alpha=butter_bandpass_filter(x[j][i], 8, 13, 512, order=6)
        beta=butter_bandpass_filter(x[j][i], 13, 30, 512, order=6)
        
        wp_delta = pywt.WaveletPacket(data=delta, wavelet='db4', maxlevel=6)
        wp_theta = pywt.WaveletPacket(data=theta, wavelet='db4', maxlevel=6)
        wp_alpha = pywt.WaveletPacket(data=alpha, wavelet='db4', maxlevel=6)
        wp_beta = pywt.WaveletPacket(data=beta, wavelet='db4', maxlevel=6)
        
        storage_delta.append(fn(wp_delta))
        storage_theta.append(fn(wp_theta))
        storage_alpha.append(fn(wp_alpha))
        storage_beta.append(fn(wp_beta))
        
    super_delta.append(storage_delta)
    super_theta.append(storage_theta)
    super_alpha.append(storage_alpha)
    super_beta.append(storage_beta)   

[d1, d2, d3, d4] = np.shape(super_delta)
[t1, t2, t3, t4] = np.shape(super_theta)
[a1, a2, a3, a4] = np.shape(super_alpha)
[b1, b2, b3, b4] = np.shape(super_beta)

matrix_delta_3d = np.asarray(super_delta)
matrix_theta_3d = np.asarray(super_theta)
matrix_alpha_3d = np.asarray(super_alpha)
matrix_beta_3d = np.asarray(super_beta)

#converting 3d matrix to 2d matrix, we are concatenating the shape[1] and shape[2] of the 3d matrix

matrix_delta_2d = np.reshape(matrix_delta_3d,(d1,d2*d3,d4))
matrix_theta_2d = np.reshape(matrix_theta_3d,(t1,t2*t3,t4))
matrix_alpha_2d = np.reshape(matrix_alpha_3d,(a1,a2*a3,a4))
matrix_beta_2d = np.reshape(matrix_beta_3d,(b1,b2*b3,b4))

matrix_2d = np.concatenate((matrix_delta_2d,matrix_theta_2d,matrix_alpha_2d,matrix_beta_2d), axis=2)
###############################################################################

x = np.asarray(x)
y = np.asarray(y)
y_new = np.zeros(420)
for i in range(len(y)):
    y_new[i] = y[i]-1
######################~Data_Preparation~#######################################

x_train_delta = matrix_delta_2d[:336,:,:]
x_train_theta = matrix_theta_2d[:336,:,:]
x_train_alpha = matrix_alpha_2d[:336,:,:]
x_train_beta = matrix_beta_2d[:336,:,:]
x_train_all = matrix_2d[:336,:,:]
y_train = y_new[:336]

x_val_delta = matrix_delta_2d[336:378,:,:]
x_val_theta = matrix_theta_2d[336:378,:,:]
x_val_alpha = matrix_alpha_2d[336:378,:,:]
x_val_beta = matrix_beta_2d[336:378,:,:]
x_val_all = matrix_2d[336:378,:,:]
y_val = y_new[336:378]

x_test_delta = matrix_delta_2d[378:,:,:]
x_test_theta = matrix_theta_2d[378:,:,:]
x_test_alpha = matrix_alpha_2d[378:,:,:]
x_test_beta = matrix_beta_2d[378:,:,:]
x_test_all = matrix_2d[378:,:,:]
y_test = y_new[378:]

##################~Classification_Models~######################################
import tensorflow
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.utils import plot_model, to_categorical
from tensorflow.keras.layers import Dense, Activation, Conv1D, MaxPooling1D, GlobalAveragePooling1D, Flatten, Reshape, Dropout, BatchNormalization, Input, LSTM
from tensorflow.keras.layers import concatenate
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score
import random
import scipy.io
import pandas as pd
import matplotlib.pyplot as plt

y_train = to_categorical(y_train)
y_val = to_categorical(y_val)

es = EarlyStopping(monitor='val_accuracy', verbose=1, patience=5)


#############################~1D-CNN Model~###################################

tensorflow.keras.backend.clear_session()
i1 = Input(shape=(x_train_all.shape[1],x_train_all.shape[2]))
b1 = BatchNormalization()(i1)
l1 = Conv1D(128, kernel_size=5,strides=1,activation='relu')(b1)
l2 = Conv1D(128, kernel_size=5,strides=1,activation='relu')(l1)
l3 = MaxPooling1D(pool_size=3, strides=3, padding="valid")(l2)
l4 = Conv1D(64, kernel_size=5,strides=1,activation='relu')(l3)
l5 = Conv1D(64, kernel_size=5,strides=1,activation='relu')(l4)
l6 = GlobalAveragePooling1D()(l5)
l7 = Dropout(0.5)(l6)
l8 = Dense(16, activation='relu')(l7)
output = Dense(7, activation='softmax')(l8)
model_1d_cnn = Model(inputs=i1, outputs=output)# summarize layers
 

model_1d_cnn.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
model_1d_cnn.fit(x_train_all, y_train,validation_data=(x_val_all, y_val),epochs=1, batch_size=8,verbose=1,callbacks=[es])
 
predictions_1d_cnn = model_1d_cnn.predict(x_test_all)
y_pred_1d_cnn = np.argmax(np.asarray(predictions_1d_cnn),axis=1)
 
model_int = Model(inputs=i1, outputs=l4)
xx = model_int.predict(x_test_all)
 
acc_1d_cnn = accuracy_score(y_test,y_pred_1d_cnn)
print(acc_1d_cnn)

#############################~LSTM Model~###################################

tensorflow.keras.backend.clear_session()
i1 = Input(shape=(x_train_all.shape[1],x_train_all.shape[2]))
x1 = LSTM(256,activation='tanh')(i1)
x1 = Dropout(0.5)(x1)
output = Dense(7, activation='softmax')(x1)
model_lstm = Model(inputs=i1, outputs=output)# summarize layers

model_lstm.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
model_lstm.fit(x_train_all, y_train,validation_data=(x_val_all, y_val),epochs=50, batch_size=8,verbose=1,callbacks=[es])

predictions_1d_cnn = model_lstm.predict(x_test_all)
y_pred_1d_cnn = np.argmax(np.asarray(predictions_1d_cnn),axis=1)

acc_1d_cnn = accuracy_score(y_test,y_pred_1d_cnn)
print(acc_1d_cnn)



#############################CNN-LSTM Model###################################
tensorflow.keras.backend.clear_session()

i1 = Input(shape=(x_train_all.shape[1],x_train_all.shape[2]))
b1 = BatchNormalization()(i1)
l1 = Conv1D(128, kernel_size=5,strides=1,activation='relu')(b1)
l2 = Conv1D(128, kernel_size=5,strides=1,activation='relu')(l1)
l3 = MaxPooling1D(pool_size=2, strides=3, padding="valid")(l2)
l4 = Conv1D(64, kernel_size=5,strides=1,activation='relu')(l3)
l5 = Conv1D(64, kernel_size=5,strides=1,activation='relu')(l4)
# l6 = GlobalAveragePooling1D()(l5)
l61 = LSTM(256,activation='tanh')(l5)
l7 = Dropout(0.5)(l61)
l8 = Dense(16, activation='relu')(l7)
output = Dense(7, activation='softmax')(l8)
model_1d_cnn = Model(inputs=i1, outputs=output)# summarize layers

model_1d_cnn.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
model_1d_cnn.fit(x_train_all, y_train,validation_data=(x_val_all, y_val),epochs=1, batch_size=8,verbose=1,callbacks=[es])

predictions_1d_cnn = model_1d_cnn.predict(x_test_all)
y_pred_1d_cnn = np.argmax(np.asarray(predictions_1d_cnn),axis=1)

model_int = Model(inputs=i1, outputs=l4)
xx = model_int.predict(x_test_all)

acc_1d_cnn = accuracy_score(y_test,y_pred_1d_cnn)
print(acc_1d_cnn)
