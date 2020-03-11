import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from hmmlearn.hmm import GaussianHMM
import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)

def create_timeseries(scaled_data,timesteps):

    X=[]
    y=[]

    split = int(.2 * len(scaled_data))

    for i in range(timesteps, scaled_data.shape[0]):
        X.append(scaled_data[i-timesteps:i,[0,2]])
        y.append(scaled_data[i,1])
    
    X_train = np.array(X)
    y_train = np.array(y)

    X_val = X_train[:split+timesteps]
    y_val = y_train[:split+timesteps]

    X_train = X_train.reshape(X_train.shape[0],timesteps, 2)
    X_val = X_val.reshape(X_val.shape[0],timesteps, 2)

    return X_train, X_val, y_train, y_val


if __name__ == "__main__":

    data = pd.read_csv("../Dataset/GoogleStocks.csv")

    #----------------------------Hyperparameters-----------------------------------
    timesteps = 50
    cells = [30,50,80]
    n_hidden_states = 4
    epoch = 1
    batch = 35


    #----------------------------Dataset Preprocessing-----------------------------
    data["average"] = (data["high"]+data["low"])/2
    feature_data = data.iloc[:,[2,3,6]].values
    #----------------------------Dataset Visuallization----------------------------
    plt.figure(1)
    plt.subplot(1,2,1)
    plt.plot(feature_data[:,0])
    plt.title("Volume of Stocks Sold Vs Days/Time")
    plt.xlabel("Days/Time")
    plt.ylabel("Volume of Stocks Sold")

    plt.subplot(1,2,2)
    plt.plot(feature_data[:,2],color = 'red')
    plt.title("Average of Highest and Lowest Stock Price Vs Days/Time")
    plt.xlabel("Days/Time)")
    plt.ylabel("Volume of stocks traded")

    plt.suptitle('Dataset Visualization', fontsize=16, fontweight="bold")

    #------------------------------------RNN---------------------------------------
    mms= MinMaxScaler(feature_range=(0,1))
    scaled_data = mms.fit_transform(feature_data[:,:])
    ndata = scaled_data[:,[0,2]]
    X_val = scaled_data[:150,[0,2]]
    y_val = scaled_data[:150,1]
    

    # X_train, X_val, y_train, y_val = create_timeseries(scaled_data,timesteps)

    model = GaussianHMM(n_components=n_hidden_states)
    model.fit(ndata)
    predicted_value= model.predict(X_val)

    plt.figure(2)
    plt.plot(predicted_value, color= 'red', label = 'Predicted Stock')
    plt.plot(y_val, color='green', label = 'Actual Stock')
    # plt.title("For {} Hidden Layers each with {} cells".format(hidden_layers[j],cells[k]))
    plt.xlabel("Days/Time")
    plt.ylabel('Predicted Stock Opening Price')
    plt.legend()
    plt.show()
            
