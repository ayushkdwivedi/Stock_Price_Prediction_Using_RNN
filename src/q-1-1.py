import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras import Sequential
from keras.layers import Dense, LSTM
from keras.layers import Dropout
import warnings
warnings.filterwarnings("ignore")

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
	timesteps = [20,50,75]
	cells = [30,50,80]
	hidden_layers = [2,3]
	epoch = 200
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
	
	
	for i in range(len(timesteps)):
		s = 1

		for j in range(len(hidden_layers)):

			for k in range(len(cells)):

				X_train, X_val, y_train, y_val = create_timeseries(scaled_data,timesteps[i])

				if hidden_layers[j] == 2:
					model = Sequential()
					model.add(LSTM(units=cells[k], return_sequences= True, input_shape=(X_train.shape[1],2)))
					model.add(Dropout(0.2))
					model.add(LSTM(units=cells[k]))
					model.add(Dropout(0.2))
					model.add(Dense(units=1))

				else:
					model = Sequential()
					model.add(LSTM(units=cells[k], return_sequences= True, input_shape=(X_train.shape[1],2)))
					model.add(Dropout(0.2))
					model.add(LSTM(units=cells[k], return_sequences=True))
					model.add(Dropout(0.2))
					model.add(LSTM(units=cells[k]))
					model.add(Dropout(0.2))
					model.add(Dense(units=1))

				model.summary()

				model.compile(optimizer='adam', loss='mean_squared_error')
				model.fit(X_train, y_train, epochs=epoch, batch_size=batch, verbose = 0)
				predicted_value= model.predict(X_val)

				plt.figure(i+2)
				plt.subplot(2,3,s)
				s+=1
				plt.plot(predicted_value, color= 'red', label = 'Predicted Stock')
				plt.plot(y_val, color='green', label = 'Actual Stock')
				plt.title("For {} Hidden Layers each with {} cells".format(hidden_layers[j],cells[k]))
				plt.xlabel("Days/Time")
				plt.ylabel('Predicted Stock Opening Price')
				plt.legend()
				plt.subplots_adjust(hspace=0.4)

				print('Simulation completed for Timesteps: {}, Hidden Layer: {}, Cells: {}'.format(timesteps[i],hidden_layers[j],cells[k]))
		plt.suptitle('Stock opening price prediction for using RNN for Time Step = {}'.format(timesteps[i]), fontsize=16, fontweight="bold")
	plt.show()
			
