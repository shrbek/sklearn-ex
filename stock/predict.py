import datetime
import json
import numpy as np
from sklearn import preprocessing


from poloniex import Poloniex
polo = Poloniex()

# specify time period
beg_date = datetime.datetime(2018,3,15)
end_date = datetime.datetime(2018,3,24)
print("data from: {} to {}".format(beg_date, end_date))

chart_data = polo.returnChartData(currencyPair='USDT_BTC',
								  period=300, 
								  start=beg_date.timestamp(), 
								  end=end_date.timestamp()
								 )


prices = []
dates = []
for chunk in chart_data:
	parsed_chunk = json.loads(str(chunk).replace("'", '"'))
	prices.append(parsed_chunk['weightedAverage'])
	dates.append(parsed_chunk['date'])

#print(prices)

#-------------------
import matplotlib.pyplot as plt
import pickle

scaler = pickle.load(open( "scaler.pkl", "rb" ))

dates = np.reshape(dates, (len(dates), 1))
dates = scaler.transform(dates)

svr_rbf = pickle.load(open( "svr_rbf_reg.pkl", "rb" ))
y_rbf = svr_rbf.predict(dates)

svr_poly = pickle.load(open( "svr_poly_reg.pkl", "rb" ))
y_poly = svr_poly.predict(dates)

# plot 
lw = 2
plt.scatter(dates, prices, color='darkorange', label='data')
plt.plot(dates, y_rbf, color='navy', lw=lw, label='RBF model')
plt.plot(dates, y_poly, color='red', lw=lw, label='POLY model')
plt.xlabel('date-time')
plt.ylabel('price')
plt.title('Support Vector Regression')
plt.legend()
plt.show()


