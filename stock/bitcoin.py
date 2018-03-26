import datetime
import json
import numpy as np
from sklearn import preprocessing


from poloniex import Poloniex
polo = Poloniex()

# specify time period
beg_date = datetime.datetime(2018,3,10)
end_date = datetime.datetime(2018,3,20)
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

dates = np.reshape(dates, (len(dates), 1))
scaler = preprocessing.StandardScaler()
dates = scaler.fit_transform(dates)

#-------------------
# train regressor
from sklearn.svm import SVR
import matplotlib.pyplot as plt

svr_rbf = SVR(kernel='rbf',C=1e4, gamma=2, epsilon=0.001)
svr_rbf.fit(dates, prices)
y_rbf = svr_rbf.predict(dates)

svr_poly = SVR(kernel='poly', degree=3 ,C=1e4, gamma=2, epsilon=0.001)
svr_poly.fit(dates, prices)
y_poly = svr_poly.predict(dates)

# save model
import pickle
pickle.dump(svr_rbf, open( "svr_rbf_reg.pkl", "wb" ))
pickle.dump(svr_poly, open( "svr_poly_reg.pkl", "wb" ))
pickle.dump(scaler, open( "scaler.pkl", "wb" ))

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



