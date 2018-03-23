from sklearn import tree
from sklearn import ensemble

# [height, weight, shoe size]

X = [[180,80,44], [177,70,43],[160,60,38],[154,54,37],[166,65,40],[159,55,37],
     [190,90,47],[175,64,39],[177,70,40],[171,75,42],[181,85,43]]

Y = [['male'],['female'],['female'],['female'],['male'],['male'],['male'],
     ['female'],['male'],['female'],['male']]

clf = tree.DecisionTreeClassifier()
clf = clf.fit(X,Y)

prediction = clf.predict([[170,60,39]])    

print (prediction)

clf2 = ensemble.RandomForestClassifier()
clf2 = clf.fit(X,Y)

prediction = clf2.predict([[170,60,39]])    

print (prediction)
