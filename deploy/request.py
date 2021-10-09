import requests

url = 'http://localhost:5000/results'
r = requests.post(url,json={'fixed acidity':7, 'volatile acidity':0.34,
     'citric acid':0.48, 'chlorides':0.11, 'free sulfur dioxide': 12,
     'total sulfur dioxide':39, 'density':1, 'pH':3,
     'sulphates':1.26, 'alcohol':10.9})

print(r.json())