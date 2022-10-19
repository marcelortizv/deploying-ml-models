"""
This script run test for heroku deployment
"""
import requests

APP_URL = 'https://ml-heroku-fastapi.herokuapp.com/'

data = {
    "age": 32,
    "workclass": "Private",
    "education": "Some-college",
    "maritalStatus": "Married-civ-spouse",
    "occupation": "Exec-managerial",
    "relationship": "Husband",
    "race": "Black",
    "sex": "Male",
    "hoursPerWeek": 60,
    "nativeCountry": "United-States"
    }


r = requests.post(APP_URL, json=data)

assert r.status_code == 200

print("Response code: %s" % r.status_code)
print("Response body: %s" % r.json())
