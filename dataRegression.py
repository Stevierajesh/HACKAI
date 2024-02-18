import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

melbourne_file_path = 'housing.csv'
melbourne_data = pd.read_csv(melbourne_file_path) 
melbourne_data.columns


for col in melbourne_data.columns:
    for index, value in melbourne_data[col].items():
        if value == 'yes':
            melbourne_data.at[index, col] = 1
        elif value == 'no':
            melbourne_data.at[index, col] = 0
        elif value == 'furnished':
            melbourne_data.at[index, col] = 2
        elif value == 'semi-furnished':
            melbourne_data.at[index, col] = 1
        elif value == 'unfurnished':
            melbourne_data.at[index, col] = 0

# Print the modified DataFrame
#print(melbourne_data)


melbourne_data = melbourne_data.dropna(axis=0)


#melbourne_data.describe

Y = melbourne_data.price

melbourne_features = [
    'area',
    'bedrooms',
    'bathrooms',
    'stories',
    'mainroad',
    'guestroom',
    'basement',
    'hotwaterheating',
    'airconditioning',
    'parking',
    'prefarea',
    'furnishingstatus']

# melbourne_features = ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']



# melbourne_features = ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']

X = melbourne_data[melbourne_features]

#X.describe
#X.head

melbourne_model = DecisionTreeRegressor(random_state=1)

# melbourne_model.fit(X, y)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=1)

train_data = X_train.join(y_train)

train_data
train_data.hist(figsize = (15,8))
plt.figure(figsize=(15,8))
sns.heatmap(train_data.corr(), annot=True, cmap="YlGnBu")
plt.figure(figsize=(15,8))
sns.scatterplot(x="area", y="price", data=train_data, hue="price", palette="coolwarm")
from sklearn.linear_model import LinearRegression

X_train, y_train = train_data.drop(['price'], axis=1), train_data['price']

reg = LinearRegression()

reg.fit(X_train, y_train)
# reg.score(X_train, y_train)
# reg.predict(X_test)

melbourne_features = [
    'area',
    'bedrooms',
    'bathrooms',
    'stories',
    'mainroad',
    'guestroom',
    'basement',
    'hotwaterheating',
    'airconditioning',
    'parking',
    'prefarea',
    'furnishingstatus']


sample = pd.read_csv('housing.csv', 
                    header=0, 
                    usecols=["area","bedrooms",	"bathrooms",	"stories",	"mainroad",	"guestroom",	"basement",	"hotwaterheating"	,"airconditioning","parking","prefarea","furnishingstatus"],
                    nrows=1)
print(sample[melbourne_features])
# reg.predict(sample[melbourne_features])

train_data = X_train.join(y_train)

train_data['area']=np.log(train_data['area']+1)
train_data['bedrooms']=np.log(train_data['bedrooms']+1)
train_data['bathrooms']=np.log(train_data['bathrooms']+1)
train_data['stories']=np.log(train_data['stories']+1)
train_data['price']=np.log(train_data['price']+1)
# train_data['bedrooms']=np.log(train_data['parking']+1)
from sklearn.linear_model import LinearRegression

X_train, y_train = train_data.drop(['price'], axis=1), train_data['price']

reg = LinearRegression()

reg.fit(X_train, y_train)
reg.predict(X_train)
print("Hello")

# reg.predict(sampleData)