import pandas as pd
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
print(melbourne_data)


melbourne_data = melbourne_data.dropna(axis=0)


melbourne_data.describe

Y = melbourne_data.price

melbourne_features = [    'price',
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
X.head

melbourne_model = DecisionTreeRegressor(random_state=1)

melbourne_model.fit(X, Y)
