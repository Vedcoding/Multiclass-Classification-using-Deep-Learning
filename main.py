import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
#read in data using pandas
train_df = pd.read_csv("train.csv")
#check data has been read in properly
print(train_df.head())

X = train_df.drop(columns=['final_result'])

#check that the target variable has been removed
print(X.head())

#create a dataframe with only the target column
y = train_df[['final_result']]

#view dataframe
print(y.head())
X_train, X_test, y_train, y_test = train_test_split(   X, y, test_size=0.33, random_state=42)

#create model
model = Sequential()

#get number of columns in training data
n_cols = X_train.shape[1]

#create model
model = Sequential()

from keras.callbacks import EarlyStopping
early_stopping_monitor = EarlyStopping(patience=5)
#add layers to model
variables_for_classification=4
model.add(Dense(250, activation='relu', input_shape=(n_cols,)))
model.add(Dense(250, activation='relu'))
model.add(Dense(250, activation='relu'))
model.add(Dense(4, activation='softmax'))
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
#train model
model.fit(X_train, y_train, epochs=30, validation_split=0.2, callbacks=[early_stopping_monitor])

