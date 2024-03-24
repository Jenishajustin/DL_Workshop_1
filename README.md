# Workshop 1 - NEURAL NETWORK MODEL BUILDING
## AIM
To build a TensorFlow sequential model for the given dataset.
## RUBRICS
1. Write a python code to convert the categorical input to numeric values - 20 Marks.

2. Write a python code to convert the categorical output to numeric values - 20 Marks.

3. Build a TensorFlow model with an appropriate activation function and the number of neurons in the output layer - 20 Marks.

4. Draw the neural network architecture for your model using the following website - 20 Marks.

5. Evaluating the Other submissions  - 20 Marks.

## NEURAL NETWORK ARCHITECTURE
![Screenshot 2024-03-24 084133](https://github.com/Jenishajustin/DL_Workshop_1/assets/119405070/3d5fd652-0315-4e24-b2ea-c6692d1ecf97)

## PROGRAM
```
Developed By : J.JENISHA
Reg. No. : 212222230056
```
```python
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
import pickle
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import BatchNormalization
import tensorflow as tf
import seaborn as sns
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import classification_report,confusion_matrix
import numpy as np
import matplotlib.pylab as plt

df = pd.read_csv('/content/mushrooms (1).csv')
df

df.columns
df.dtypes
df.shape

df.isnull().sum()
df['class'].unique()
df['cap-shape'].unique()
df['cap-surface'].unique()
df['cap-color'].unique()
df['bruises'].unique()
df['odor'].unique()
df['gill-attachment'].unique()
df['gill-spacing'].unique()
df['gill-size'].unique()
df['gill-color'].unique()
df['stalk-shape'].unique()
df['stalk-root'].unique()
df['stalk-surface-above-ring'].unique()
df['stalk-surface-below-ring'].unique()
df['stalk-color-above-ring'].unique()
df['stalk-color-below-ring'].unique()
df['veil-type'].unique()
df['veil-color'].unique()
df['ring-number'].unique()
df['ring-type'].unique()
df['spore-print-color'].unique()
df['population'].unique()
df['habitat'].unique()

# Categorical Input to Numeric Values

categories_list=[['x', 'b', 's', 'f', 'k', 'c'],
           ['s', 'y', 'f', 'g'],
           ['n', 'y', 'w', 'g', 'e', 'p', 'b', 'u', 'c', 'r'],
           ['t', 'f'],
           ['p', 'a', 'l', 'n', 'f', 'c', 'y', 's', 'm'],
            ['f', 'a'],
               ['c', 'w'],
            ['n', 'b'],
              ['k', 'n', 'g', 'p', 'w', 'h', 'u', 'e', 'b', 'r', 'y', 'o'],
             ['e', 't'],
                ['e', 'c', 'b', 'r', '?'],
                 ['s', 'f', 'k', 'y'],
                  ['s', 'f', 'y', 'k'],
                     ['w', 'g', 'p', 'n', 'b', 'e', 'o', 'c', 'y'],
                      ['w', 'p', 'g', 'b', 'n', 'e', 'y', 'o', 'c'],
                 ['p'],
                   ['w', 'n', 'o', 'y'],
                  ['o', 't', 'n'],
                 ['p', 'e', 'l', 'f', 'n'],
                ['k', 'n', 'u', 'h', 'w', 'r', 'o', 'y', 'b'],
                  ['s', 'n', 'a', 'v', 'y', 'c'],
                 ['u', 'g', 'm', 'd', 'p', 'w', 'l']
           ]
enc = OrdinalEncoder(categories=categories_list)
df_1 = df.copy()
df_1[['cap-shape',
             'cap-surface',
              'cap-color','bruises',
              'odor','gill-attachment','gill-spacing','gill-size',
      'gill-color','stalk-shape','stalk-root','stalk-surface-above-ring',
      'stalk-surface-below-ring','stalk-color-above-ring',
      'stalk-color-below-ring','veil-type','veil-color','ring-number',
      'ring-type','spore-print-color','population','habitat']] = enc.fit_transform(df_1[['cap-shape',
             'cap-surface',
              'cap-color','bruises',
              'odor','gill-attachment','gill-spacing','gill-size',
      'gill-color','stalk-shape','stalk-root','stalk-surface-above-ring',
      'stalk-surface-below-ring','stalk-color-above-ring',
      'stalk-color-below-ring','veil-type','veil-color','ring-number',
      'ring-type','spore-print-color','population','habitat']])
df_1.dtypes

# Categorical Output to Numeric values

le = LabelEncoder()
df_1['class'] = le.fit_transform(df_1['class'])
df_1.dtypes
df_1 = df_1.drop('gill-spacing',axis=1)
df_1.dtypes
# Calculate the correlation matrix
corr = df_1.corr()
# Plot the heatmap
sns.heatmap(corr,
        xticklabels=corr.columns,
        yticklabels=corr.columns,
        cmap="BuPu",
        annot= True)
sns.pairplot(df_1)
sns.distplot(df_1['habitat'])
plt.figure(figsize=(10,6))
sns.countplot(df_1['population'])
plt.figure(figsize=(10,6))
sns.boxplot(x='habitat',y='population',data=df_1)
plt.figure(figsize=(10,6))
sns.scatterplot(x='population',y='gill-attachment',data=df_1)
plt.figure(figsize=(10,6))
sns.scatterplot(x='population',y='odor',data=df_1)
df_1.describe()
df_1['class'].unique()
X=df_1[['cap-surface','cap-color','bruises','odor','gill-attachment','gill-size','gill-color','stalk-shape','stalk-root','stalk-surface-above-ring','stalk-surface-below-ring','population','habitat']].values
y1 = df_1[['class']].values
one_hot_enc = OneHotEncoder()
one_hot_enc.fit(y1)
y1.shape
y = one_hot_enc.transform(y1).toarray()
y.shape
y1[0]
y[0]
X.shape
X_train,X_test,y_train,y_test=train_test_split(X,y,
                                               test_size=0.33,
                                               random_state=50)
X_train[0]
X_train.shape
scaler_age = MinMaxScaler()
scaler_age.fit(X_train[:,2].reshape(-1,1))
X_train_scaled = np.copy(X_train)
X_test_scaled = np.copy(X_test)
X_train_scaled[:,2] = scaler_age.transform(X_train[:,2].reshape(-1,1)).reshape(-1)
X_test_scaled[:,2] = scaler_age.transform(X_test[:,2].reshape(-1,1)).reshape(-1)

# Tensorflow Model creation

Ai_brain = Sequential([
    # Hidden ReLU layers
                       Dense(units=5, activation='relu',input_shape=[13]),
                       Dense(units=3, activation='relu'),
    # Linear Output layer
                       Dense(units=2, activation="softmax")
                      ])
Ai_brain.compile(optimizer='adam',
                 loss= 'categorical_crossentropy',
                 metrics=['accuracy'])
early_stop = EarlyStopping(monitor='val_loss', patience=2)
Ai_brain.fit(x=X_train_scaled,y=y_train,
             epochs= 2000,
             batch_size= 256,
             validation_data=(X_test_scaled,y_test),
             )
metrics = pd.DataFrame(Ai_brain.history.history)
metrics.head()
metrics[['loss','val_loss']].plot()
x_test_predictions = np.argmax(Ai_brain.predict(X_test_scaled), axis=1)
x_test_predictions.shape
y_test_truevalue = np.argmax(y_test,axis=1)
y_test_truevalue.shape
print(confusion_matrix(y_test_truevalue,x_test_predictions))
print(classification_report(y_test_truevalue,x_test_predictions))
# Saving the Model
Ai_brain.save('mushroom_classification_model.h5')
# Saving the data
with open('mushroom_data.pickle', 'wb') as fh:
   pickle.dump([X_train_scaled,y_train,X_test_scaled,y_test,df_1,scaler_age,enc,one_hot_enc,le], fh)
# Loading the Model
ai_brain = load_model('mushroom_classification_model.h5')
# Loading the data
with open('mushroom_data.pickle', 'rb') as fh:
   [X_train_scaled,y_train,X_test_scaled,y_test,df_1,scaler_age,enc,one_hot_enc,le]=pickle.load(fh)

# Prediction for a Single input

x_single_prediction = np.argmax(ai_brain.predict(X_test_scaled[1:2,:]), axis=1)
print(x_single_prediction)
print(le.inverse_transform(x_single_prediction))
```
## OUTPUT
### Dataset
![Screenshot 2024-03-24 083011](https://github.com/Jenishajustin/DL_Workshop_1/assets/119405070/762f04b6-1d0f-499b-bece-6b850518578b)
![Screenshot 2024-03-24 083103](https://github.com/Jenishajustin/DL_Workshop_1/assets/119405070/4d1facad-8a4d-4eb7-bf06-4709f8833a86)
![Screenshot 2024-03-24 083123](https://github.com/Jenishajustin/DL_Workshop_1/assets/119405070/5c2eb0b6-40d7-4c16-97cb-e8590040de1e)

### Categorical Input to Numeric values
![Screenshot 2024-03-24 083151](https://github.com/Jenishajustin/DL_Workshop_1/assets/119405070/439011e2-fa8c-4225-8ef3-29dbc07248f5)

### Categorical Output to Numeric values
![Screenshot 2024-03-24 083230](https://github.com/Jenishajustin/DL_Workshop_1/assets/119405070/32d1f7f7-6670-4335-814f-a48b693e8db5)

### Model Creation and Training
![Screenshot 2024-03-24 083400](https://github.com/Jenishajustin/DL_Workshop_1/assets/119405070/2c841b34-aae7-4aef-ad55-28b0e2089f16)

### Loss and Accuracy
![Screenshot 2024-03-24 083453](https://github.com/Jenishajustin/DL_Workshop_1/assets/119405070/974c1e13-2f3a-44d8-a522-afe47f27bfd5)

![Screenshot 2024-03-24 083532](https://github.com/Jenishajustin/DL_Workshop_1/assets/119405070/7990554a-7580-4117-ac2a-fc76cc927a39)

### Confusion Matrix
![Screenshot 2024-03-24 083553](https://github.com/Jenishajustin/DL_Workshop_1/assets/119405070/4126053c-08cf-4370-ae0b-cc11e6c5925c)

### Classification Report
![Screenshot 2024-03-24 083640](https://github.com/Jenishajustin/DL_Workshop_1/assets/119405070/3153be02-846f-4999-9efd-61245611424f)

### Prediction for Single Input
![Screenshot 2024-03-24 083700](https://github.com/Jenishajustin/DL_Workshop_1/assets/119405070/2f32b898-2376-4cac-b699-88616466a129)

## RESULT
Therefore, Neural network model for classifaction problem has been executed successfully.
