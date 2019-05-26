"""
 Python Project - 2.1  Machine Learning using KERAS & data visualization using BOKEH Plot libraries
========================================================================================================================
                                                                    Student Name    : Karthik Krishna Kamath
                                                                    Student ID      : 18201357
                                                                    Enrolled Course : Msc Data and Computational Science
========================================================================================================================

Project Synopsis: Model building is done using Keras and accuracy is compared with the model build in pytorch.

Note: The csv file for the project can be selected within the file open dialog box while running the code.
"""

# Importing Libraries needed for the project.

import numpy as np
import pandas as pd
import keras.models as km
import keras.layers as kl
import sklearn.model_selection as skms
import sklearn.preprocessing as skp
import tkinter as tk
from tkinter import filedialog
import bokeh.plotting as bp
from bokeh.io import show, output_file
from bokeh.palettes import Spectral11
from bokeh.transform import linear_cmap
import plotly.offline as po
import plotly.plotly as py
import plotly.figure_factory as pff
from plotly.offline import init_notebook_mode
init_notebook_mode()
import plotly.graph_objs as go
import torch, torch.nn as nn
from torch.autograd import Variable
from bokeh.models import ColumnDataSource
from keras.utils import  plot_model

"""
========================================================Functions=======================================================
"""


def openfile():
    root = tk.Tk()
    root.withdraw()
    filepath = filedialog.askopenfilename()
    return filepath

def missing_val_handling(df):               # df is the dataset
    missingVal = df.isnull().sum()
    percentageMissingVal: Union[float, Any] = missingVal*100/ len(df)
    table = pd.concat([missingVal, percentageMissingVal],
                      axis=1).rename(columns={0: 'Missing Values', 1: 'Percentage'})
    if missingVal == 0:
        print("There are no null values in the data set")
        return table
    else:
        drop = [col for col in df if (df[col].isnull().sum() / len(df) >= percentage)]
        df = df.drop(columns=drop)
        return df

def categorical_var_encoder(category_col_index, xArray):
    le = skp.LabelEncoder()
    for i in category_col_index:
        xArray[:, i] = le.fit_transform(xArray[:, i])
    return xArray


def keras_model_seq(x_ip):
    mod = km.Sequential()
    mod.add(kl.Dense(1024, input_dim=x_ip.shape[1], activation='relu'))
    mod.add(kl.Dropout(0.2))
    mod.add(kl.Dense(512, activation='relu'))
    mod.add(kl.Dropout(0.2))
    mod.add(kl.Dense(1, activation='relu'))
    mod.compile(loss='mean_squared_error', optimizer='adam', metrics=['mae'])
    mod.summary()
    return mod


def pytorch_model(D_in, h1, h2, D_out):
    model = nn.Sequential(nn.Linear(D_in, H1),
                         nn.ReLU(),
                         nn.Linear(H1, H2),
                         nn.ReLU(),
                         nn.Linear(H2, D_out),
                         nn.ReLU()
                         )
    return model


def epoch_iteration(ep, x, y):
    total_losses = []
    for i in range(ep):
        predicted_op = Pt_model(x)
        loss = loss_fn(predicted_op, y)
        total_losses.append(loss.data)
        if i % 100:
            print('Epoch: {}, Loss: {}'.format(i, loss))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return total_losses


# Importing Dataset

filepath = openfile()
avocado_data = pd.read_csv(filepath)

# Data Visualization

avocado_new_data = avocado_data
avocado_new_data['Date'] = pd.to_datetime(avocado_new_data['Date'])
avocado_new_data['month'] = avocado_new_data['Date'].apply(lambda x:x.month)
avocado_new_data['day'] = avocado_new_data['Date'].apply(lambda x:x.day)

"""
# ==================================================Plots done in Bokeh===================================================
"""

# PLOT:1, Barplot for plotting average Volume in each region.

places = avocado_new_data['region'].unique()
avgVolume_per_region = {}
for i in places:
    if i != '':
        avgVolume = sum(avocado_new_data[avocado_new_data['region'] == i]
                        ['Total Volume'])/list(avocado_new_data['region']==i).count(True)
        avgVolume_per_region[i] = avgVolume

averageVolume = list(avgVolume_per_region.values())
output_file("Average_Volume_in_eachRegion.html")
barplotVol = bp.figure(title='Average Volume in each Region', x_range=places)
barplotVol.xaxis.axis_label = "Regions"
barplotVol.yaxis.axis_label = "Average Volume"
barplotVol.vbar(x=places, top=averageVolume, width=0.5, legend='Average Volume')
barplotVol.legend.orientation = "horizontal"
barplotVol.legend.location = "top_center"
barplotVol.xaxis.major_label_orientation= "vertical"
show(barplotVol)


# PLOT:2, Barplot for plotting Average Price in each region

avg_price_per_region = {}
for i in places:
    if i != '':
        avgPrice = sum(avocado_new_data[avocado_new_data['region'] == i]
                       ['AveragePrice'])/list(avocado_new_data['region']==i).count(True)
        avg_price_per_region[i] = avgPrice

averagePrice = list(avg_price_per_region.values())

output_file("Average_volume_in_each_region.html")

mapper = linear_cmap(field_name='averagePrice', palette=Spectral11, low=min(averagePrice), high=max(averagePrice))
sources = ColumnDataSource(data=dict(places=places, averagePrice=averagePrice))

barplotPrice = bp.figure(title="Average Price in each Region", x_range=places, y_range=(0, 2))
barplotPrice.xaxis.axis_label = "Regions"
barplotPrice.yaxis.axis_label = "Average Price"
barplotPrice.vbar(x='places', top='averagePrice', width=0.2, source=sources, color=mapper, line_color=mapper)
barplotPrice.xaxis.major_label_orientation = "vertical"
show(barplotPrice)

#Plot:3 Plot for distribution of average price

price = list(avocado_data['AveragePrice'])
hist, edges = np.histogram(price, density=True, bins=50)

pricedist = bp.figure(title='Distribution of Average Price', tools='', background_fill_color="grey")
pricedist.quad(top=hist, bottom=0, left=edges[:-1], right=edges[1:], fill_color="navy", line_color="white", alpha=0.5)
pricedist.y_range.start = 0
pricedist.legend.location = "center_right"
pricedist.xaxis.axis_label = 'Price distribution'
pricedist.yaxis.axis_label = 'Frequency'
show(pricedist)


#Data Cleansing and Data Wrangling

#finding the missing values in the data

missingVal = list(avocado_data.isnull().sum())
if missingVal == 0:
    print("The dataset is not having null values")
else:
    print("Missing Values have founded in the dataset")

# Changing categorical variables into factors.
cat_col_index = [8, 9, 10]
X_in = avocado_data.drop(['Date', 'AveragePrice'], axis=1).values
y_out = avocado_data['AveragePrice'].values
X_in = categorical_var_encoder(cat_col_index, X_in)

# Standardization
mms = skp.MinMaxScaler()
X_std = mms.fit_transform(X_in)
y_out = y_out.reshape(-1, 1)
y_std = mms.fit_transform(y_out)

# Splitting the data into train set and test set
X_train, X_test, y_train, y_test = skms.train_test_split(X_std, y_std, test_size=0.2, random_state=0)

"""
KERAS MODEL
"""

batchsize= 1024
epoch= 50

print('/n======================================Keras model implementation==============================/n/n')
model = keras_model_seq(X_train)
model_history = model.fit(X_train, y_train, validation_data=[X_test, y_test], batch_size=batchsize, epochs=epoch)
predict_nn = model.predict(X_test)
evaluation_loss = model.evaluate(X_test, y_test, verbose=1)
print(evaluation_loss)


"""
Pytorch Model
"""

X_train_pt = Variable(torch.FloatTensor(X_train))
y_train_pt =Variable(torch.FloatTensor(y_train))
X_test_pt = Variable(torch.FloatTensor(X_test))
y_test_pt = Variable(torch.FloatTensor(y_test))

D_in, H1, H2, D_out = X_train_pt.shape[1], 150, 75, 1   # D_in is the input dimension. D_out is the output dimension.
# H1 and H2 are the hidden layer;

Pt_model = pytorch_model(D_in, H1, H2, D_out)
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(Pt_model.parameters(), lr=1e-3)
print('=====================================Pytorch Model Training details==========================/n')


total_train_loss = epoch_iteration(epoch, X_train_pt, y_train_pt)
total_train_loss = np.array(total_train_loss, dtype=np.float)


predicted_output = Pt_model(X_test_pt)

# Plotting loss value from model prediction of keras and Pyplot

output_file("loss_value_from _kerasmodel.html")
plotlossk = bp.figure(title="Models Loss value in each iteration", x_axis_label="Epoch", y_axis_label="Model Loss Value")
plotlossk.line(model_history.epoch, model_history.history['loss'], legend="Keras loss value", line_color='red', line_width=3)
plotlossk.line(range(0, epoch), total_train_loss, legend="Pytorch loss value", line_color='blue', line_width=3)
show(plotlossk)


"""
===================================End of Code==============================
"""