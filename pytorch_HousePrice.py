"""
Python Project -2.2 Machine Learning using Pytorch and Data visualization using Plotly libraries.
========================================================================================================================
                                                                    Student Name: Karthik Krishna Kamath
                                                                    Student ID  : 18201357
                                                                    Course      : MSc in Data and Computational Science
========================================================================================================================


Project Synopsis: The machine learning model is created using pytorch and data visualization is done using the Plotly
library.


Note: The csv file for the project can be selected while running the code from the dialog box getting opened.
"""
"""
=====================================Importing Libraries for the Project================================================
"""

import pandas as pd
import sklearn.model_selection as skms
import sklearn.preprocessing as skp
import tkinter as tk
import tkinter.filedialog as tkf
import torch, torch.nn as nn
from torch.autograd import Variable
import plotly.offline as po
from plotly.offline import init_notebook_mode
init_notebook_mode()
import plotly.graph_objs as go


"""
=======================================================Functions========================================================
"""

def openfile():
    root = tk.Tk()
    root.withdraw
    filepath = tkf.askopenfilename()
    return filepath


def missing_val_handling(df):               # df is the dataset
    missingVal = df.isnull().sum()
    percentageMissingVal: Union[float, Any] = missingVal*100/ len(df)
    table = pd.concat([missingVal, percentageMissingVal],
                      axis=1).rename(columns={0: 'Missing Value count', 1: 'Percentage null Values with total'})
    return table


def drop_missing_val(df, percentage):       # df is the dataset and percentage value should be provided between 0 and 1
    drop = [col for col in df if(df[col].isnull().sum()/len(df) >= percentage)]
    df = df.drop(columns=drop)
    return df


def categorical_data_handler(df, c_index):
    enc = skp.LabelEncoder()
    for i in c_index:
        df[:, i] = enc.fit_transform(df[:, i])
    return df


def sales_per_month(df, month, sales):
    monthlySales = df['Price'].loc[df['month'] == month].sum()/sales
    return monthlySales


# Importing Dataset
filepath = openfile()
housePriceData = pd.read_csv(filepath)

# Data Visualization
"""
====================================================Plots in Plotly=====================================================
"""

# PLOT 1: Price distribution of houses sold in Melbourne

plot1 = go.Histogram(x=housePriceData['Price'])
plotstyle = go.Layout(title='Distribution of Housing Price in Melbourne',
                      xaxis=dict(range=[0, 7e6], title='Price in Australian dollars',
                                 titlefont=dict(family='Courier New, monospace', size=18, color='black')),
                      yaxis=dict(title='Frequency',
                                 titlefont=dict(family='Courier New, monospace', size=18, color='black')))
plotdata = [plot1]
hist_fig = go.Figure(data=plotdata, layout=plotstyle)
po.plot(hist_fig, filename='housingPrice_distribution.html')


# PLOT 2: Statistical value of housing price in Melbourne

housePriceData_new = housePriceData
housePriceData_new['Date'] = pd.to_datetime(housePriceData_new['Date'])
housePriceData_new['month'] = housePriceData_new['Date'].apply(lambda x:x.month)
housePriceData_new['year'] = housePriceData_new['Date'].apply(lambda x:x.year)

data1 = housePriceData_new.groupby(["Date"]).Price.mean()
data1 = data1.reset_index()
data2 = housePriceData_new.groupby(["Date"]).Price.max()
data2 = data2.reset_index()
data3 = housePriceData_new.groupby(["Date"]).Price.min()
data3 = data3.reset_index()

line_pt1 = go.Scatter(x=data1.Date, y=data1.Price, mode="lines",
                      line=dict(color='orange', width=2), name='Average')
line_pt2 = go.Scatter(x=data2.Date, y=data2.Price, mode="lines",
                      line=dict(color='grey', width=2), name='Maximum')
line_pt3 = go.Scatter(x=data3.Date, y=data3.Price, mode="lines",
                      line=dict(color='lightblue', width=2), name='Minimum')

lineplot_data = [line_pt1, line_pt2, line_pt3]
lineplotstyle = go.Layout(title='Statistical values of Housing price in Melbourne',
                          xaxis=dict(title='Sale date',
                                     titlefont=dict(family='Courier New, monospace', size=18, color='black')),
                          yaxis=dict(title='Price',
                                     titlefont=dict(family='Courier New, monospace', size=18, color='black')))

lineplot = go.Figure(data=lineplot_data, layout=lineplotstyle)
po.plot(lineplot, filename='Avg_Max_Min_housing_sale_price.html')


# PLOT 3: Pie chart of sales in each month

sale_total = housePriceData_new['Price'].sum()

jan_sales = sales_per_month(housePriceData_new, 1, sale_total)
feb_sales = sales_per_month(housePriceData_new, 2, sale_total)
mar_sales = sales_per_month(housePriceData_new, 3, sale_total)
apr_sales = sales_per_month(housePriceData_new, 4, sale_total)
may_sales = sales_per_month(housePriceData_new, 5, sale_total)
jun_sales = sales_per_month(housePriceData_new, 6, sale_total)
jul_sales = sales_per_month(housePriceData_new, 7, sale_total)
aug_sales = sales_per_month(housePriceData_new, 8, sale_total)
sep_sales = sales_per_month(housePriceData_new, 9, sale_total)
oct_sales = sales_per_month(housePriceData_new, 10, sale_total)
nov_sales = sales_per_month(housePriceData_new, 11, sale_total)
dec_sales = sales_per_month(housePriceData_new, 12, sale_total)

monthly_sales = [jan_sales, feb_sales, mar_sales, apr_sales, may_sales, jun_sales, jul_sales,
                 aug_sales, sep_sales, oct_sales, nov_sales, dec_sales]

legend_labels = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August',
                 'September', 'October', 'November', 'December']

chart_colors = ['orange', 'yellow', 'green', 'violet', 'red', 'rose',
                'purple', 'grape', 'grey', 'blue', 'lightblue', 'lime']

pieplot = go.Pie(labels=legend_labels, values=monthly_sales, hoverinfo='label+percent',
                 marker=dict(colors=chart_colors, line=dict(color='darkgrey', width=2)))

piechart_data = [pieplot]
titlename = go.Layout(title="Monthly sales percentage of housing price in Melbourne")
piechart = go.Figure(data=piechart_data, layout=titlename)
po.plot(piechart, filename='monthly_housing_sale_distribution.html')


# Data Pre-processing

missing_val_handling(housePriceData)                # Gives a table that show the missing value.

housePriceData = drop_missing_val(housePriceData, .55)  # Drop columns for percentage of missing values more than 55%
housePriceData = housePriceData.dropna(subset=['Price', 'SellerG', 'CouncilArea', 'Regionname'], how='any')

# Address columns is irrelevant and doesn't contribute much in price prediction hence we can drop the column,.
housePriceData = housePriceData.drop(['Address'], axis=1)

housePriceData = housePriceData.fillna(0)

X_input = housePriceData.drop(['Date', 'Price'], axis=1)
# Categorical label encoding

df_categoryName = list(X_input.select_dtypes(include=['object']).copy())
df_col_index = [X_input.columns.get_loc(c) for c in X_input.columns if c in df_categoryName]
X_input = X_input.values
X_input = categorical_data_handler(X_input, df_col_index)



y_target = housePriceData['Price'].values
y_target = y_target.reshape(-1, 1)

# Standardisation

sc = skp.StandardScaler()
X_input = sc.fit_transform(X_input)
y_target = sc.fit_transform(y_target)


# splitting the data to training and test sets
X_train, X_test, y_train, y_test = skms.train_test_split(X_input, y_target, test_size=0.2, random_state=0)


"""
=======================================Pytorch model implementation===================================
"""
X_train_pt = Variable(torch.FloatTensor(X_train))
y_train_pt = Variable(torch.FloatTensor(y_train))
X_test_pt = Variable(torch.FloatTensor(X_test))
y_test_pt = Variable(torch.FloatTensor(y_test))

D_in, H1, H2, H3, D_out = X_train.shape[1], 200, 100, 50, 1
# D_in is the input dimension; H1,H2, H3 are hidden layer; D_out is output dimension

Pt_model = nn.Sequential(nn.Linear(D_in, H1),
                         nn.ReLU(),
                         nn.Linear(H1, H2),
                         nn.ReLU(),
                         nn.Linear(H2, H3),
                         nn.ReLU(),
                         nn.Linear(H3, D_out),
                         nn.RReLU()
                         )

loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(Pt_model.parameters(), lr=1e-2)

print('=====================================Pytorch Model Epoch details==========================/n')

Epochs = 256
total_train_loss = []
for i in range(Epochs):
    predicted_op = Pt_model(X_train_pt)
    loss = loss_fn(predicted_op, y_train_pt)
    total_train_loss.append(loss.data)
    if i % 100:
        print('Epoch: {}, Loss: {}'.format(i, loss))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

predicted_output = Pt_model(X_test_pt)

"""
=============================end of code============================
"""