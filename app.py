# importing files

import pandas as pd
import numpy as np
from datetime import datetime
from datetime import date
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import datetime
import streamlit as st


def CreateTempdf(data, country):
    temp_df = data[data["Country"] == country]  # filtering data based on country selected
    new_id = [x for x in range(1, len(temp_df)+1)]  # adding a new column to keep count of days by assigning id to dates
    # note : total days available in dataset = 869
    temp_df.insert(0, "new_id", new_id)
    
    return temp_df


def Prediction(final_data, days):
    req_col = final_data[['New_cases', 'New_deaths', 'Cumulative_deaths']]  # getting features to predict in first layer
    Result = np.zeros(shape=(3, 1))  # creating empty numpy array to store predicted results
    for i in range(len(req_col)):
        Result[i] = CreateModel(final_data, req_col.iloc[:, [i]], days, i)  # prediction for each feature
        if i == 2:
            break;
     
    return Result


def CreateModel(final_data, req_col, days, i):
    x = np.array(final_data['new_id']).reshape(-1, 1)
    y = np.array(req_col).reshape(-1, 1)
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    
    model = RandomForestRegressor(n_estimators=212, max_features='sqrt', max_depth=4)
    model.fit(x_train, y_train.ravel())
    test_predict2 = model.predict(x)  # prediction on available data of 869 days for the model performance graph

    if i == 2:
     # creating a graph for cumulative deaths
     test_predict22 = test_predict2.flatten()
     y2 = y.flatten()
     
     combine1 = np.vstack((test_predict22, y2)).T
     chart_data1 = pd.DataFrame(
     combine1,
     columns=['Predicted Values', 'True Values'])
     st.line_chart(chart_data1)
     
    return int(model.predict([[final_data['new_id'].iloc[-1] + days]]))  # prediction on added days according to input date


def days_difference(data, date):
    # getting number of days between last date available in dataset and date of prediction
    d0 = data['Date_reported'].max()
    d1 = datetime.datetime.strptime(d0, "%Y-%m-%d").date()
    date1 = date
    diff = (date1 - d1).days
    return diff


def Prediction_Layer2(new_df, input_new_cases_2d):
    x = new_df[['new_id', 'New_cases', 'New_deaths', 'Cumulative_deaths']]
    y = new_df[['Cumulative_cases']]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    model2 = RandomForestRegressor(n_estimators=212, max_features='sqrt', max_depth=4)
    model2.fit(x_train, y_train)
    
    predict = model2.predict(x)  # set of predictions for performance graph
    cumulative_cases1 = model2.predict(input_new_cases_2d)
    cumulative_cases_result = np.int_(cumulative_cases1)[0]  # prediction for input date

    # creating a graph for cumulative cases
    y = y.to_numpy().flatten()
    combine = np.vstack((predict, y)).T
    chart_data = pd.DataFrame(
    combine,
    columns=['Predicted Values', 'True Values'])
    st.line_chart(chart_data)

    return cumulative_cases_result
    

# Creating streamlit containers
header = st.container()
input_data = st.container()
prediction = st.container()


with header:
    st.title("Covid-19 Cases Prediction")
    st.text("Get prediction of total Covid-19 cases and deaths by choosing the country & date.")


with input_data:
    country_col, days_col = st.columns(2)  # creating streamlit columns for input

    df = pd.read_csv("covid_data.csv")

    countries_list = df['Country'].unique()
    Country = country_col.selectbox('Select Country', countries_list)  # getting country input
    new_df = CreateTempdf(df, Country)
    user_days = days_difference(new_df, days_col.date_input("Enter date", min_value=date.today()))  # getting date input
    

with prediction:
    st.markdown("<h1 style='text-align: center;font-weight:normal;'>Cumulative Deaths</h1>", unsafe_allow_html=True)

    # first layer of model to predict cumulative deaths
    input_new_cases = Prediction(new_df, user_days)
    input_new_cases = input_new_cases.flatten()
    cumulative_deaths = np.int_(input_new_cases[2])
    
    html_str = f"""<style>p.a {{font: bold {20}px Courier;}}</style><p class="a">Cumulative Deaths : {cumulative_deaths}</p>"""
    st.markdown(html_str, unsafe_allow_html=True)  # displaying cumulative deaths
    
    s_no = len(new_df)+user_days
    input_new_cases = np.insert(input_new_cases,0,s_no)  # adding new column to predicted first layer to keep track of days
    input_new_cases_2d = np.reshape(input_new_cases, (1, 4))

    st.markdown("<h1 style='text-align: center; font-weight:normal;'>Cumulative Cases</h1>", unsafe_allow_html=True)

    cumulative_cases_result = Prediction_Layer2(new_df, input_new_cases_2d)  # second layer for prediction of cumultive cases
    
    html_str1 = f"""<style>p.a {{font: bold {20}px Courier;}}</style><p class="a">Cumulative Cases : {cumulative_cases_result}</p>"""
    st.markdown(html_str1, unsafe_allow_html=True)  # displaying cumulative cases
    
    ratio = cumulative_cases_result/cumulative_deaths
    ratio_float = "{:.2f}".format(ratio)
    html_str2 = f"""<style>p.a {{font: bold {14}px Courier;}}</style><p class="a">Ratio of Cumulative cases to Cumulative Deaths : {ratio_float}</p>"""
    st.markdown(html_str2, unsafe_allow_html=True)
    
    html_str3 = f"""<style>p.a {{font:{14}px Courier;}}</style><p class="a">Higher the ratio of cumulative cases to cumulative deaths better is the Health Infrastructure of {Country}</p>"""
    st.markdown(html_str3, unsafe_allow_html=True)

    st.text('**Disclaimer : These graphs show performance of our model based on the\n'
            'dataset which ranges from 03-01-2020 to 20-05-2022 i.e total 869 days**')
    






