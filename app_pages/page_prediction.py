import streamlit as st
import pandas as pd
from datetime import date
from src.data_management import load_housing_data, load_pkl_file
from src.machine_learning.predictive_analysis_ui import predict_price


def page_prediction():
    """
    Predict the sales price of any house based on most relevant features
    """
    # load predict price files
    version = 'v1'
    regression_pipe = load_pkl_file(
        f"outputs/ml_pipeline/predict_price/{version}/regression_pipeline.pkl")
    house_features = (pd.read_csv(
        f"outputs/ml_pipeline/predict_price/{version}/X_train.csv")
        .columns
        .to_list()
    )

    # Generate Live Data
    # check_variables_for_UI(price_features)
    st.write("### House Price Predictor Interface (BR2)")

    st.write("#### Predict the sale price of a house\
             based on these key features:")
    st.write("Fill in the form and click on the 'Predict Sale Price'\
         button to get a prediction.")

    X_live = DrawInputsWidgets()

    # predict on live data
    if st.button("Predict Sale Price"):
        price_prediction = predict_price(
            X_live, house_features, regression_pipe)

        if price_prediction == 1:
            predict_price(X_live, house_features, regression_pipe)


def check_variables_for_UI(house_features):
    """
    Displays the house's most influencial features to the user
    """
    st.write(
        f"* There are {len(house_features)}\
        features for the UI: \n\n {house_features}"
    )


def DrawInputsWidgets():
    """
    Displays widget for feature input
    and house predicted price
    """
    # load dataset
    df = load_housing_data()
    df['TotalSF'] = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF']

    percentageMin, percentageMax = 0.4, 2.0

    # we create input widgets only for 6 features
    col1, col2, col3, col4 = st.beta_columns(4)
    col5, col6, col7, col8 = st.beta_columns(4)

    # We are using these features to feed the ML pipeline
    # - values copied from check_variables_for_UI() result

    # create an empty DataFrame, which will be the live data
    X_live = pd.DataFrame([], index=[0])

    # from here on we draw the widget based on the variable type
    # (numerical or categorical) and set initial values
    with col1:
        feature = "GarageArea"
        st_widget = st.number_input(
            label=feature,
            min_value=int(df[feature].min()*percentageMin),
            max_value=int(df[feature].max()*percentageMax),
            value=int(df[feature].median()),
            step=50
        )

    X_live[feature] = st_widget

    with col2:
        feature = "GrLivArea"
        st_widget = st.number_input(
            label=feature,
            min_value=int(df[feature].min()*percentageMin),
            max_value=int(df[feature].max()*percentageMax),
            value=int(df[feature].median()),
            step=50
        )
    X_live[feature] = st_widget

    with col3:
        feature = "OverallQual"
        st_widget = st.number_input(
            label=feature,
            min_value=1,
            max_value=10,
            value=5,
            step=1
        )
    X_live[feature] = st_widget

    with col4:
        feature = "TotalBsmtSF"
        st_widget = st.number_input(
            label=feature,
            min_value=int(df[feature].min()*percentageMin),
            max_value=int(df[feature].max()*percentageMax),
            value=int(df[feature].median()),
            step=50
        )
    X_live[feature] = st_widget

    with col5:
        feature = "YearBuilt"
        st_widget = st.number_input(
            label=feature,
            min_value=int(df[feature].min()*percentageMin),
            max_value=date.today().year,
            value=int(df[feature].median()),
            step=1
        )
    X_live[feature] = st_widget

    return X_live
