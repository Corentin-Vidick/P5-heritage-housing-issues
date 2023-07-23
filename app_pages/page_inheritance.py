import streamlit as st
import pandas as pd
from datetime import date
from src.data_management import load_pkl_file, load_heritage_data
from src.machine_learning.predictive_analysis_ui import predict_inherited_house_price


def page_inheritance():
    """
    Predicts the sales price of the inherited properties
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

    st.write("### Predicting sales price of inherited houses (BR2)")
    st.info(
        f"* **2** - The client is interested in predicting the house sale\
            price from her four inherited houses and any other house\
            in Ames, Iowa."
    )

    # Predict sales prices of inherited houses
    st.write(f"###### Predicted sales price of the four inherited houses\n"
             f"* See PredictedSalePrice column in the table below.")

    X_inherited = load_heritage_data()
    X_inherited['TotalSF'] = X_inherited['TotalBsmtSF'] + \
        X_inherited['1stFlrSF'] + X_inherited['2ndFlrSF']
    summed_price = 0
    predicted_sale_price = []
    for i in range(X_inherited.shape[0]):
        pprice = predict_inherited_house_price(
            X_inherited.iloc[[i,]], house_features, regression_pipe)
        predicted_sale_price.append(round(pprice))
        summed_price = summed_price + pprice
        summed_price = round(summed_price)
    X_inherited = X_inherited.filter(house_features)
    X_inherited['PredictedSalePrice'] = predicted_sale_price
    st.write(X_inherited.head())
    st.write(
        f"* Summed price: **${summed_price}** \n"
        f"* Features used: **{X_inherited.columns.to_list()[:-1]}**.\n"
        f"The Machine Learning model successfully predicted the sale\
        prices of the 4 inherited houses, and we were able to find\
        the summed value of the properties in question."
    )
