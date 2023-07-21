import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from src.data_management import load_housing_data, load_pkl_file
from src.machine_learning.evaluate_reg import regression_performance, regression_evaluation_plots


def page_technical_info():
    """
    Displays ML pipeline, feature importance and ML and regression
    performance plots 
    """
    # load price pipeline files
    version = 'v1'
    price_pipe = load_pkl_file(
        f"outputs/ml_pipeline/predict_price/{version}/regression_pipeline.pkl")
    # price_feat_importance = plt.imread(
    #     f"outputs/ml_pipeline/predict_price/{version}/features_importance.png")
    X_train = pd.read_csv(
        f"outputs/ml_pipeline/predict_price/{version}/X_train.csv")
    X_test = pd.read_csv(
        f"outputs/ml_pipeline/predict_price/{version}/X_test.csv")
    y_train = pd.read_csv(
        f"outputs/ml_pipeline/predict_price/{version}/y_train.csv")
    y_test = pd.read_csv(
        f"outputs/ml_pipeline/predict_price/{version}/y_test.csv")

    st.write("### ML Pipeline: Predict House Price")
    # display pipeline training summary conclusions
    st.info(
        f"* To answer BR2, we wanted to train a Regressor model and tune the pipeline aiming to ensure\
             at least 0.75 accuracy in predicting the sales price of a property with a given set of\
                 attributes. We were able to achieve this success metric, but we trained different\
                 versions of the model to make sure that we don't pverlook potential improvement.\n"
        f"* The pipeline performance for the best model on the train and test set is\
             R2 == 0.84 and R2 == 0.77 respectively.\n"
        f"* We present the pipeline steps, best features list along with feature importance plot, pipeline performance and regression performance report below. "
    )
    st.write("---")

    # show pipeline steps
    st.write("* ML pipeline to predict sales prices of houses ")
    st.code(price_pipe)
    st.write("---")

    # show best features
    st.write("* The features the model was trained and their importance")
    st.write(X_train.columns.to_list())
    # st.image(price_feat_importance)
    st.write("---")

    st.write("### Pipeline Performance")
    st.write("##### Performance goal of the predictions:\n")
    st.write("* We agreed with the client an R2 score of at least 0.75 on the train set as well as on the test set.")
    st.write(f"* Our ML pipeline performance shows that our model performance\
         metrics have been successfully satisfied.")
    regression_performance(X_train=X_train, y_train=y_train,
                           X_test=X_test, y_test=y_test,
                           pipeline=price_pipe)

    st.write("### Regression Performance Plots")
    st.write("* The regression performance plots below indicate that our model,\
         in most part, is able to predict sale prices well. The model looks less effective for houses with high prices though.")
    regression_evaluation_plots(
        X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, pipeline=price_pipe)
