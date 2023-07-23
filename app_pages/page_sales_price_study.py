import ppscore as pps
import seaborn as sns
from feature_engine.encoding import OneHotEncoder
import plotly.express as px
import numpy as np
from feature_engine.discretisation import ArbitraryDiscretiser
import streamlit as st
from src.data_management import load_housing_data
import matplotlib.pyplot as plt
sns.set_style("whitegrid")


def page_sales_price_study():
    """
    Display correlated features and a checkbox to show the show
    house price per variable.
    """
    df_prices = load_housing_data()

    vars_to_study = ['1stFlrSF',
                     'BsmtFinType1',
                     'GarageArea',
                     'GarageFinish',
                     'GarageYrBlt',
                     'GrLivArea',
                     'KitchenQual',
                     'LotFrontage',
                     'OverallQual',
                     'TotalBsmtSF',
                     'YearBuilt',
                     'YearRemodAdd']

    st.write("### Housing Prices Correlation Study (Business Requirement 1)")
    st.info(
        f"* **- 1 -** - The client is interested in discovering how the house\
        attributes correlate with the sale price. Therefore, the client\
        expects data visualisations of the correlated variables against\
        the sale price to show that."
    )

    # inspect data
    if st.checkbox("Inspect Housing Data"):
        st.write(
            f"* The dataset has\
            {df_prices.shape[0]} rows and {df_prices.shape[1]} columns.\
            We show the first 10 rows below.\n"
            f"* SalePrice is our target variable, and we want to identify\
                features correlated to it."
        )
        st.write(df_prices.head(10))
    st.write("---")

    # Correlation Study Summary
    st.write(
        f"* We conducted a correlation study in the notebook to better\
        understand how the variables are correlated to sale price of a\
        property. This addresses the first business requirement (BR1)\
        of the project. \n"
        f"* We found that the most correlated variable are:\
        \n **{vars_to_study}**"
    )

    # Text based on "03 - Sales Price Study" notebook
    # - "Conclusions and Next steps" section
    st.info(
        f"We make the following observations from both the correlation\
        analysis and the plots (particularly the heatmaps below).\n"
        f"* Higher values of 1stFlrSF, GarageArea, GrLivArea, LotFrontage and\
        TotalBsmtSF are associated with higher sale price.\
        The bigger the property the higher the price\n"
        f"* More recent properties, or properties that have been recently\
        modified (GarageYrBlt, YearBuilt and YearRemodAdd)\
        have typically higher prices.\n"
        f"* Features that represent the quality of a property\
        (BsmtFinType1, GarageFinish, KitchenQual or OverallQual) are\
        also positively correlated to sale price of a house.\n\n"
        f"While the plots corroborate these observations, we should also\
        notice from the plots of sale price against\
        the correlated features, that the relationships become less clear\
        at higher values of the variables.\n"
        f"* When the size of 1stFlrSF is around 2500, for example,\
        sale price can have both low and high values.\n"
        f"* We see similar pattern in the regression plot of sale price and\
        GarageArea when the it's value is around 800.\n"
    )

    df_eda = df_prices.filter(vars_to_study + ['SalePrice'])
    target_var = 'SalePrice'
    st.write("#### Data visualizations")
    # Distribution of target variable
    if st.checkbox("Distribution of target variable"):
        plot_target_hist(df_eda, target_var)

    # Individual plots per variable
    if st.checkbox("House prices per feature"):
        house_price_per_variable(df_eda)

    if st.checkbox("Heatmaps: Pearson, Spearman and PPS Correlations"):
        # Used to avoid deprecation warning message
        st.set_option('deprecation.showPyplotGlobalUse', False)
        df_corr_pearson, df_corr_spearman = calculate_correlation(df_eda)
        st.pyplot(display_correlation_pearson(
            df_corr_pearson=df_corr_pearson,
            CorrThreshold=0.45,
            figsize=(20, 12), annot_size=10))
        st.pyplot(display_correlation_spearman(
            df_corr_spearman=df_corr_spearman,
            CorrThreshold=0.45,
            figsize=(20, 12), annot_size=10))
        pps_matrix = CalculatePPS(df_eda)
        st.pyplot(DisplayPPS(pps_matrix=pps_matrix, PPS_Threshold=0.06,
                             figsize=(12, 10), font_annot=10))


def house_price_per_variable(df_eda):
    """
    Generate box plot, line plot or scatter plot of SalePrice and
    the house features
    """
    vars_to_study = ['1stFlrSF',
                     'BsmtFinType1',
                     'GarageArea',
                     'GarageFinish',
                     'GarageYrBlt',
                     'GrLivArea',
                     'KitchenQual',
                     'LotFrontage',
                     'OverallQual',
                     'TotalBsmtSF',
                     'YearBuilt',
                     'YearRemodAdd']
    time = ['YearBuilt', 'YearRemodAdd']
    target_var = 'SalePrice'

    for col in vars_to_study:
        if len(df_eda[col].unique()) <= 10:
            plot_box(df_eda, col, target_var)
            print("\n\n")
        else:
            if col in time:
                plot_line(df_eda, col, target_var)
                print("\n\n")
            else:
                plot_reg(df_eda, col, target_var)
                print("\n\n")


def plot_target_hist(df, target_var):
    """
    Function to plot a histogram of the target variable
    """
    fig, axes = plt.subplots(figsize=(12, 6))
    sns.histplot(data=df, x=target_var, kde=True)
    plt.title(f"Distribution of {target_var}", fontsize=20)
    st.pyplot(fig)


def plot_reg(df, col, target_var):
    """
    Generate scatter plot
    """
    fig, axes = plt.subplots(figsize=(12, 6))
    sns.regplot(data=df, x=col, y=target_var, ci=None)
    plt.title(f"Regression plot of {target_var} against {col}", fontsize=20)
    st.pyplot(fig)  # st.pyplot() renders image, in notebook is plt.show()


def plot_line(df, col, target_var):
    """
    Generate line plot
    """
    fig, axes = plt.subplots(figsize=(12, 6))
    sns.lineplot(data=df, x=col, y=target_var)
    plt.title(f"Line plot of {target_var} against {col}", fontsize=20)
    st.pyplot(fig)  # st.pyplot() renders image, in notebook is plt.show()


def plot_box(df, col, target_var):
    """
    Generate box plot
    """
    fig, axes = plt.subplots(figsize=(12, 6))
    sns.boxplot(data=df, x=col, y=target_var)
    plt.title(f"Box plot of {target_var} against {col}", fontsize=20)
    st.pyplot(fig)  # st.pyplot() renders image, in notebook is plt.show()


# Heatmaps
def heatmap_correlation(df, threshold, title, figsize=(20, 12), annot_size=8):
    """
    Function to create heatmap using Pearson's and Spearman's correlations.
    """
    if len(df.columns) > 1:
        # we create the mask for the upper diagonal and
        # show only values greater than the threshold
        mask = np.zeros_like(df, dtype=np.bool)
        mask[np.triu_indices_from(mask)] = True
        mask[abs(df) < threshold] = True

        # we plot the heatmap
        fig, axes = plt.subplots(figsize=figsize)
        sns.heatmap(data=df, annot=True, xticklabels=True, yticklabels=True,
                    mask=mask, cmap='viridis', annot_kws={"size": annot_size},
                    ax=axes, linewidth=0.5).set(title=title)
        plt.ylim(len(df.columns), 0)


def calculate_correlation(df):
    """
    Function to calculate correlations.
    """
    df_corr_spearman = df.corr(method="spearman")
    df_corr_spearman.name = 'corr_spearman'
    df_corr_pearson = df.corr(method="pearson")
    df_corr_pearson.name = 'corr_pearson'

    return df_corr_pearson, df_corr_spearman


def display_correlation_spearman(df_corr_spearman, CorrThreshold,
                                 figsize=(20, 12), annot_size=8):
    """
    Function to display the correlations.
    """
    st.write(f"\n*** Heatmap: Spearman Correlation ***\n\
            Evaluates the monotonic relationship\n")
    heatmap_correlation(df=df_corr_spearman, threshold=CorrThreshold,
                        figsize=figsize, annot_size=annot_size, title="")


def display_correlation_pearson(df_corr_pearson, CorrThreshold,
                                figsize=(20, 12), annot_size=8):
    """
    Function to display the correlations.
    """
    st.write(f"\n*** Heatmap: Pearson Correlation ***\n\
            Evaluates the linear relationship\n")
    heatmap_correlation(df=df_corr_pearson, threshold=CorrThreshold,
                        figsize=figsize, annot_size=annot_size, title="")


def heatmap_pps(df, threshold, figsize=(20, 12), font_annot=8):
    """
    Function to create heatmap using pps.
    """
    if len(df.columns) > 1:

        mask = np.zeros_like(df, dtype=np.bool)
        mask[abs(df) < threshold] = True

        fig, ax = plt.subplots(figsize=figsize)
        ax = sns.heatmap(df, annot=True, xticklabels=True,
                         yticklabels=True,
                         mask=mask, cmap='rocket_r',
                         annot_kws={"size": font_annot},
                         linewidth=0.05, linecolor='grey')

        plt.ylim(len(df.columns), 0)
        plt.show()


def CalculatePPS(df):
    """
    Function to calculate pps.
    """
    pps_matrix_raw = pps.matrix(df)
    pps_matrix = pps_matrix_raw.filter(['x', 'y', 'ppscore']).pivot(
        columns='x', index='y', values='ppscore')

    pps_score_stats = pps_matrix_raw.query(
        "ppscore < 1").filter(['ppscore']).describe().T
    print("PPS threshold - check PPS score IQR to decide\
          threshold for heatmap \n")
    print(pps_score_stats.round(3))

    return pps_matrix


def DisplayPPS(pps_matrix, PPS_Threshold,
               figsize=(20, 12), font_annot=8):
    """
    Function to display the pps.
    """
    st.write(f"\n*** Heatmap: Power Predictive Score(PPS) ***\n\
            PPS detects linear or non-linear relationships between\
            two columns.\n The score ranges from 0 (no predictive power)\
            to 1 (perfect predictive power) \n")
    heatmap_pps(df=pps_matrix, threshold=PPS_Threshold,
                figsize=figsize, font_annot=font_annot)
