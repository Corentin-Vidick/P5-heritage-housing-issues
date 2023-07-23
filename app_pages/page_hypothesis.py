import streamlit as st


def page_hypothesis():
    """
    Presents our three hypothesises and their confirmation
    """
    # conclusions taken from "03 - Sales Price Study" notebook
    st.write("### Project Hypotheses and Validation")
    st.success(
        f"**Hypothesis one:**\t"
        f"The size of the house is positively correlated to the sale price.\
            We can see this through the features 1stFlrSF, GarageArea,\
            GrLivArea, LotFrontage and TotalBsmtSF.\n\n"
        f"**Hypothesis two:**\t"
        f"The quality/condition of the property is positively correlated\
            to the sale price. Features such as BsmtFinType1, GarageFinish,\
            KitchenQual or OverallQual confirm this.\n\n"
        f"**Hypothesis three:**\t"
        f"The age of a house is negatively correlated to the sale price.\
            GarageYrBlt, YearBuilt and YearRemodAdd confirm this hypothesis."
    )
