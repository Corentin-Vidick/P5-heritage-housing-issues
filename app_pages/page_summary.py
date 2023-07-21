import streamlit as st


def page_summary():
    """
    Displays contents of the project summary page
    """
    st.write("### Quick Project Summary")

    st.info(
        f"**Project Terms & Jargons**\n\n"
        f"* **Sales price** of a house refers to the current market price, in US dollars,\
         of a house with with various attributes.\n"
        f"* **Inherited house** is a house that the client inherited from grandparents.\n"
        f"* **Summed price** is the total of the predicted sales prices of the four inherited houses.\n\n"
    )

    # text based on README file - "Dataset Content" section
    st.info(
        f"**Project Dataset**\n"
        f"*- This project uses a dataset sourced from [Kaggle](https: // www.kaggle.com/codeinstitute/housing-prices-data).\
        - The first part of the dataset, house_price_records, has 1460 rows and 24 columns. 23 of them represent the house\
        profile from properties in Ames, Iowa, built between 1872 and 2010 (i.e: Floor Area, Basement, Garage, Kitchen, Lot,\
        Porch, Wood Deck, Year Built). The last column represents the sale price for the property, this will be our target\
        variable."
    )

    # Link to README file, so the users can have access to full project documentation
    st.write(
        f"* For additional information, please visit and read the "
        f"[Project README file](https://github.com/Corentin-Vidick/P5-heritage-housing-issues/blob/main/README.md).")

    # copied from README file - "Business Requirements" section
    st.success(
        f"**Project Business Requirements**\n\n"

        f"The project has 2 business requirements:\n"
        f"* **- 1 -** The client is interested in discovering how the house attributes correlate with the sale price."
        f" Therefore, the client expects data visualisations of the correlated variables against the sale price to show that.\n\n"
        f"* **- 2 -** The client is interested in predicting the house sale price from her four inherited houses and any other\
            house in Ames, Iowa."
    )
