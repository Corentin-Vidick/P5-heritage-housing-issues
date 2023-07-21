import streamlit as st
from app_pages.multipage import MultiPage

# load pages scripts
from app_pages.page_summary import page_summary
from app_pages.page_hypothesis import page_hypothesis
from app_pages.page_sales_price_study import page_sales_price_study
from app_pages.page_inheritance import page_inheritance
from app_pages.page_technical_info import page_technical_info

# Create an instance of the app
app = MultiPage(app_name="Heritage Housing Issues")

# Add your app pages here using .add_page()
app.add_page("Project summary", page_summary)
app.add_page("Sales price study", page_sales_price_study)
app.add_page("Project Hypothesises", page_hypothesis)
app.add_page("Inheritance sales price", page_inheritance)
app.add_page("Technical information", page_technical_info)

app.run()  # Run the  app
