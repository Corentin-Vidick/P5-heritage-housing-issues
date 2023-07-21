import streamlit as st
from app_pages.multipage import MultiPage

# load pages scripts
from app_pages.page_1 import page_1
from app_pages.page_2 import page_2

app = MultiPage(app_name="Churnometer")  # Create an instance of the app

# Add your app pages here using .add_page()
app.add_page("Page 1", page_1)
app.add_page("Page 2", page_2)

app.run()  # Run the  app
