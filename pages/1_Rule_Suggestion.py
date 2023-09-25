import streamlit as st
import openai
import pandas as pd
import numpy as np

##################################################

st.set_page_config(page_title="Rule Suggestion",page_icon='kpmg_logo.jpg')
hide_menu_style ="""
        <style>
        #MainMenu {visibility: show;}
        footer {visibility: hidden;}
        </style>
        """
st.markdown(hide_menu_style,unsafe_allow_html=False)
st.columns(3)[2].image('kpmg_logo.jpg')
st.title("Chatting with Advisory ChatGPT")
st.sidebar.header("Instructions")
st.sidebar.info(
    '''This is a web application that allows you to interact with 
       the OpenAI API's implementation of the ChatGPT model.
       Enter a **query** in the **text box** and **press enter** to receive 
       a **response** from the ChatGPT
       '''
    )


#####################################################

###########################################

data = pd.DataFrame()

st.subheader('1 - Select the metadata to build and deploy model')

try:
    ref_table = st.selectbox(
        "Select metadata",['Transaction_Cost_3_months.xlsx']
    )

    data_load_status = st.text('Loading data ...')
    #data = load_excel_data(ref_table)
    data_load_status.text('')
except:
    pass

st.markdown('Metadata Preview')
st.write(data.head())

###########################################