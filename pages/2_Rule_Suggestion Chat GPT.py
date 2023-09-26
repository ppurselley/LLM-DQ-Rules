
import streamlit as st
import openai
import pandas as pd
import numpy as np




##################################################

st.set_page_config(page_title="LLM Rule Suggestion",page_icon='kpmg_logo.jpg')
hide_menu_style ="""
    <style>
    MainMenu {visibility: show;}
    footer {visibility: hidden;}
    </style>
    """
st.markdown(hide_menu_style,unsafe_allow_html=False)
st.columns(3)[2].image('kpmg_logo.jpg')
st.title("DQ Rules with LLaMA 2")
st.sidebar.header("Instructions")
st.sidebar.info(
    '''This is a web application that allows you to interact with 
       the Chat GPT API's implementation of a LLM.
       Pick a **table** from the **drop down** and **press enter** to receive 
       a **response** from the Chat GPT
       '''
    )


#####################################################

@st.cache
def load_excel_data(df):
    data = pd.read_excel(df)
    data = data.convert_dtypes()
    return data

###########################################

data = pd.DataFrame()

st.subheader('1 - Select the metadata to build rules on top of')

try:
    ref_table = st.selectbox(
        "Select Data",['Choose Your Data From The Dropdown','Transaction_Cost_3_months.xlsx']
    )

    data_load_status = st.text('Loading data ...')
    data = load_excel_data(ref_table)
    data_load_status.text('')
except:
    pass

st.markdown('Data Preview')
st.write(data.head())

###########################################

#st.stop()


##################################################


#############################################3######

openai.api_key = "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"

# Create an input box for the user to type their message

user_input = st.text_input("Enter your message here:")

# Use the OpenAI API to generate a response to the user's message

chat_response = openai.Completion.create(
    engine="gpt-4",
    prompt=user_input,
    max_tokens=1024,
    temperature=0.5,
)

# Create an output box to display the ChatGPT's response

st.write(chat_response.choices[0].text)

######################################################


model_engine = "text-davinci-003"
openai.api_key = "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"

def main():
    '''
    This function gets the user input, pass it to ChatGPT function and 
    displays the response
    '''
    # Get user input
    user_query = st.text_input("Enter query here, to exit enter :q", "what is Python?")
    if user_query != ":q" or user_query != "":
        # Pass the query to the ChatGPT function
        response = ChatGPT(user_query)
        return st.write(f"{user_query} {response}")

def ChatGPT(user_query):
    ''' 
    This function uses the OpenAI API to generate a response to the given 
    user_query using the ChatGPT model
    '''
    # Use the OpenAI API to generate a response
    completion = openai.Completion.create(
                                  engine = model_engine,
                                  prompt = user_query,
                                  max_tokens = 1024,
                                  n = 1,
                                  temperature = 0.5,
                                      )
    response = completion.choices[0].text
    return response

main()

##################################################












