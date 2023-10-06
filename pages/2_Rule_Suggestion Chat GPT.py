
import streamlit as st
import openai
import pandas as pd
import numpy as np

import requests as r
import json
#import pandas_profiling as pp
from ydata_profiling import ProfileReport

from requests.packages.urllib3.exceptions import InsecureRequestWarning

r.packages.urllib3.disable_warnings(InsecureRequestWarning)
r.packages.urllib3.disable_warnings()

import os

#os.environ['REQUESTS_CA_BUNDLE'] = 'sni_cloudflaressl_com.crt'


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
st.title("DQ Rules with Chat GPT")
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

df_InfoSchema = load_excel_data('transaction_cost_info_schema.xlsx')

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

st.write(data.head())
#data = data.convert_dtypes('string')
data_string = data.to_string()

#st.markdown('Data Preview')
#st.write(data.type())

###########################################

#st.stop()

bearer_token = "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"

message_params = {
    "model": "gpt-3.5-turbo",
    "messages": [
        { 'role': 'user', 'content': f'can you provide data quality rules for this dataset : \n{data_string}' }
    ],
    "max_tokens": 200,
    "temperature": 0.7,
}

message_head = {
    "Authorization": f"Bearer {bearer_token}"
}

#st.write(message_params)


st.header('Ask ChatGPT for Data Quality Rules')

    
if st.button('Ask For Rule Suggestions'):
    req = r.post('https://api.openai.com/v1/chat/completions', headers=message_head, json=message_params, verify=False)
    print(req.text)
    messages = json.loads(req.text)
    st.write(messages["choices"][0]["message"]["content"])
else: pass

#st.write(req, req.text)

##########################################################

st.header('Ask ChatGPT for Profile')
message_params = {
    "model": "gpt-3.5-turbo",
    "messages": [
        { 'role': 'user', 'content': f'can you provide a profiling report for this dataset : \n{data_string}' }
    ],
    "max_tokens": 200,
    "temperature": 0.7,
}

    
if st.button('Ask For Profile'):
    req = r.post('https://api.openai.com/v1/chat/completions', headers=message_head, json=message_params, verify=False)
    print(req.text)
    messages = json.loads(req.text)
    st.write(messages["choices"][0]["message"]["content"])
else: pass

#########################################################
st.header('Ask ChatGPT for DQ Rule SQL')
message_params = {
    "model": "gpt-3.5-turbo",
    "messages": [
        { 'role': 'user', 'content': f'can you provide SQL for data quality rules this dataset : \n{data_string}' }
    ],
    "max_tokens": 200,
    "temperature": 0.7,
}

    
if st.button('Ask For SQL'):
    req = r.post('https://api.openai.com/v1/chat/completions', headers=message_head, json=message_params, verify=False)
    print(req.text)
    messages = json.loads(req.text)
    st.write(messages["choices"][0]["message"]["content"])
else: pass

###########################################################
st.stop()
#########################################################
profile = ProfileReport(data,minimal=True)

#profile_2 = pp.ProfileReport(filtered_pr_df,minimal=True)

profile.to_file("Profile_Report_Test.html")
#html_profile_2 = profile_2.to_file('Profile_Report_2.html')

#path_to_html = "./Profile_Report.html"

#with open(path_to_html,'r') as f:
    #html_data = f.read()

st.header("Pandas Profiling Report")
st.write(profile.to_widgets())
#st.components.v1.html(html_data,height=10000)


########################################################

st.stop()

#####################################

#NULL Checks

st.subheader('NULL Check Suggestions')
st.write('Potential columns for NULL Value Check:')
df_NotNull=df_InfoSchema[df_InfoSchema['IS_NULLABLE'] == 'NO']
Col_NotNull=list(df_NotNull['COLUMN_NAME'].values)
cols_Unnecessary=list(data)
Cols_Unnecessaryremoved_forNotNULLs=Col_NotNull
statement1=''
for i in Cols_Unnecessaryremoved_forNotNULLs:
    statement1=statement1+str(i)+', '
if(statement1):
    Statement_Final = statement1+ ' are necessary columns which should not have NULL values as per Information Schema.'
    st.write(Statement_Final)


dataframe_InfoSchema_and_Stats=pd.merge(df_InfoSchema,data,left_on=['COLUMN_NAME'],right_on=['Columns'],how='outer')

#st.write(dataframe_InfoSchema_and_Stats.head())
#st.write(data.head())
#st.stop()

#df_CanbeNull=dataframe_InfoSchema_and_Stats[dataframe_InfoSchema_and_Stats['IS_NULLABLE'] == 'YES']

st.write(data.head())
st.write(dataframe_InfoSchema_and_Stats.head())
#st.write(df_CanbeNull.head())

#####################################################################

df_CanbeNull_per=df_CanbeNull[df_CanbeNull['Average_NullPercent']<=1]
Col_CanbeNull=list(df_CanbeNull_per['COLUMN_NAME'].values)
Cols_Unnecessaryremoved_forNULLs=set(Col_CanbeNull)

statement2=''
for i in Cols_Unnecessaryremoved_forNULLs:
    statement2=statement2+str(i)+', '
if(statement2):
    Statement_Final2=statement2 +' are necessary columns which can have NULL values as per Information Schema but their Average NULL Percent for the three months is less than 1%'
    st.write(Statement_Final2)

if((not statement1) and (not statement2)):
    st.write('There are no potential columns for NULL Value chesk in this dataset.')

data_null = data.isnull().sum()

st.stop()

####################################################################

def ask_gpt(prompt):
    req = r.post('https://api.openai.com/v1/chat/completions', headers=message_head, json=message_params, verify=False)
    print(req.text)
    messages = json.loads(req.text)
    return messages  

st.header('Ask ChatGPT to Profile this Dataset')
prompt_a = "Can you profile this {data}"

if st.button('Ask for Profile'):
    reply = ask_gpt(prompt_a)["choices"][0]["message"]["content"]
    message_params["messages"].append({'role': 'assistant', 'content': reply})
    message_params["messages"].append({'role': 'user', 'content': prompt_a})  

##################################################
st.stop()

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












