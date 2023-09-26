
import streamlit as st
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
       the LLaMA 2 API's implementation of a LLM.
       Pick a **table** from the **drop down** and **press enter** to receive 
       a **response** from the LLaMA 2
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

#####################################################


# Imports
import time
import copy
import pandas
from typing import List
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
import numpy as np
from pandas import DataFrame
import config
import os
import re
import tiktoken
import requests
import bisect


############################################################

MAX_RESPONSE_TOKENS = 200

class AdvisoryGPTPrompter:
    def __init__(self, bearer_token, model_name="gpt-35-turbo", system_message = None):
        self.model_name = model_name
        self.bearer_token = bearer_token
        if system_message:
            self.messages = [{"role": "system", "content": system_message}]
        else:
            self.messages = []
        # consider adding rate limiting - TODO let's do that as a subclass

    def __call__(self, prompt):
        prompt = prompt.strip()
        new_message = {"role": "user", "content": prompt} # TODO presumably we can set up a system message?
        url = f'https://digitalmatrix-cat.kpmgcloudops.com/workspace/api/v1/ai/deployment/{self.model_name}/chat/completion'
        headers = {
            'accept': 'application/json',
            'Authorization': f'Bearer {self.bearer_token}',
            'Content-Type': 'application/json'
        }
        data = {
            'engagementCode': 'NEEDED',
            'messages': self.messages + [new_message],
            "temperature": 0,
            "choiceCount": 1,
            "maxTokens": MAX_RESPONSE_TOKENS, # maximum number of tokens in the response
            "appendTrailingWhitespaceToStop": False
        }
        
        response = requests.post(url, headers=headers, json=data)
        response_json = response.json()
        try:
            response_str = response_json["choices"][0]['message']['content']
        except:
            raise Exception(str(response_json))
        new_response = {"role": "Assistant", "content": response_str}
        # self.messages += [new_message, new_response] # in our use case, we only need one at a time. We would need this line if we were having a back-and-forth conversation
        return response_str
    
class AdvisoryGPTEmbedder:
    def __init__(self, bearer_token, model_name="text-embedding-ada-002"):
        self.model_name = model_name
        self.bearer_token = bearer_token

    def __call__(self, prompt):
        prompt = prompt.strip()
        url = f'https://digitalmatrix-cat.kpmgcloudops.com/workspace/api/v1/ai/deployment/{self.model_name}/embedding'
        headers = {
            'accept': 'application/json',
            'Authorization': f'Bearer {self.bearer_token}',
            'Content-Type': 'application/json'
        }
        data = {
            'engagementCode': 'NEEDED',
            'input': prompt
        }
        
        response = requests.post(url, headers=headers, json=data)
        print(response)
        response_json = response.json()
        try:
            response_str = response_json["choices"][0]['message']['content']
        except:
            raise Exception(str(response_json))
        return response_str

class ChatGPTTriager:
    prompt_template = f"""'You are an operations agent triaging and assigning tickets. Based on the description and environment in the issue, and using Similar incidents, determine the Priority and Assignment group.
        Use JSON format like {{{{"Priority": "3 - Medium","Assignment group":"Spectrum Service Desk"}}}}
        Possible Assignment groups:
        {{assignment_groups}}
        Possible Priorities:
        {{priorities}}
        Similar incidents: {{context}}
        Issue: {{issue}}
        Priority and Assignment group:"""
    
    target_query_token_count = 7000
    max_tokens = {
        "gpt-4": 8192
    }

    def __init__(self, prompter, priorities, assignment_groups):
        self.prompter = prompter
        self.prompt_dict = {
            "priorities": priorities,
            "assignment_groups": assignment_groups,
        }

    def triage(self, issue, vector_store):
        starting_similar_incident_count = 40

        while True:
            docs_and_scores = vector_store.similarity_search_with_score(issue, k=starting_similar_incident_count)
            df = pandas.DataFrame(docs_and_scores, columns=['doc','score'])
            df['formatted'] = df.apply(lambda row: f"{row['doc'].page_content}\nPriority {row['doc'].metadata['priority']}\nAssignment group {row['doc'].metadata['assignment_group']}", axis=1)
            df['tokens'] = df.apply(lambda row: self.num_tokens_from_string(row['formatted'] + '\n\n'), axis=1) # add newlines to account for them being added in the join
            self.prompt_dict["context"] = '\n\n'.join("")
            self.prompt_dict["issue"] = issue
            prompt = copy.deepcopy(self.prompt_template).format(**self.prompt_dict)
            no_context_token_count = self.num_tokens_from_string(prompt)
            df['running_sum_of_tokens'] = df['tokens'].cumsum()
            target_incident_tokens = self.target_query_token_count - no_context_token_count - MAX_RESPONSE_TOKENS
            top_n_incidents = bisect.bisect_left(df['running_sum_of_tokens'].values, target_incident_tokens)
            self.prompt_dict['context'] = '\n\n'.join(df['formatted'].head(n=top_n_incidents))
            prompt = copy.deepcopy(self.prompt_template).format(**self.prompt_dict)

            if top_n_incidents == df.shape[0]: # if we picked the last element as the closest to the target value, we need to check to see if larger values are a better fit
                starting_similar_incident_count *= 2 # try again with more incidents
            else:
                print(f'using {top_n_incidents} incidents as context')
                break

        #sanity check
        if self.num_tokens_from_string(prompt) > self.max_tokens[self.prompter.model_name] - MAX_RESPONSE_TOKENS:
            raise Exception(f"Prompt has too many tokens:\n{prompt}")

        result = self.prompter(prompt)

        token_count = self.num_tokens_from_string(prompt + result)

        # TODO add result parsing

        return result, token_count


    def num_tokens_from_string(self, string: str, encoding_name: str = "text-embedding-ada-002") -> int:
        if not isinstance(string, str):
            return 0
        encoding = tiktoken.encoding_for_model(encoding_name)
        num_tokens = len(encoding.encode(string))
        return num_tokens


###################################################################################

# deal with rate limiting
RATE_LIMIT_TOKENS_PER_MINUTE = 120_000
SECONDS_PER_MINUTE = 60
TARGET_USAGE_FRACTION = .6
TARGET_RATE_LIMIT_TOKENS_PER_SECOND = (RATE_LIMIT_TOKENS_PER_MINUTE / SECONDS_PER_MINUTE) * TARGET_USAGE_FRACTION

def num_tokens_from_string(self, string: str, encoding_name: str = "text-embedding-ada-002") -> int:
    if not isinstance(string, str):
        return 0
    encoding = tiktoken.encoding_for_model(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

# Vector Store
FORCE_RELOAD_DATA = False
def get_or_create_vectorstore(df: DataFrame,file_name:str, columns_to_embed: List[str], embeddings = HuggingFaceEmbeddings()) -> FAISS:
    if os.path.exists(file_name) and not FORCE_RELOAD_DATA:
        print(f"Loading existing FAISS vector store with {type(embeddings)} embeddings.")
        db_ms = FAISS.load_local(file_name, embeddings)
    else:
        print("Creating new FAISS vector store")
        if type(embeddings) == HuggingFaceEmbeddings:
            print("Using HuggingFace embeddings")
            db_ms = FAISS.from_texts(texts=[re.sub('\s+', ' ', row.to_string()) for _,row in df[columns_to_embed].iterrows()], embedding=embeddings, metadatas=[{col: row[col] for col in df.columns if col not in columns_to_embed} for _, row in df.iterrows()])
        else:
            print("Non-HuggingFace embeddings; throttling to not hit API rate limits")
            previous_query_token_count = 0
            last_query_time = time.monotonic()
            db_ms = FAISS()
            text_embeddings = []
            for text in [re.sub('\s+', ' ', row.to_string()) for _,row in df[columns_to_embed].iterrows()]:
                seconds_to_wait = previous_query_token_count / TARGET_RATE_LIMIT_TOKENS_PER_SECOND
                time.sleep(max(0, seconds_to_wait - (time.monotonic() - last_query_time)))
                last_query_time = time.monotonic()
                # retreive embeddings
                text_embeddings.append(embed(text)) # TODO embed each text with preferred embeddings tool
                #db_ms.add_texts(texts=[text], embeddings=embeddings, metadatas=[metadata])
                previous_query_token_count = num_tokens_from_string(text)
            db_ms.add_embeddings(
                text_embeddings=text_embeddings,
                metadatas=[{col: row[col] for col in df.columns if col not in columns_to_embed} for _, row in df.iterrows()],
            )
        print(f"Saving to {file_name}")
        db_ms.save_local(file_name)
    return db_ms



def cross_validation(dfs: List[DataFrame], vector_stores: List[FAISS], triager: ChatGPTTriager, columns_to_embed: List[str]):
    results = []
    for i, df in enumerate(dfs, start=1):
        # if i <= 5: # experiment got cut off after iteration 5 due to token expiry
        #     continue
        # df, vs represent our test data
        context_provider = None
        for j, vs in enumerate(vector_stores, start=1):
            if j == i:
                continue # skip the 'test' vectorstore
            if context_provider is None:
                context_provider = copy.deepcopy(vs)
            else:
                context_provider.merge_from(vs)
        
        # take a sample of tickets from df that we will run our tests with
        # 100 * 10 = 1000 datapoints should be enough for our initial try
        SAMPLES_PER_CROSS_VALIDATION_STEP = 5
        ground_truth = df.sample(n=SAMPLES_PER_CROSS_VALIDATION_STEP) # we'll just do a sample for now for cost/rate limiting (time) reasons
        print(ground_truth.columns)
        test_issues = [re.sub('\s+', ' ', row.to_string()) for _,row in ground_truth[columns_to_embed].iterrows()]

        predictions=[]
        token_counts = []
        previous_query_token_count = 0
        last_query_time = time.monotonic()
        for issue in test_issues:
            seconds_to_wait = previous_query_token_count / TARGET_RATE_LIMIT_TOKENS_PER_SECOND
            print(f"Last prompt was {previous_query_token_count} tokens; waiting {seconds_to_wait} seconds.")
            time.sleep(max(0, seconds_to_wait - (time.monotonic() - last_query_time)))
            last_query_time = time.monotonic()
            # LLM output might be unreliable, so let's just keep the raw outputs to validate manually
            prediction, previous_query_token_count = triager.triage(issue, context_provider)
            predictions.append(prediction)
            token_counts.append(previous_query_token_count)

        ground_truth['predictions'] = predictions
        ground_truth['query_token_count'] = token_counts

        ground_truth.to_csv(f"experiment_part_{i}")
        ground_truth.to_parquet(f"experiment_part_{i}_pq")
        results.append(ground_truth)
    return results


##################################################################################################################

bearer_token = ""

embedder = AdvisoryGPTEmbedder(bearer_token=bearer_token)
embedder("test string")
# prompter = ChatGPTPrompter("gpt-4", bearer_token, 'Give the user a friendly greeting when they ask a question.')

# prompter("Can you please tell me about yourself?")

# results = cross_validation(dfs, vector_stores, columns_to_embed=columns_to_embed)


#######################################################################################################






