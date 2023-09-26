# https://medium.com/@murtuza753/using-llama-2-0-faiss-and-langchain-for-question-answering-on-your-own-data-682241488476

import streamlit as st
#import openai
import pandas as pd
import numpy as np

from torch import cuda, bfloat16
import transformers

import bitsandbytes
#https://stackoverflow.com/questions/76924239/accelerate-and-bitsandbytes-is-needed-to-install-but-i-did

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

############################################


model_id = 'meta-llama/Llama-2-7b-chat-hf'

device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'

# set quantization configuration to load large model with less GPU memory
# this requires the `bitsandbytes` library
bnb_config = transformers.BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=bfloat16,
    llm_int8_enable_fp32_cpu_offload=True
)

# begin initializing HF items, you need an access token
hf_auth = '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'
model_config = transformers.AutoConfig.from_pretrained(
    model_id,
    use_auth_token=hf_auth
)

model = transformers.AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True,
    config=model_config,
    quantization_config=bnb_config,
    device_map='auto',
    use_auth_token=hf_auth
)

# enable evaluation mode to allow model inference
model.eval()

st.markdown(f"Model loaded on {device}")


####################################################################

st.stop()

######################################################################


tokenizer = transformers.AutoTokenizer.from_pretrained(
    model_id,
    use_auth_token=hf_auth
)

######################################################################

stop_list = ['\nHuman:', '\n```\n']

stop_token_ids = [tokenizer(x)['input_ids'] for x in stop_list]
stop_token_ids

######################################################################

import torch

stop_token_ids = [torch.LongTensor(x).to(device) for x in stop_token_ids]
stop_token_ids

#######################################################################

from transformers import StoppingCriteria, StoppingCriteriaList

# define custom stopping criteria object
class StopOnTokens(StoppingCriteria):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        for stop_ids in stop_token_ids:
            if torch.eq(input_ids[0][-len(stop_ids):], stop_ids).all():
                return True
        return False

stopping_criteria = StoppingCriteriaList([StopOnTokens()])

########################################################################


generate_text = transformers.pipeline(
    model=model, 
    tokenizer=tokenizer,
    return_full_text=True,  # langchain expects the full text
    task='text-generation',
    # we pass model parameters here too
    stopping_criteria=stopping_criteria,  # without this model rambles during chat
    temperature=0.1,  # 'randomness' of outputs, 0.0 is the min and 1.0 the max
    max_new_tokens=512,  # max number of tokens to generate in the output
    repetition_penalty=1.1  # without this output begins repeating
)

###############################################################################

res = generate_text("Explain me the difference between Data Lakehouse and Data Warehouse.")
st.markdown(res[0]["generated_text"])


################################################################################

st.stop()

##########################################################

from langchain.llms import HuggingFacePipeline

llm = HuggingFacePipeline(pipeline=generate_text)

# checking again that everything is working fine
llm(prompt="Explain me the difference between Data Lakehouse and Data Warehouse.")

###############################################################

from langchain.document_loaders import WebBaseLoader

web_links = ["https://www.databricks.com/","https://help.databricks.com","https://databricks.com/try-databricks","https://help.databricks.com/s/","https://docs.databricks.com","https://kb.databricks.com/","http://docs.databricks.com/getting-started/index.html","http://docs.databricks.com/introduction/index.html","http://docs.databricks.com/getting-started/tutorials/index.html","http://docs.databricks.com/release-notes/index.html","http://docs.databricks.com/ingestion/index.html","http://docs.databricks.com/exploratory-data-analysis/index.html","http://docs.databricks.com/data-preparation/index.html","http://docs.databricks.com/data-sharing/index.html","http://docs.databricks.com/marketplace/index.html","http://docs.databricks.com/workspace-index.html","http://docs.databricks.com/machine-learning/index.html","http://docs.databricks.com/sql/index.html","http://docs.databricks.com/delta/index.html","http://docs.databricks.com/dev-tools/index.html","http://docs.databricks.com/integrations/index.html","http://docs.databricks.com/administration-guide/index.html","http://docs.databricks.com/security/index.html","http://docs.databricks.com/data-governance/index.html","http://docs.databricks.com/lakehouse-architecture/index.html","http://docs.databricks.com/reference/api.html","http://docs.databricks.com/resources/index.html","http://docs.databricks.com/whats-coming.html","http://docs.databricks.com/archive/index.html","http://docs.databricks.com/lakehouse/index.html","http://docs.databricks.com/getting-started/quick-start.html","http://docs.databricks.com/getting-started/etl-quick-start.html","http://docs.databricks.com/getting-started/lakehouse-e2e.html","http://docs.databricks.com/getting-started/free-training.html","http://docs.databricks.com/sql/language-manual/index.html","http://docs.databricks.com/error-messages/index.html","http://www.apache.org/","https://databricks.com/privacy-policy","https://databricks.com/terms-of-use"] 

loader = WebBaseLoader(web_links)
documents = loader.load()

##################################################################

from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
all_splits = text_splitter.split_documents(documents)

####################################################################

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

model_name = "sentence-transformers/all-mpnet-base-v2"
model_kwargs = {"device": "cuda"}

embeddings = HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs)

# storing embeddings in the vector store
vectorstore = FAISS.from_documents(all_splits, embeddings)

#######################################################################

from langchain.chains import ConversationalRetrievalChain

chain = ConversationalRetrievalChain.from_llm(llm, vectorstore.as_retriever(), return_source_documents=True)

##########################################################################

chat_history = []

query = "What is Data lakehouse architecture in Databricks?"
result = chain({"question": query, "chat_history": chat_history})

print(result['answer'])

#############################################################################

chat_history = [(query, result["answer"])]

query = "What are Data Governance and Interoperability in it?"
result = chain({"question": query, "chat_history": chat_history})

st.markdown(result['answer'])

#############################################################################

st.markdown(result['source_documents'])

##########################################################################










