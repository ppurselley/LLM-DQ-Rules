#### import libraries

import streamlit as st

#run to configure the home page
def run():
    st.set_page_config(page_title="Home",page_icon="kpmg_logo.jpg",layout="wide")
    hide_menu_style = """
        <style>
        #MainMenu {visibility: show; }
        footer {visibility: hidden;}
        #root > div:nth-child(1) > div > div > div > div > section > div {padding-top: 0rem;}
        </style>
        """
    st.markdown(hide_menu_style,unsafe_allow_html=True)

    #with st.columns(3)[1]:
        #st.image('kpmg_logo_large.jpg')
        #st.header('Intuitive DQ Features')
    st.image('kpmg_logo_large.jpg')
    st.header('KPMG Advisory Chat DQ Rules')
    
    if st.button('Click to insert credentials'):
        st.write('Feature coming soon...')

    else:
        pass

    st.subheader('Requirements for this application')
    st.markdown('1) Connection to Advisory GPT')
    st.markdown('2) tbd')

if __name__ == "__main__":
    run()