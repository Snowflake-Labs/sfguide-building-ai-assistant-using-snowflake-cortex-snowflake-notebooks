import streamlit as st
from snowflake.snowpark.context import get_active_session
from snowflake.cortex import Complete
import re

st.set_page_config(layout="wide", initial_sidebar_state="expanded")
st.title(":telephone_receiver: :robot_face: LLM Driven Analytics for Sales Calls")

session = get_active_session()

################################################################

df = session.table("CALLS_TRANSCRIPT").to_pandas()
column_specifications = [col_name for col_name in df.columns]

st.subheader('Call Transcripts:')
st.dataframe(df.head())
    
prompt = st.text_area('What do you want to visualize?')

def extract_python_code(text):
    pattern = r"```python(.*?)```"
    match = re.search(pattern, text, re.DOTALL)
    
    if match:
        return match.group(1).strip()
    else:
        return "No Python code found in the input string."

if st.button('Visualize'):
    prompt = f'You are a python developer that writes code using altair and streamlit to visualize data. \
    Your data input is a pandas dataframe that you can access with df. \
    The pandas dataframe has the following columns: {column_specifications}.\
    {prompt}\
    If you are asked to return a list, create a dataframe and use st.dataframe() to display the dataframe.'
    with st.spinner("Waiting for Results"):
        code = Complete('llama3.1-8b', prompt.replace("'", "''"))
    execution_code = extract_python_code(code)
    col1, col2 = st.columns(2)
    with col1:
        st.subheader('This is the executed code:')
        st.code(execution_code, language="python", line_numbers=False)
    with col2:
        with st.spinner("Plotting ..."):
            exec(execution_code)
