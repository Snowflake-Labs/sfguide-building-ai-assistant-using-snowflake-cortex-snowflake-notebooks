# Import python packages
import streamlit as st
from snowflake.snowpark.context import get_active_session
from snowflake.cortex import Complete
import snowflake.snowpark.functions as F
from snowflake.core import Root

# Set Streamlit page configuration
st.set_page_config(layout="wide", initial_sidebar_state="expanded")
st.title(":telephone_receiver: :robot_face: Sales Calls: Q&A Assistant")
st.caption(
    f"""Welcome! This application suggests answers to product or sales GTM questions based 
    on sales calls captured in Gong or an equivalent platform like Zoom or Microsoft Teams.
    """
)

# Get the current credentials
session = get_active_session()
root = Root(session)

################################################################

# Constants
CHAT_MEMORY = 20
CHAT_SERVICE = "call_transcript_search_service"
DATABASE = "SALES_CALLS_DB"
SCHEMA = "SALES_CALLS_SCHEMA"

# Reset chat conversation
def reset_conversation():
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "What question do you need assistance answering?",
        }
    ]


##########################################
#      Select LLM
##########################################
with st.expander(":gear: Settings"):
    model = st.selectbox(
        "Change chatbot model:",
        [
            "mistral-large",
            "reka-flash",
            "llama2-70b-chat",
            "gemma-7b",
            "mixtral-8x7b",
            "mistral-7b",
            "snowflake-arctic",
            "llama3.1-8b"
        ],
    )
    st.button("Reset Chat", on_click=reset_conversation)


##########################################
#       Cortex Search
##########################################
def get_context(chat):
    chat_summary = summarize(chat)
    return find_similar_doc(chat_summary)


def summarize(chat):
    summary = Complete(
        model,
        "Provide the most recent question with essential context from this support chat: "
        + chat,
    )
    return summary.replace("'", "")

# Update to have multiple docs, return as dictionary
def find_similar_doc(query):
    cortex_search_service = (
        root.databases[DATABASE]
        .schemas[SCHEMA]
        .cortex_search_services[CHAT_SERVICE]
    )
    
    # Retrieve relevant document chunks
    context_documents = cortex_search_service.search(
        query, columns=["ID", "CHUNK"], limit=5  # Limiting to 5 relevant documents
    )
    
    results = context_documents.results
    if not results:
        st.warning("No relevant documents found.")
        return []  # Return an empty list if no results found
    
    # Return a list of dictionaries
    result_list = [{"ID": doc["ID"], "CHUNK": doc["CHUNK"]} for doc in results]
    st.info(f"Selected Sources: {[doc['ID'] for doc in result_list]}")
    return result_list


##########################################
#       Prompt Construction
##########################################
if "background_info" not in st.session_state:
    st.session_state.background_info = (
        session.table("CALLS_TRANSCRIPT")  # Replace with your actual sales data table
        .select("TRANSCRIPT")  # Adjust the column name to match your data schema
        .collect()[0][0]
    )


def get_prompt(chat, context, documents):
    document_references = "\n".join([f"Document ID: {doc['ID']}\n{doc['CHUNK']}" for doc in documents])
    prompt = f"""Answer this new sales-related query based on the latest call transcript.
        Use the provided context from relevant sales transcripts or previous customer interactions.
        Be concise and only answer the most recent query.
        The question is in the chat.
        Chat: <chat> {chat} </chat>.
        Context: <context> {context} </context>.
        Background Info: <background_info> {st.session_state.background_info} </background_info>.
        Documents used to answer: <documents> {document_references} </documents>."""
    return prompt.replace("'", "")

##########################################
#       Chat with LLM
##########################################
if "messages" not in st.session_state:
    reset_conversation()

if user_message := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": user_message})

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if st.session_state.messages[-1]["role"] != "assistant":
    chat = str(st.session_state.messages[-CHAT_MEMORY:]).replace("'", "")
    with st.chat_message("assistant"):
        with st.status("Answering..", expanded=True) as status:
            st.write("Finding relevant documents & support chat logs...")
            # Get relevant information from context and documents
            documents = find_similar_doc(chat)  # Returns a list of dictionaries
            if not documents:
                st.error("No documents found to provide an answer.")
            
            # Combine all relevant context
            context = "\n".join([doc['CHUNK'] for doc in documents])
            st.write("Using search results to answer your question...")
            
            # Ask LLM
            prompt = get_prompt(chat, context, documents)
            response = Complete(model, prompt)
            status.update(label="Complete!", state="complete", expanded=False)
        
        # Show the context and the response    
        st.markdown("### Answer:")
        st.markdown(response)
        
        st.markdown("**Context used to generate the answer:**")
        for document in documents:
            with st.expander(f"Transcript ID: {document['ID']}"):
                st.markdown(document["CHUNK"])

    st.session_state.messages.append({"role": "assistant", "content": response})
