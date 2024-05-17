import streamlit as st
from langchain_community.llms import HuggingFaceHub
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from initializations import *
from streamlit_chat import message
from utils import PrepareDocsAndEmbeddings, Query


def build_app():
    st.set_page_config(page_icon="🎓")
    st.header("ScholarSAGE")
    st.subheader("Your personal research assistant")

    if 'responses' not in st.session_state:
        st.session_state['responses'] = ["Hi, How can I assist you?"]

    if 'requests' not in st.session_state:
        st.session_state['requests'] = []

    vector_db = PrepareDocsAndEmbeddings().get_embeddings(prepare_embeddings=False)
    response = Query(vector_db=vector_db)

    llm = HuggingFaceHub(
        huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN,
        repo_id="google/flan-t5-large",
        model_kwargs={
            "max_length": 4000,
            "temperature": 0.2,
            "repetition_penalty": 1.03,
        },
    )

    if 'buffer_memory' not in st.session_state:
        st.session_state.buffer_memory = ConversationBufferWindowMemory(k=3, return_messages=True, memory_key="chat_history")

    # container for chat history
    response_container = st.container()
    # container for text box
    text_container = st.container()

    with text_container:
        query = st.text_input("Query: ", key="input")
        if query:
            with st.spinner("Typing..."):
                conversation_string = response.get_conversation_string()
                query_answer = response.get_query_answer(llm=llm, query=query)
            st.session_state.requests.append(query)
            st.session_state.responses.append(query_answer)

    with response_container:
        if st.session_state['responses']:
            for i in range(len(st.session_state['responses'])):
                message(st.session_state['responses'][i], key=str(i))
                if i < len(st.session_state['requests']):
                    message(st.session_state["requests"][i], is_user=True, key=str(i) + '_user')


if __name__ == "__main__":
    build_app()
