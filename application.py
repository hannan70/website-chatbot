import os
from dotenv import load_dotenv
load_dotenv()

# load env variable
groq_api_key = os.environ.get('GROQ_API_KEY') 
huggface_api_key = os.getenv("HUGGINGFACE_TOKEN") 



from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_chroma import Chroma
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
import streamlit as st

st.secrets['GROQ_API_KEY'] = groq_api_key
st.secrets['HUGGINGFACE_TOKEN'] = huggface_api_key


# setup llm
llm = ChatGroq(model='llama-3.3-70b-versatile', api_key=groq_api_key)

# setup embedding
embeddings = HuggingFaceEmbeddings(model="sentence-transformers/all-MiniLM-L6-v2")

# load vector database and retriver
vector_store = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
retriver = vector_store.as_retriever()

# setup prompt
prompt = ChatPromptTemplate.from_template(
    """
    You are a helpful and knowledgeable AI assistant for a directory website. 
    Answer the following user question using only the information provided in the context below.

    <context>
    {context}
    </context>

    Question: {input}

    If the answer is not in the context, respond with:
    "I’m sorry, I don’t have enough information to answer that."

    Answer in a clear and concise way.
    """
)

qa_chain = create_stuff_documents_chain(llm=llm, prompt=prompt)
rag_chain = create_retrieval_chain(retriver, qa_chain)

st.caption("Ask question about (https://citylocalbiz.us/) this website")

# handle session state
if "messages" not in st.session_state:
    st.session_state['messages'] = [
        {
            "role": "assistant",
            "content": "Hi there! I’m your virtual assistant. Feel free to ask me anything about this website — I’m here to help!."
        }
    ]

for msg in st.session_state.messages:
    st.chat_message(msg['role']).write(msg['content'])

# final response
def generate_response(question):
    response = rag_chain.invoke({"input": question})
    return response

question = st.chat_input("Ask me anything about this website")

if question is not None:
    if question.strip():
        # append user message before processing
        st.session_state.messages.append({"role": "user", "content": question})
        st.chat_message("user").write(question)

        with st.spinner("Analyzing Response..."):
            
            final_response = generate_response(question)

            st.session_state.messages.append({"role": "assistant", "content": final_response['answer']})
            # st.chat_message("assistant").write(final_response['answer'])
            st.success(final_response['answer'])

    else:
        st.error("Please provide a valid question")


