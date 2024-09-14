import streamlit as st 
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
from langchain.llms import HuggingFaceHub
from langchain_community.llms import Ollama
from langchain_groq import ChatGroq
import os 




#extraction of the text from the pdfs
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    
    return text

#dividing the raw text in different chunks
def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator= "\n" ,
        chunk_size=1000,
        chunk_overlap=200,
        length_function= len
        )
    
    chunks = text_splitter.split_text(text)
    return chunks 

#creating a vector store embeddings from huggingface 
def get_vectorstore(text_chunks):
   # embeddings = OpenAIEmbeddings()
    embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

#creating a conversation chain to store the context for follow up question
def get_conversation_chain(vectorstore, groq_api_key):

    #llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})
    #llm = Ollama(model="llama2")
    llm=ChatGroq(groq_api_key=groq_api_key,
             model_name="llama3-70b-8192")
    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain


#handling the user input 
def handle_userinput(user_question):
    response = st.session_state.conversation({'question' : user_question})
    #st.write(response)
    st.session_state.chat_history = response['chat_history']

    for i , message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html= True)
        else:
            st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html= True)

def main():
    load_dotenv()
    #os.environ['OPENAI_API_KEY']=os.getenv("OPENAI_API_KEY")
    groq_api_key=os.getenv('GROQ_API_KEY')

    st.set_page_config("Chat with your pdf!!!!", page_icon=":books:")

    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Chat with your pdf!!! :books:")

    #question section
    user_question = st.text_input("Wanna ask something???")

    if user_question:
        handle_userinput(user_question)



    with st.sidebar:
        st.subheader("Your documents")
        #generally supports single file at a time. Need the enable the option to access multiple files
        pdf_docs = st.file_uploader("Upload your pdf file", type=["pdf"], accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                #get the pdf text 
                raw_text = get_pdf_text(pdf_docs)

                #get the text chunks 
                text_chunks = get_text_chunks(raw_text)

                #create the vector store with embeddings 
                vectorstore = get_vectorstore(text_chunks)
                
                #create the conversation chain
                st.session_state.conversation = get_conversation_chain(vectorstore, groq_api_key)


if __name__ == '__main__':
    main()