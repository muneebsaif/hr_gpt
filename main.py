import os
import numpy as np
from constants import mycey
from langchain.llms import OpenAI
from langchain import PromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.chains import SequentialChain
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import PyMuPDFLoader, UnstructuredPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.utilities import SQLDatabase
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import faiss

class basic:
    def __init__(self,dim=1536):
        os.environ["OPENAI_API_KEY"]=mycey
        self.llm=OpenAI(temperature=0.8,verbose=False)
        self.dim=dim
        self.index = faiss.IndexFlatL2(dim)
        self.embeddings = OpenAIEmbeddings()
        self.chunks_embedding =''
        self.chunks=[]
        self.pickle_path="openai_embedding.pkl"
        self.db = SQLDatabase.from_uri("sqlite:///DB/demo.db")
        self.query=""


    def load_document(self,path_to_document='./Documents/HR_document.pdf'):
        loader = PyMuPDFLoader(path_to_document)
        # Load the document
        documents = loader.load()
        #text splitter 
        r_splitter=RecursiveCharacterTextSplitter(
                        separators=["\n\n","\n"," "],
            chunk_size=300,
            chunk_overlap=50
        )
        self.chunks=r_splitter.split_text(documents[0].page_content)

    def store_embeddings(self,):
        import pickle
        with open( self.pickle_path,"wb") as f:
            pickle.dump(self.chunks_embedding, f)

    def read_embeddings(self):
        import pickle
        # Open the pickle file in read-binary mode
        with open(self.pickle_path, "rb") as f:
            self.chunks_embedding = pickle.load(f)

    def get_embeddings(self,chunks="",query=False):
        if not query:
            chunks_embedding=self.embeddings.embed_documents(self.chunks)
            self.chunks_embedding =np.array(chunks_embedding)
            self.dim=len(self.chunks_embedding [1])
        else:
            print("i am here",chunks)
            query_v = self.embeddings.embed_query(chunks)
            query_v=np.array(np.array(query_v) ).reshape(1,-1) #reshaping
            return query_v 


    def vector_db_insert(self,):
        self.index.add(self.chunks_embedding)

    def set_query(self,query):
        self.query=query

    def vector_db_find(self):
        svec = self.get_embeddings(chunks=self.query,query=True)
        distances, I = self.index.search(svec, k=1)
        row_indices = I.tolist()[0]
        return self.chunks[row_indices[0]]

    def get_schema(self):
        schema = self.db.get_table_info()
        return schema
    
    def sql_query(self):
        template = """
        Based on the table schema below, write a SQL query that would answer the user's question:
        {schema}
        Question: {question}
        SQL Query:
        """
        prompt = ChatPromptTemplate.from_template(template)
        sql_chain = (
        RunnablePassthrough.assign(schema=lambda x: self.get_schema())
        | prompt
        | self.llm.bind(stop=["\nSQLResult:"])
        | StrOutputParser()
    )
        return sql_chain.invoke({"question": self.query})
    
    def sql_result(self):
        sql_query_=self.sql_query()
        return sql_query_, self.db.run(sql_query_)
        
    def main_chain(self):
        sql_query,sql_query_result=self.sql_result()
        final_prompet=PromptTemplate(
        input_variables=['schema','sqlquery','dbresult','vectorresult','question'],
        template="""
        According to the DB schema: {schema},
        I ran the SQL query: {sqlquery},
        which give me resutl: {dbresult},
        and according the information: {vectorresult}
        answer the question: {question}
        """
        )
        chain=LLMChain(
        llm=self.llm,prompt=final_prompet,verbose=False,output_key='person')
        print(chain.run({"schema":self.get_schema(),"sqlquery":sql_query,"dbresult":sql_query_result,"vectorresult":self.vector_db_find(),
                   "question":self.query}))

query="My id is 5 how many my leaves remaining this year?"





# #step1 extract_chunks 
agent=basic()
agent.load_document()
#incase you replace the document uncoment these both lines 
# agent.get_embeddings()
# agent.store_embeddings()
agent.read_embeddings()
agent.vector_db_insert()
agent.set_query(query)
agent.main_chain()