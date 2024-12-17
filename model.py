import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.embeddings.openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import Docx2txtLoader
from langchain.prompts import ChatPromptTemplate 
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser



load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_API_KEY"] = openai_api_key
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")


loader = Docx2txtLoader("D:\Projects\AI opinion generator\Indoor Allergen.docx")
loader1 = Docx2txtLoader("D:\Projects\AI opinion generator\Pollens.docx")

data = loader.load()
data1 = loader1.load()

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(data+data1)
embeddings = OpenAIEmbeddings()
db = FAISS.from_documents(docs, embeddings)
retriever = db.as_retriever()

llm = ChatOpenAI(model_name = "gpt-4o-mini", temperature=0)


template = """ you are an expert allergy specialist. 
        Analyze the test report which is given as the query/Question.
        Given the following Context, generate as many specific questions asking the patient 
        whether they have the corresponding symptoms in the context if the test report/query has the condition "Positive". 

        Question: {question}
        Context: {context}
        
        Answer:"""

prompt = ChatPromptTemplate.from_template(template)


rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    |StrOutputParser()
    
)


loader2 = Docx2txtLoader("D:\Projects\AI opinion generator\MAMTHA A9151.docx")
data2 = loader2.load()


Ques=rag_chain.invoke(data2[0].page_content)
print(Ques)


template1 = Ques+""" you are an expert allergy specialist. 
        Analyze the Questions and provide an opinion on whether the patient really has those conditions or not.
       
        Answers:{answers}
        
        """

prompt1 = ChatPromptTemplate.from_template(template1)

rag_chain1 = (
    {"answers": RunnablePassthrough()}
    | prompt1
    | llm
    |StrOutputParser()
    
)

ip=input("Enter the answer: ")
opinion= rag_chain1.invoke(ip)


print(opinion)