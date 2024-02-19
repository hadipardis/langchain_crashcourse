#!pip install langchain
#!pip install sentence-transformers
#!pip install faiss-gpu
#!pip install pypdf

from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain.embeddings.huggingface import HuggingFaceEmbeddings

loader = PyPDFLoader('langchain.pdf')
#loader = TextLoader("./sample_text_file.txt")

documents = loader.load()

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

embeddings = HuggingFaceEmbeddings()
db = FAISS.from_documents(docs, embeddings)

from langchain import HuggingFaceHub

repo_id = "tiiuae/falcon-7b-instruct"
key = "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" #Use your own API key

llm=HuggingFaceHub(huggingfacehub_api_token= key,
                   repo_id=repo_id,
                   model_kwargs={"temperature":0.1 ,"max_length":512})

from langchain.chains import RetrievalQA
from langchain.schema import retriever

retriever = db.as_retriever()
chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

query="When LangChain was lunched?"
out = chain.invoke(query)
print(out['result'])

