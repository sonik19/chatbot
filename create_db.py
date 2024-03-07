from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import Chroma
from langchain.document_loaders import DirectoryLoader
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-base")

loader = DirectoryLoader('./papers/', glob="./*.pdf", loader_cls=PyPDFLoader)

documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=100)
texts = text_splitter.split_documents(documents)

persist_directory = 'db'
vectordb = Chroma.from_documents(documents=texts, 
                                        embedding=embeddings,
                                        persist_directory=persist_directory)

print('DB was succesfully created!')