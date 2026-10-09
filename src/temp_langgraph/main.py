import bs4
import requests
from langchain_core.documents import Document
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

# Below is a minimal helper for demonstration purposes.
def load_web_page(url: str, bs_kwargs: dict | None = None) -> list[Document]:
    response = requests.get(url, timeout=20)
    response.raise_for_status()
    soup = bs4.BeautifulSoup(response.text, "html.parser", **(bs_kwargs or {}))
    return [Document(page_content=soup.get_text(), metadata={"source": url})]


urls = [
    "https://lilianweng.github.io/posts/2024-11-28-reward-hacking/",
    "https://lilianweng.github.io/posts/2024-07-07-hallucination/",
    "https://lilianweng.github.io/posts/2024-04-12-diffusion-video/",
]

docs = [load_web_page(url) for url in urls]

docs_list = [item for sublist in docs for item in sublist]

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=100,
    chunk_overlap=50,
)
doc_splits = text_splitter.split_documents(docs_list)


from functools import lru_cache

from langchain_core.vectorstores import InMemoryVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings


@lru_cache(maxsize=1)
def _get_retriever():
    vectorstore = InMemoryVectorStore.from_documents(
        documents=doc_splits,
        embedding=GoogleGenerativeAIEmbeddings(model="gemini-embedding-2"),
    )
    return vectorstore.as_retriever()


from langchain.tools import tool


@tool
def retrieve_blog_posts(query: str) -> str:
    """Search and return information about Lilian Weng blog posts."""
    retriever = _get_retriever()
    retrieved_docs = retriever.invoke(query)
    return "\n\n".join([doc.page_content for doc in retrieved_docs])


retriever_tool = retrieve_blog_posts