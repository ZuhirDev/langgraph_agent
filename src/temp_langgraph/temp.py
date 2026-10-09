from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2")
vector = embeddings.embed_query("hello, world!")
print(vector[:5])

