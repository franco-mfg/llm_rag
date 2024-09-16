from sentence_transformers import SentenceTransformer

model = SentenceTransformer('nomic-ai/nomic-embed-text-v1', trust_remote_code=True)

sentences = ['search_query: What is TSNE?’, ‘search_query: Who is Laurens van der Maaten?']

embeddings = model.encode(sentences)

print(embeddings)
print(len(embeddings[0]))

# from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings

model_name = "sentence-transformers/all-mpnet-base-v2"
model_name = 'nomic-embed-text-v1'
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': False}

hf = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

em=hf.embed_documents(sentences)
print(len(em[0]))