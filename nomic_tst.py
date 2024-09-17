from sentence_transformers import SentenceTransformer

model = SentenceTransformer('nomic-ai/nomic-embed-text-v1', trust_remote_code=True)

sentences = ['search_query: What is TSNE?’, ‘search_query: Who is Laurens van der Maaten?']

embeddings = model.encode(sentences)

print(embeddings)
print(len(embeddings[0]))

# from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings

model_name = "sentence-transformers/all-mpnet-base-v2"
model_name = 'nomic-ai/nomic-embed-text-v1'
model_kwargs = {'device': 'cpu','trust_remote_code':True}
encode_kwargs = {'normalize_embeddings': True, 'weights_only': True}

hf = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

em=hf.embed_documents(sentences)
print(embeddings)
print(len(em[0]))


print("""
  ---------------------------------------------------------------
   4.52690804e-03 -2.63925381e-02  7.41514983e-03  1.48856621e-02
  -3.09508499e-02  1.32238921e-02  4.03460022e-03  1.68822948e-02
  -3.03972531e-02  7.28435963e-02 -7.83350598e-03 -1.54700233e-02
  -3.37302312e-02 -6.24511298e-03  1.17269112e-02  3.97152342e-02
  -1.11692110e-02  1.97132062e-02 -6.06567506e-03 -5.31067746e-03]]
""")