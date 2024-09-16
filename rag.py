import os, sys, time

sys.path.append('../')

from pytools.ollamarag import OllamaRAG
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms       import Ollama
from langchain_chroma               import Chroma


llmModel='qwen2:1.5b'

embeddings = OllamaEmbeddings(model="nomic-embed-text")
vectorstore = Chroma(persist_directory="../Landscape_rag/DBG", embedding_function=embeddings)
rag=OllamaRAG(db=vectorstore,llm_model=llmModel,show_progress=True)

def do_query(query:str, multiquery=None):
  tm_on=time.perf_counter()

  if len(query)>0:

    if multiquery!=None:
      rag.set_multiquery(multiquery)

    answer=rag.query(query)
  else:
    answer='fai una domanda'

  counter=time.perf_counter()-tm_on

  print(answer)

  return {'answer':answer,
          'time': counter}

if __name__ == '__main__':
  print(do_query("protezione ambiente"))