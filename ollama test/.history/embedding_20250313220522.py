from ollama import embeddings

resopnse = embeddings(model='nomic-embed-text', prompt='hello')
print(resopnse['embedding'])
print(type(resopnse['embedding']))
