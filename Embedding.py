from config import EMBEDDING_NAME

def embedding(client, chunk: str) -> list[float]:
    completion = client.embeddings.create(
        model=EMBEDDING_NAME,
        input=chunk,
        encoding_format="float"
    )
    return completion.data[0].embedding
