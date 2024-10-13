

from config import CHUNK_LENGTH



def chunker(text: str, chunk_length: int=CHUNK_LENGTH) -> list[str]:
    chunk_list = []

    current = 0
    while(1):
        if current + chunk_length < len(text):
            chunk_list.append(text[current:current+chunk_length])
            current += chunk_length
        else:
            chunk_list.append(text[current:])
            break

    return chunk_list
