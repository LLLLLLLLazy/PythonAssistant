


CHUNK_LENGTH = 20


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



# 测试
#with open("data/test.text", "r", encoding = 'utf-8') as file:
#    text = file.read()
#print(chunker(text))


