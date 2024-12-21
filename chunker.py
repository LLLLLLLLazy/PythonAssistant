

from config import CHUNK_LENGTH, OVERLAP



def chunking(text: str, chunk_length: int=CHUNK_LENGTH, overlap: int=OVERLAP) -> list[str]:
    """
    将文本分块，每块长度为 chunk_length，块之间有 overlap 的重叠。

    参数:
    - text (str): 输入文本
    - chunk_length (int): 每块的长度
    - overlap (int): 块之间的重叠长度

    返回:
    - List[str]: 分块后的文本列表
    """
    if chunk_length <= overlap:
        raise ValueError("chunk_length should be greater than overlap")

    paragraphs = [p for p in text.split("\n\n") if p.strip()]

    for paragraph in paragraphs:
        print(paragraph)
    chunk_list = []
    
    for paragraph in paragraphs:
        sentences = paragraph.split("。")
        current_chunk = ""
        for sentence in sentences:
            if len(current_chunk) + len(sentence) + 1 <= chunk_length:
                current_chunk += sentence + "。"
            else:
                chunk_list.append(current_chunk.strip())
                current_chunk = sentence + "。"
        if current_chunk:
            chunk_list.append(current_chunk.strip())

    for chunk in chunk_list:
        print(chunk)

    if overlap > 0:
        final_chunks = []
        for i in range(len(chunk_list)):
            if i == 0:
                final_chunks.append(chunk_list[i])
            elif i < len(chunk_list) - 1:
                pre_chunk_length = len(chunk_list[i - 1])
                next_chunk_length = len(chunk_list[i + 1])
                if pre_chunk_length < overlap:
                    pre_overlap = pre_chunk_length
                else:
                    pre_overlap = overlap
                if next_chunk_length < overlap:
                    next_overlap = next_chunk_length
                else:
                    next_overlap = overlap
                overlap_text = chunk_list[i - 1][-pre_overlap:] + chunk_list[i] + chunk_list[i + 1][:next_overlap]
                final_chunks.append(overlap_text)
            else:
                pre_chunk_length = len(chunk_list[i - 1])
                if pre_chunk_length < overlap:
                    pre_overlap = pre_chunk_length
                else:
                    pre_overlap = overlap
                final_chunks.append(chunk_list[i - 1][-pre_overlap:] + chunk_list[i])
        return final_chunks

    return chunk_list


