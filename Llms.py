


def generate_prompt(question: str, top_k: list[str])->str:
    '''
    生成llm的输入
    输入：问题，最相似的k个chunk
    输出：llm的输入
    '''
    prompt = f"""假如你是一个智能课程助手，根据已知信息，回答问题，如果根据已有信息无法回答，请回答‘根据知识库的信息无法回答’：
    问题：{question}
    已知信息：{top_k}"""

    return prompt