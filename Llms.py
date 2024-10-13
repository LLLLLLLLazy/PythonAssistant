


def llm(tokenizer, model, question: str, top_k: list[str])->str:
    prompt = f"""假如你是一个智能课程助手，根据你的知识库已有信息，回答问题，如果根据已有信息无法回答，请回答‘根据知识库的信息无法回答’：
    问题：{question}
    已知信息：{top_k}"""
    
    prompt_length = len(prompt)

    # 编码输入文本
    inputs = tokenizer(prompt, return_tensors="pt")

    # 生成文本
    output = model.generate(**inputs, max_length=200, do_sample=True, top_k=50)

    # 解码生成的文本
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

    response = generated_text[prompt_length:]

    return response
