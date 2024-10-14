import torch



def llm(tokenizer, model, question: str, top_k: list[str]) -> str:
    top_k_str = "".join(top_k)
    prompt = (
        "假如你是一个智能课程助手，根据你的知识库已知信息回答问题，如果根据已有信息无法回答，请回答“抱歉，我的知识库没有相关信息。”\n"
        + "问题：\n"
        + question + "\n"
        + "已知信息：\n"
        + top_k_str
    )
    print(prompt)
    # 检查是否有可用的 GPU，否则使用 CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 将模型移动到设备
    model.to(device)
    
    # 编码输入文本，并将其移动到相同的设备
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    # 生成文本
    output = model.generate(**inputs, max_length=200, do_sample=True, top_k=50)

    # 解码生成的文本
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

    # 去除 prompt 部分，保留生成的内容
    if generated_text.startswith(prompt):
        response = generated_text[len(prompt):].strip()
    else:
        response = generated_text.strip()

    return response
