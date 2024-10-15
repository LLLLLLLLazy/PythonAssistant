import torch



def llm(tokenizer, model, question: str, top_k: list[str]) -> str:
    top_k_str = ""
    for chunk in top_k:
        top_k_str += chunk + "\n"
    prompt = (
        "给出以下信息：\n"
        + "------------------\n"
        + top_k_str
        + "------------------\n"
        + f"根据以上信息，回答问题：{question}\n"
        + "不要给出无关的信息，如果没有相关信息，请回答“问的个啥呀，回答不了”\n"
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
