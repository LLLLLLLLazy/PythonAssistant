import torch



def embedding(model, tokenizer, chunk: str) -> list[float]:
    '''
    生成chunk的ebbedding
    需提前：
    tokenizer = AutoTokenizer.from_pretrained(EBBEDDING_MODEL_NAME)
    model = AutoModel.from_pretrained(EBBEDDING_MODEL_NAME)
    '''
    # 处理文本
    inputs = tokenizer(chunk, return_tensors='pt')
    
    with torch.no_grad():
        embedings = model(**inputs).last_hidden_state
    
    # 获取句子的嵌入（通常取最后一层的[CLS]标记或平均）
    vector = embedings[:, 0, :].squeeze().numpy()
    
    return vector.tolist()

