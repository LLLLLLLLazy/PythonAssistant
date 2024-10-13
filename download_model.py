from transformers import AutoTokenizer, AutoModel


from config import EMBEDDING_MODEL_NAME, EMBEDDING_MODEL_PATH, LLM_NAME, LLM_PATH



# 下载embedding model及其tokenizer
tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL_NAME)
model = AutoModel.from_pretrained(EMBEDDING_MODEL_NAME)

tokenizer.save_pretrained(EMBEDDING_MODEL_PATH)
model.save_pretrained(EMBEDDING_MODEL_PATH)


# 下载llm及其tokenizer
tokenizer = AutoTokenizer.from_pretrained(LLM_NAME)
model = AutoModel.from_pretrained(LLM_NAME, trust_remote_code=True)

tokenizer.save_pretrained(LLM_PATH)
model.save_pretrained(LLM_PATH)