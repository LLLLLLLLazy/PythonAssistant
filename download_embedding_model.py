from transformers import AutoTokenizer, AutoModel


model_name = "sentence-transformers/all-MiniLM-L6-v2"
save_path = "F:\\embedding_model"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

tokenizer.save_pretrained(save_path)
model.save_pretrained(save_path)