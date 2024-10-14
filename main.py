import os
from transformers import AutoTokenizer, AutoModel, Qwen2ForCausalLM


from config import EMBEDDING_MODEL_PATH, LLM_PATH, FILES_PATH, DATABASE_NAME
from Database import Database
from Llms import llm



def init_models():
    print("Loading Embedding Model...")
    tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL_PATH)
    embedding_model = AutoModel.from_pretrained(EMBEDDING_MODEL_PATH)

    print("Loading LLM...")
    llm_tokenizer = AutoTokenizer.from_pretrained(LLM_PATH)
    llm = Qwen2ForCausalLM.from_pretrained(LLM_PATH)
    
    print("Models loaded.")
    return tokenizer, embedding_model, llm_tokenizer, llm


def init_database(tokenizer, embedding_model):
    database = Database(DATABASE_NAME, tokenizer, embedding_model)
    
    FILES = os.listdir(FILES_PATH)
    files = FILES.copy()

    for file in database.files:
        if file not in files:
            files.append(file)
    
    for file in files:
        if file in database.files and file in FILES:
            print(f"{file} already in {database.db}.")
            continue
        elif file in database.files:
            database.delete(file)
            print(f"{file} deleted from {database.db}.")
        else:
            database.add(file)
            print(f"{file} added to {database.db}.")
    
    print(f"{database.db} initialized.")
    return database


def ask_ai(database, llm_tokenizer, llmodel):
    question = input("我是AI课程助手，有什么可以帮你？\n")
    while True:
        top_k = database.top_k_chunks(question)
        answer = llm(llm_tokenizer, llmodel, question, top_k)
        print(answer)
        question = input()


def main():
    embedding_tokenizer, embedding_model, llm_tokenier, llm = init_models()
    
    database = init_database(embedding_tokenizer, embedding_model)
    
    ask_ai(database, llm_tokenier, llm)


if __name__ == "__main__":
    main()


