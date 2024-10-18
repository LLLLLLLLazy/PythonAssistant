import os
from transformers import AutoTokenizer, AutoModel, Qwen2ForCausalLM


from config import EMBEDDING_MODEL_PATH, LLM_PATH, FILES_PATH, DATABASE_NAME
from Database import Database
from Llms import llm


class Runner:
    def __init__(self):
        self.embedding_tokenizer = None
        self.embedding_model = None
        self.llm_tokenizer = None
        self.llm = None

    def init_models(self):
        print("Loading Embedding Model...")
        self.embedding_tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL_PATH)
        self.embedding_model = AutoModel.from_pretrained(EMBEDDING_MODEL_PATH)

        print("Loading LLM...")
        self.llm_tokenizer = AutoTokenizer.from_pretrained(LLM_PATH)
        self.llm = Qwen2ForCausalLM.from_pretrained(LLM_PATH)
    
        print("Models loaded.")

    def init_database(self):
        self.database = Database(DATABASE_NAME, self.embedding_tokenizer, self.embedding_model)
    
        FILES = os.listdir(FILES_PATH)
        files = FILES.copy()

        for file in self.database.files:
            if file not in files:
                files.append(file)
    
        for file in files:
            if file in self.database.files and file in FILES:
                print(f"{file} already in {self.database.db}.")
                continue
            elif file in self.database.files:
                self.database.delete(file)
                print(f"{file} deleted from {self.database.db}.")
            else:
                self.database.add(file)
                print(f"{file} added to {self.database.db}.")
    
        print(f"{self.database.db} initialized.")

    def ask_ai(self):
        question = input("我是AI课程助手，有什么可以帮你？\n")
        while True:
            top_k = self.database.top_k_chunks(question)
            answer = llm(self.llm_tokenizer, self.llm, question, top_k)
            print(answer)
            question = input()
    
    

def main():
    ai = Runner()
    ai.init_models()
    ai.init_database()
    ai.ask_ai()

if __name__ == "__main__":
    main()
