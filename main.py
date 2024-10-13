from transformers import AutoTokenizer, AutoModel, Qwen2ForCausalLM


from config import EMBEDDING_MODEL_PATH, LLM_PATH, FILES, DATABASE_NAME, K
from Database import Database
from Llms import llm



def init():
    print("Loading Embedding Model...")
    tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL_PATH)
    embedding_model = AutoModel.from_pretrained(EMBEDDING_MODEL_PATH)

    print("Loading LLM...")
    llm_tokenizer = AutoTokenizer.from_pretrained(LLM_PATH)
    llm = Qwen2ForCausalLM.from_pretrained(LLM_PATH)

    return tokenizer, embedding_model, llm_tokenizer, llm


def create_database(tokenizer, embedding_model):
    database = Database(DATABASE_NAME, tokenizer, embedding_model)

    for file in FILES:
        database.add(file)
    print(f"Add ALL files to {database.db}")

    return database


def AskAI(database, llm_tokenizer, llmodel):
    question = input("我是AI课程助手，有什么可以帮你？\n")
    while True:
        top_k = database.top_k_chunks(question, k=K)
        answer = llm(llm_tokenizer, llmodel, question, top_k)
        print(answer)
        question = input("还有什么问题吗？\n")


def main():
    embedding_tokenizer, embedding_model, llm_tokenier, llm = init()
    database = create_database(embedding_tokenizer, embedding_model)
    AskAI(database, llm_tokenier, llm)


if __name__ == "__main__":
    main()


