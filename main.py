import os
from transformers import AutoTokenizer, AutoModel


from Database import Database
from Llms import generate_prompt



# 知识文件
FILES = ['test0.txt', 'test1.md', 'maogai.txt']
DATABASE_NAME = 'database_test'


EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
K = 3



def create_database():
    tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL_NAME)
    embedding_model = AutoModel.from_pretrained(EMBEDDING_MODEL_NAME)

    database = Database(DATABASE_NAME, tokenizer, embedding_model)

    for file in FILES:
        database.add(file)
    print(f"Add ALL files to {database.db}")

    return database


def AskAI(database, question: str)->str:
    top_k = database.top_k_chunks(question, K)
    prompt = generate_prompt(question, top_k)
    return prompt


def main():
    database = create_database()

    question = input("我是AI课程助手，有什么可以帮你？\n")
    while True:
        answer = AskAI(database, question)
        print(answer)
        question = input("还有什么问题吗？\n")


if __name__ == "__main__":
    main()


