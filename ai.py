from openai import OpenAI
import time


from database import Database
from config import DATABASE_NAME, LLM_NAME

class AI():
    def __init__(self, db=DATABASE_NAME):
        self.db = Database(db)
        self.messages = [
            {
                "role": "system",
                "content": """你是一名AI课程助手，你的任务是回答用户的问题。
                根据已知信息回答用户的问题，如果已知信息不足，请回答“问的个啥呀，回答不了”。""",
            }
        ]
        self.client = OpenAI(
                api_key="sk-28053374a8a74d44807e9c7743ab92ee",
                base_url="https://api.deepseek.com",
            )
    
    def llm(self, prompt: str) -> str:
        # 将用户问题信息添加到messages列表中
        self.messages.append({"role": "user", "content": prompt})
    
        # 调用大模型生成回复信息
        completion = self.client.chat.completions.create(model=LLM_NAME, messages=self.messages, stream=False)
        response = completion.choices[0].message.content
    
        # 将大模型的回复信息添加到messages列表中
        self.messages.append({"role": "assistant", "content": response})
    
        # 返回回复信息
        return response
    
    def generate_prompt(self, question: str, top_k: list[str]) -> str:
        top_k_str = ""
        for chunk in top_k:
            top_k_str += chunk + "\n"
        prompt = "已知信息：\n" + top_k_str + f"用户问题：{question}\n"
        return prompt
    
    def ask(self):
        question = input()
        t1 = time.time()
        top_k = self.db.top_k_chunks(question)
        t2 = time.time()
        prompt = self.generate_prompt(question, top_k)
        answer = self.llm(prompt)
        t3 = time.time()        
        print(answer)
        print(f"search time: {t2 - t1}")
        print(f"llm time: {t3 - t2}")

def main():
    ai = AI()
    print("我是AI课程助手，有什么可以帮你？")
    while True:
        ai.ask()

if __name__ == "__main__":
    main()