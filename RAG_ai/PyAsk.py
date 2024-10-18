import jieba
import pdfplumber
from xinference.client import RESTfulClient
from rank_bm25 import BM25Okapi
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

def Readpdf(pdfname):
    pdf = pdfplumber.open(pdfname)
    len(pdf.pages)  # 页数
    pdf.pages[0].extract_text()

    pdf_content = []
    for page_idx in range(len(pdf.pages)):
        pdf_content.append({
            'page': 'page_' + str(page_idx + 1),
            'content': pdf.pages[page_idx].extract_text()
        })

    return pdf_content


def AskAI(user_content):
    client = RESTfulClient("http://127.0.0.1:9997")
    model = client.get_model("chatglm3")
    messages = [{"role": "system",
                 "content": '''你python是一个专家，帮我结合给定的资料，先简述所给资料的内容再回答一个问题。如果问题无法从资料中获得，请输出给定的资料中没有相关内容。'''},
                {"role": "user", "content": user_content}]
    temp = model.chat(
        messages,
        generate_config={
            "max_tokens": 2048,
            "temperature": 0.7
        }
    )
    return temp


# 读取PDF内容
pdf_content = Readpdf("testfiles\\python_test1.pdf")

# 创建BM25索引
pdf_content_words = [jieba.lcut(x['content']) for x in pdf_content]
bm25 = BM25Okapi(pdf_content_words)

while True:
    # 获取用户输入的提问
    user_question = input("请输入您的问题（输入'退出'以结束）：")
    if user_question.lower() == '退出':
        break

    # 进行BM25检索
    doc_scores = bm25.get_scores(jieba.lcut(user_question))
    max_score_page_idxs = doc_scores.argsort()[-3:]

    # 选择最大的三个分数对应的内容并重排序
    pairs = []
    for idx in max_score_page_idxs:
        pairs.append([user_question, pdf_content[idx]['content']])

    # 加载重排序模型
    tokenizer = AutoTokenizer.from_pretrained('./Rerank/bge-reranker-base/')
    rerank_model = AutoModelForSequenceClassification.from_pretrained('./Rerank/bge-reranker-base/')
    rerank_model.cuda()

    inputs = tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=512)
    with torch.no_grad():
        inputs = {key: inputs[key].cuda() for key in inputs.keys()}
        scores = rerank_model(**inputs, return_dict=True).logits.view(-1, ).float()

    max_score_page_idx = max_score_page_idxs[scores.cpu().numpy().argmax()]

    # 构建提问和参考资料内容
    prompt = '''你是一个python专家，帮我结合给定的资料，先简述所给资料的内容再回答一个问题。如果问题无法从资料中获得，请输出给定的资料中没有相关内容。
    资料：{0}

    问题：{1}

    参考资料页码：{2}
        '''.format(
        pdf_content[max_score_page_idx]['content'],
        user_question,
        pdf_content[max_score_page_idx]['page'][5:]
    )

    answer = AskAI(prompt)['choices'][0]['message']['content']
    print("回答：", answer)