import jieba
import numpy as np
from sentence_transformers import SentenceTransformer
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from xinference.client import RESTfulClient
from rank_bm25 import BM25Okapi
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import ReadPDF

def AskAI(user_content):
    client = RESTfulClient("http://127.0.0.1:9997")
    model = client.get_model("chatglm3")
    messages = [{"role": "system", "content": '''你是一个汽车专家，帮我结合给定的资料，回答一个问题。如果问题无法从资料中获得，请输出结合给定的资料，无法回答问题。'''},
    {"role": "user", "content": user_content}]
    temp = model.chat(
        messages,
        generate_config={
            "max_tokens": 512,
            "temperature": 0.7
        }
    )
    return temp

pdf_content = ReadPDF.Readpdf()
questions = ReadPDF.Loadquestion()

# 对提问和PDF内容进行分词
# question_words = [' '.join(jieba.lcut(x['question'])) for x in questions]
# pdf_content_words = [' '.join(jieba.lcut(x['content'])) for x in pdf_content]
#
# tfidf = TfidfVectorizer()
# tfidf.fit(question_words + pdf_content_words)
#
# # 提取TFIDF
# question_feat = tfidf.transform(question_words)
# pdf_content_feat = tfidf.transform(pdf_content_words)
#
# # 进行归一化
# question_feat = normalize(question_feat)
# pdf_content_feat = normalize(pdf_content_feat)
#
# # 检索进行排序
# for query_idx, feat in enumerate(question_feat):
#     score = feat @ pdf_content_feat.T
#     score = score.toarray()[0]
#     max_score_page_idx = score.argsort()[-1] + 1
#     questions[query_idx]['reference'] = 'page_' + str(max_score_page_idx)

# #向量索引
# tokenizer = AutoTokenizer.from_pretrained('./Rerank/bge-reranker-base/')
# rerank_model = AutoModelForSequenceClassification.from_pretrained('./Rerank/bge-reranker-base/')
# rerank_model.cuda()

# pdf_content_words = [jieba.lcut(x['content']) for x in pdf_content]
# bm25 = BM25Okapi(pdf_content_words)
# count = 0
# for query_idx in range(len(questions)):
#     # 首先进行BM25检索
#     doc_scores = bm25.get_scores(jieba.lcut(questions[query_idx]["question"]))
#     max_score_page_idxs = doc_scores.argsort()[-3:]

#     # top3进行重排序
#     pairs = []
#     for idx in max_score_page_idxs:
#         pairs.append([questions[query_idx]["question"], pdf_content[idx]['content']])

#     inputs = tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=512)
#     with torch.no_grad():
#         inputs = {key: inputs[key].cuda() for key in inputs.keys()}
#         scores = rerank_model(**inputs, return_dict=True).logits.view(-1, ).float()

#     max_score_page_idx = max_score_page_idxs[scores.cpu().numpy().argmax()]
#     questions[query_idx]['reference'] = 'page_' + str(max_score_page_idx + 1)
count = 0
model = SentenceTransformer('Embedding\\bge-small-zh-v1.5\\')
question_sentences = [x['question'] for x in questions]
pdf_content_sentences = [x['content'] for x in pdf_content]

question_embeddings = model.encode(question_sentences, normalize_embeddings=True)
pdf_embeddings = model.encode(pdf_content_sentences, normalize_embeddings=True)

for query_idx, feat in enumerate(question_embeddings):
    score = feat @ pdf_embeddings.T
    max_score_page_idx = score.argsort()[-1] + 1
    questions[query_idx]['reference'] = 'page_' + str(max_score_page_idx)
    prompt = '''你是一个汽车专家，帮我结合给定的资料，回答一个问题。如果问题无法从资料中获得，请输出结合给定的资料，无法回答问题。
    资料：{0}

    问题：{1}
    
    参考资料页码：{2}
        '''.format(
        pdf_content[max_score_page_idx]['content'],
        questions[query_idx]["question"],
        pdf_content[max_score_page_idx]['page']
    )
    print(prompt)
    answer = AskAI(prompt)['choices'][0]['message']['content']
    print(answer)
    questions[query_idx]['answer'] = answer

    count+=1
    if(count > 6):
        break


with open('testfiles\\submit.json', 'w', encoding='utf8') as up:
    json.dump(questions, up, ensure_ascii=False, indent=4)


# #BM25的方法
# pdf_content_words = [jieba.lcut(x['content']) for x in pdf_content]
# bm25 = BM25Okapi(pdf_content_words)
#
# for query_idx in range(len(questions)):
#     doc_scores = bm25.get_scores(jieba.lcut(questions[query_idx]["question"]))
#     max_score_page_idx = doc_scores.argsort()[-1] + 1
#     questions[query_idx]['reference'] = 'page_' + str(max_score_page_idx)

# with open('submit.json', 'w', encoding='utf8') as up:
#     json.dump(questions, up, ensure_ascii=False, indent=4)


# #向量索引
# model = SentenceTransformer('Embedding\\bge-small-zh-v1.5\\')
# question_sentences = [x['question'] for x in questions]
# pdf_content_sentences = [x['content'] for x in pdf_content]

# question_embeddings = model.encode(question_sentences, normalize_embeddings=True)
# pdf_embeddings = model.encode(pdf_content_sentences, normalize_embeddings=True)

# for query_idx, feat in enumerate(question_embeddings):
#     score = feat @ pdf_embeddings.T
#     max_score_page_idx = score.argsort()[-1] + 1
#     questions[query_idx]['reference'] = 'page_' + str(max_score_page_idx)

# with open('submit.json', 'w', encoding='utf8') as up:
#     json.dump(questions, up, ensure_ascii=False, indent=4)


# def ask_glm(content):
#     url = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
#     headers = {
#       'Content-Type': 'application/json',
#       'Authorization': generate_token("填写key", 1000)
#     }
#
#     data = {
#         "model": "glm-3-turbo",
#         "messages": [{"role": "user", "content": content}]
#     }
#
#     response = requests.post(url, headers=headers, json=data)
#     return response.json()



