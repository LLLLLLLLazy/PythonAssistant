import jieba
import jieba
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

# #向量索引
tokenizer = AutoTokenizer.from_pretrained('./Rerank/bge-reranker-base/')
rerank_model = AutoModelForSequenceClassification.from_pretrained('./Rerank/bge-reranker-base/')
rerank_model.cuda()

pdf_content_words = [jieba.lcut(x['content']) for x in pdf_content]
bm25 = BM25Okapi(pdf_content_words)
count = 0
for query_idx in range(len(questions)):
    # 首先进行BM25检索
    doc_scores = bm25.get_scores(jieba.lcut(questions[query_idx]["question"]))
    max_score_page_idxs = doc_scores.argsort()[-3:]

    # top3进行重排序
    pairs = []
    for idx in max_score_page_idxs:
        pairs.append([questions[query_idx]["question"], pdf_content[idx]['content']])

    inputs = tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=512)
    with torch.no_grad():
        inputs = {key: inputs[key].cuda() for key in inputs.keys()}
        scores = rerank_model(**inputs, return_dict=True).logits.view(-1, ).float()

    max_score_page_idx = max_score_page_idxs[scores.cpu().numpy().argmax()]
    questions[query_idx]['reference'] = 'page_' + str(max_score_page_idx + 1)
    prompt = '''你是一个汽车专家，帮我结合给定的资料，回答一个问题。如果问题无法从资料中获得，请输出结合给定的资料，无法回答问题。
    资料：{0}

    问题：{1}
    
    参考资料页码：{2}
        '''.format(
        pdf_content[max_score_page_idx]['content'],
        questions[query_idx]["question"],
        pdf_content[max_score_page_idx]['page'][5:]
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



