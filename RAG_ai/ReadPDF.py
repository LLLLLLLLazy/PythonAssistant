import json
import pdfplumber
def Readpdf():
    pdf = pdfplumber.open("testfiles\\初赛训练数据集.pdf")
    len(pdf.pages)  # 页数
    pdf.pages[0].extract_text()

    pdf_content = []
    for page_idx in range(len(pdf.pages)):
        pdf_content.append({
            'page': 'page_' + str(page_idx + 1),
            'content': pdf.pages[page_idx].extract_text()
        })

    return pdf_content

def Loadquestion():
    questions = json.load(open("testfiles\\questions.json", encoding='utf-8'))
    print(questions[0])
    return questions