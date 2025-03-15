# �Ȱ�װ��Ҫ��
# pip install pymupdf pdfplumber markdownify unstructured langchain unstructured[pdf]

import re
import fitz  # pymupdf
import pdfplumber
from markdownify import markdownify as md
from langchain_text_splitters import RecursiveCharacterTextSplitter
from unstructured.partition.pdf import partition_pdf

class PDFProcessor:
    def __init__(self, file_path):
        self.file_path = file_path
        self.raw_text = ""
        self.processed_text = ""
        
    def extract_text(self, strategy="hybrid"):
        """������ı���ȡ���Ľ��棩"""
        # ����1������������ȡ���ʺϴ���飩
        with pdfplumber.open(self.file_path) as pdf:
            layout_text = "\n".join([
                self._clean_page(page.extract_text(layout=True))
                for page in pdf.pages
            ])
        
        # ����2�������ṹ��ȡ���ʺ϶��䣩
        elements = partition_pdf(
            self.file_path,
            strategy=strategy,
            infer_table_structure=True,
            include_page_breaks=False
        )
        struct_text = "\n".join([str(el) for el in elements])
        
        # ����3����ȡ����飨PyMuPDF���⴦��
        code_blocks = self._extract_code_blocks()
        
        # �ϲ���ȥ��
        combined_text = self._merge_texts([layout_text, struct_text, code_blocks])
        self.raw_text = combined_text
        return self
    
    def _extract_code_blocks(self):
        """ר�ô������ȡ����"""
        doc = fitz.open(self.file_path)
        code_text = []
        code_pattern = re.compile(r'^\s{4,}|.*>>> |^# .*')  # ʶ����������ʽ����
        
        for page in doc:
            blocks = page.get_text("blocks")
            for b in blocks:
                text = b[4].strip()
                if code_pattern.match(text):
                    # ת��ΪMarkdown�����
                    code_text.append(f"```python\n{text}\n```")
        return "\n".join(code_text)
    
    def clean_text(self):
        """��ǿ���ı���ϴ"""
        text = self.raw_text
        
        # 1. ����PDF�����ַ�
        text = re.sub(r"\x0c", "\n", text)  # ��ҳ��ת����
        text = re.sub(r"-\n+", "", text)    # �޸��ϴ�
        
        # 2. ����ҳüҳ�ţ�����ҳ����ҳ�ţ�
        text = re.sub(r"Page \d+ of \d+", "", text)
        text = re.sub(r"^��.*ҳ$", "", text, flags=re.MULTILINE)
        
        # 3. �ϲ����ж���
        text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)  # ������ת�ո�
        text = re.sub(r"\n{3,}", "\n\n", text)       # �໻�б�׼��
        
        # 4. ת��Markdown��ʽ
        text = md(text, heading_style="ATX")  # ת�������ʽ
        
        # 5. �����ؼ��ṹ
        text = re.sub(r"(^#+ .*$)", r"\n\1\n", text, flags=re.M)  # ������ӿ���
        
        self.processed_text = text
        return self
    
    def split_chapters(self):
        """���½����ֿܷ�"""
        # ʶ���½ڱ���ģʽ������ʵ�ʽ̲ĵ�����
        chapter_pattern = re.compile(r"^(#+)\s+(��?[һ�����������߰˾�ʮ]+��?)\s+(.*)$")
        
        chapters = []
        current_chapter = []
        current_level = 0
        
        for line in self.processed_text.split('\n'):
            match = chapter_pattern.match(line)
            if match:
                if current_chapter:
                    chapters.append("\n".join(current_chapter))
                current_chapter = [line]
                current_level = len(match.group(1))
            else:
                current_chapter.append(line)
        
        if current_chapter:
            chapters.append("\n".join(current_chapter))
        
        return chapters
    
    def process(self):
        """������������"""
        return (
            self.extract_text()
            .clean_text()
            .split_chapters()
        )

# ʹ��ʾ��
processor = PDFProcessor("python�̲�.pdf")
chapters = processor.process()

# ������ǿ�ͷֿ���
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1024,
    chunk_overlap=256,
    separators=[
        "\n```\n",  # ���ȱ��������
        "\n## ",     # ��������
        "\n### ",    # ��������
        "\n\n",
        "��", "��", "��",
        "\n"
    ],
    keep_separator=True
)

# ���շֿ鴦��
all_chunks = []
for chap in chapters:
    chunks = splitter.split_text(chap)
    all_chunks.extend(chunks)

print(f"����{len(all_chunks)}��֪ʶ��")