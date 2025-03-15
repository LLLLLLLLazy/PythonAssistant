# 先安装必要库
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
        """多策略文本提取（改进版）"""
        # 策略1：保留布局提取（适合代码块）
        with pdfplumber.open(self.file_path) as pdf:
            layout_text = "\n".join([
                self._clean_page(page.extract_text(layout=True))
                for page in pdf.pages
            ])
        
        # 策略2：保留结构提取（适合段落）
        elements = partition_pdf(
            self.file_path,
            strategy=strategy,
            infer_table_structure=True,
            include_page_breaks=False
        )
        struct_text = "\n".join([str(el) for el in elements])
        
        # 策略3：提取代码块（PyMuPDF特殊处理）
        code_blocks = self._extract_code_blocks()
        
        # 合并并去重
        combined_text = self._merge_texts([layout_text, struct_text, code_blocks])
        self.raw_text = combined_text
        return self
    
    def _extract_code_blocks(self):
        """专用代码块提取方法"""
        doc = fitz.open(self.file_path)
        code_text = []
        code_pattern = re.compile(r'^\s{4,}|.*>>> |^# .*')  # 识别代码的启发式规则
        
        for page in doc:
            blocks = page.get_text("blocks")
            for b in blocks:
                text = b[4].strip()
                if code_pattern.match(text):
                    # 转换为Markdown代码块
                    code_text.append(f"```python\n{text}\n```")
        return "\n".join(code_text)
    
    def clean_text(self):
        """增强型文本清洗"""
        text = self.raw_text
        
        # 1. 处理PDF特殊字符
        text = re.sub(r"\x0c", "\n", text)  # 换页符转换行
        text = re.sub(r"-\n+", "", text)    # 修复断词
        
        # 2. 清理页眉页脚（假设页码在页脚）
        text = re.sub(r"Page \d+ of \d+", "", text)
        text = re.sub(r"^第.*页$", "", text, flags=re.MULTILINE)
        
        # 3. 合并断行段落
        text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)  # 单换行转空格
        text = re.sub(r"\n{3,}", "\n\n", text)       # 多换行标准化
        
        # 4. 转换Markdown格式
        text = md(text, heading_style="ATX")  # 转换标题格式
        
        # 5. 保留关键结构
        text = re.sub(r"(^#+ .*$)", r"\n\1\n", text, flags=re.M)  # 给标题加空行
        
        self.processed_text = text
        return self
    
    def split_chapters(self):
        """按章节智能分块"""
        # 识别章节标题模式（根据实际教材调整）
        chapter_pattern = re.compile(r"^(#+)\s+(第?[一二三四五六七八九十]+章?)\s+(.*)$")
        
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
        """完整处理流程"""
        return (
            self.extract_text()
            .clean_text()
            .split_chapters()
        )

# 使用示例
processor = PDFProcessor("python教材.pdf")
chapters = processor.process()

# 配置增强型分块器
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1024,
    chunk_overlap=256,
    separators=[
        "\n```\n",  # 优先保留代码块
        "\n## ",     # 二级标题
        "\n### ",    # 三级标题
        "\n\n",
        "。", "！", "？",
        "\n"
    ],
    keep_separator=True
)

# 最终分块处理
all_chunks = []
for chap in chapters:
    chunks = splitter.split_text(chap)
    all_chunks.extend(chunks)

print(f"生成{len(all_chunks)}个知识块")