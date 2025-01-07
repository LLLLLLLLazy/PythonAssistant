from markitdown import MarkItDown
import re
import os

def convert_files_to_markdown(file_list):
    md = MarkItDown()
    results = []
    for file in file_list:
        result = md.convert(file)
        results.append(result.text_content)
    print("Converted files to markdown")
    return results

def cleaning():
    for file in os.listdir('Processed'):
        with open(f'Processed/{file}', 'r', encoding='utf-8') as f:
            content = f.read()
            content = re.sub(r'[^\x09\x0A\x0D\x20-\x7E\xA0-\xFF\u4E00-\u9FFF]', '', content) # 去除不可见字符

        with open(f'Processed/{file}', 'w', encoding='utf-8') as f:
            f.write(content)
    print("Cleaned files")

def main():
    raw_files = os.listdir('Raw')
    raw_files = [f'Raw/{file}' for file in raw_files]
    markdown_files = convert_files_to_markdown(raw_files)
    for i in range(len(markdown_files)):
        with open(f'Processed/{raw_files[i].split("/")[1].split(".")[0]}.md', 'w', encoding='utf-8') as f:
            f.write(markdown_files[i])
    cleaning()

if __name__ == "__main__":
    main()