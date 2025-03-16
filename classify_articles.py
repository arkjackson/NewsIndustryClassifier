import torch
from transformers import pipeline, AutoTokenizer
import pymysql

# 분류할 산업 섹터터
industries = [
    "Information Technology (IT)",
    "Telecommunications",
    "Healthcare and Biotechnology",
    "Pharmaceuticals and Life Sciences",
    "Energy",
    "Electrical and Electronics",
    "Semiconductors",
    "Automotive",
]

def truncate_text(text, max_length=1024):
    # 모델 최대 길이에 맞게 텍스트를 자르기
    tokens = tokenizer.tokenize(text)
    tokens = tokens[:max_length]
    return tokenizer.convert_tokens_to_string(tokens)

def split_and_summarize(text, chunk_size=512):
    # 긴 텍스트를 더 작은 청크로 나눠 요약
    tokens = tokenizer.tokenize(text)
    chunks = [tokens[i:i + chunk_size] for i in range(0, len(tokens), chunk_size)]
    summaries = []

    for chunk in chunks:
        chunk_text = tokenizer.convert_tokens_to_string(chunk)
        summary = summarizer(chunk_text, max_length=100, min_length=30, do_sample=False)[0]['summary_text']
        summaries.append(summary)

    return " ".join(summaries)


# MySQL 서버에 연결
connection = pymysql.connect(
    host='DB_HOST',
    user='DB_USER',
    password='DB_PASSWORD',
    database='DB_NAME',
    charset='utf8mb4',
    cursorclass=pymysql.cursors.DictCursor
)

# 모델 로드
device = 0 if torch.cuda.is_available() else -1
summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=device)
tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device=device)

# 데이터 처리
result = {}
try:
    with connection.cursor() as cursor:
        sql = "SELECT * FROM news_articles"
        cursor.execute(sql)
        articles = cursor.fetchall()
        contents = [row['content'] for row in articles]
        titles = [row['title'] for row in articles]

        for idx, content in enumerate(contents):
            if len(tokenizer.tokenize(content)) > 1024:
                summary = split_and_summarize(content)
            else:
                content = truncate_text(content)
                summary = summarizer(content, max_length=150, min_length=30, do_sample=False)[0]['summary_text']

            classification = classifier(summary, candidate_labels=industries)
            result[classification['labels'][0]] = {"title": titles[idx], "content":content}
            print(f"기사: {titles[idx]} -> 예측 레이블: {classification['labels'][0]}")

finally:
    connection.close()