# News Article Industry Classifier

This project classifies news articles into various industries using natural language processing (NLP) techniques. It leverages pre-trained models from the `transformers` library to summarize articles and perform zero-shot classification based on a predefined list of industries. The data is fetched from a MySQL database, processed, and categorized efficiently.

## Features
- Summarizes long news articles using `facebook/bart-large-cnn`.
- Classifies summaries into industries using `facebook/bart-large-mnli` with zero-shot classification.
- Handles large datasets with chunk-based summarization and multi-threading for performance.

## Prerequisites
- Python 3.10+
- MySQL database with a `news_articles` table (columns: `title`, `content`)
- CUDA-enabled GPU (optional, for faster processing)

## Usage

1. **Run the script:**
   ```bash
   python classify_articles.py
   ```
   This will:
   - Connect to your MySQL database.
   - Fetch articles from the `news_articles` table.
   - Summarize and classify each article into an industry.
   - Output the results as a dictionary and print classifications.

2. **Example output:**
   ```
   기사: AI Search Startup Perplexity in Talks for $9 Billion Valuation -> 예측 레이블: Artificial Intelligence

   기사: Most Asian Shares Rise, Gold Touches Record High: Markets Wrap -> 예측 레이블: Asset Management and Investment

   기사: Oil Rebounds as Israel Plans Next Iran Move After Weekend Attack -> 예측 레이블: Energy

   기사: Bitcoin Flirts With $70,000 After $2.4 Billion Inflow Into ETFs -> 예측 레이블: Crypto & Blockchain
   ```

## Project Structure
```
NewsIndustryClassifier/
├── classify_articles.py  
└── README.md            
```

## How It Works
1. **Database Connection**
2. **Text Processing**:
   - Long articles (>1024 tokens) are split into chunks and summarized.
   - Shorter articles are truncated and summarized directly.
3. **Classification**: Uses zero-shot classification to map summaries to industries.

## Customization
- **Industries**: Modify the `industries` list in `classify_articles.py` to suit your needs.
- **Database**: Adjust the SQL query or table name in the script if your schema differs.
- **Models**: Swap out `facebook/bart-large-cnn` or `facebook/bart-large-mnli` for lighter models for faster inference.

## License

[MIT License](LICENSE)

## Contact

Email: mihy1968@gmail.com

---
