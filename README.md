# 📊 Amazon Competitor Analysis Tool
-----------------------
*AI-powered Amazon product and competitor intelligence dashboard for pricing, sentiment, and strategy.*

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776ab?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.39%2B-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o--mini-412991?logo=openai&logoColor=white)](https://openai.com/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker&logoColor=white)](https://www.docker.com/)
[![MongoDB](https://img.shields.io/badge/MongoDB-4.6%2B-47A248?logo=mongodb&logoColor=white)](https://www.mongodb.com/)

## Table of Contents

• [📹 Demo](#-demo) • [🚀 project overview](#-project-overview) • [📥 Setup Guide](#-setup-guide) • [📖 How to use](#-how-to-use) • [💡 Optimizations](#-optimizations) • [✨ Features](#-features) • [📃 License](#-license)

## 📹 Demo
[Watch demo video (MP4)](assets/amazon.mp4)

------------------------------

## 🚀 project overview
This project is a Streamlit web application that helps sellers and product teams analyze Amazon products against their competitors.

### Why it is useful
It is useful because it centralizes tasks that are usually manual and fragmented:
- Product scraping and competitor discovery
- Customer review scraping and sentiment extraction
- AI-generated market insights and pricing suggestions
- Side-by-side visual comparison for faster decision-making

### What problem it solves
The main problem it solves is time-to-insight. Instead of checking each product page, collecting review feedback by hand, and building spreadsheets manually, users can get a structured analysis flow in one place.

### Who it is built for
The application is designed for:
- E-commerce sellers who need fast competitor monitoring
- Product managers who want evidence-based positioning decisions
- Analysts who need exportable data and visual summaries
- Teams that want a guided assistant for interpreting competitive signals

### Architecture
Core capabilities include:
- Scraping Amazon product data via Oxylabs
- Finding and ranking competitors
- Running AI-based sentiment and aspect analysis on reviews
- Generating strategic insights with GPT models
- Recommending price positioning from market statistics

## 📥 Setup Guide
How to install

1. Clone the repository:
```bash
git clone <repository-url>
cd AmazonPriceCompetitor
```

2. Configure environment variables in a `.env` file:
```env
OXYLABS_USERNAME=your_oxylabs_username
OXYLABS_PASSWORD=your_oxylabs_password
OPENAI_API_KEY=your_openai_api_key
```

how to configure

1. Ensure your Oxylabs credentials are valid and enabled for product/review scraping.
2. Ensure your OpenAI API key has active billing and model access.
3. If needed, configure MongoDB connection values used by your local environment.

### Run with Docker

1. Build and start the app:
```bash
docker-compose up --build
```
2. Open: `http://localhost:8501`
3. Stop containers when done:
```bash
docker-compose down
```

### Run without Docker (local)

1. Install dependencies:
```bash
pip install -r requirements.txt
```
2. Run the app:
```bash
streamlit run main.py
```
3. Open: `http://localhost:8501`

## 📖 How to use
step by step usage

1. Open the app and add an Amazon ASIN in the sidebar.
2. Choose marketplace domain and optional zip code, then scrape the product.
3. Go to the products section and trigger competitor discovery.
4. Open competitor analysis to compare price, rating, and score.
5. Run review scraping and sentiment analysis to identify pain points.
6. Generate AI insights to get strategy recommendations.
7. Export CSV/TXT outputs for team sharing.


## ✨ Features

### Alex - AI Assistant

- **Model**: OpenAI GPT-4o-mini
- **Language**: LangChain + OpenAI API
- **Context**: Has full access to your product database
- **Knowledge**: Understands all features, workflow, and analysis capabilities
- **Capabilities**:
  - Answer product questions
  - Provide competitive analysis
  - Guide through analysis tools
  - Explain features and workflow
  - Suggest strategies based on data

### Sentiment Analysis Model

**LLM Mode (OpenAI GPT-4o-mini)**
- **Use Case**: Accurate aspect-based sentiment extraction
- **Accuracy**: ~92% on Amazon reviews
- **Speed**: 2-3 seconds per batch (10 reviews)
- **Cost**: Low (uses GPT-4o-mini, less expensive model)
- **Output**: Detailed aspect sentiments and reasoning

### Text Preprocessing

- **Library**: NLTK 3.8+
- **Features**:
  - Sentence tokenization with fallback support
  - Word tokenization
  - Stopword removal (English)
  - Lemmatization (WordNetLemmatizer)
  - Text cleaning (HTML, URLs, special characters)
- **Robust**: Graceful fallbacks when NLTK resources unavailable

### Competitor Analysis

- **Algorithm**: Custom scoring based on:
  - Price competition (vs average)
  - Rating scores (vs average)
  - Review volume (vs average)
  - Stock availability
- **Output**: Competitor rank scores used for sorting
- **Product Type Matching**: Smart product category detection for accurate competitor finding

### Data Processing

- **Framework**: pandas, numpy, scikit-learn
- **Feature Engineering**: Automatic extraction of price, ratings, reviews
- **Data Validation**: Pydantic models for robust data handling

## 💡 Optimizations
What can be the next step to make it even better

- Add scheduled background scraping jobs for daily/weekly monitoring.
- Introduce alerting when competitor price or rating crosses thresholds.
- Track historical trends to visualize how competition evolves over time.
- Add role-based access and workspace separation for multi-team usage.
- Improve model prompts with few-shot examples for more consistent insights.
- Add automated report generation and email/slack delivery.


## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.