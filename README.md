# 📊 Amazon Competitor Analysis Tool

A powerful web application that scrapes Amazon product data and provides AI-powered competitive analysis using Oxylabs API and OpenAI's GPT models.

[![Python](https://img.shields.io/badge/Python-3.8%2B-3776ab?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.0%2B-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4-412991?logo=openai&logoColor=white)](https://openai.com/)
[![Oxylabs](https://img.shields.io/badge/Oxylabs-API-00A359?logo=data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjQiIGhlaWdodD0iMjQiIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48Y2lyY2xlIGN4PSIxMiIgY3k9IjEyIiByPSIxMSIgZmlsbD0id2hpdGUiIG9wYWNpdHk9IjAuOCIvPjwvc3ZnPg==&logoColor=white)](https://oxylabs.io/)
[![TinyDB](https://img.shields.io/badge/TinyDB-JSON_DB-4CAF50?logo=database&logoColor=white)](https://tinydb.readthedocs.io/)

## ✨ Features

- **Product Scraping**: Extract detailed product information from Amazon marketplaces worldwide
- **Competitor Discovery**: Automatically find and analyze competing products in the same category
- **AI-Powered Analysis**: Get strategic insights on market positioning and competitive advantages using GPT-4
- **Multi-Marketplace Support**: Works with Amazon.com, .de, .co.uk, .fr, .it, .es, .ca, .jp, .br, and more
- **Geo-Location Targeting**: Scrape with specific postal codes for accurate regional pricing
- **Interactive Dashboard**: Beautiful Streamlit UI with product cards, pagination, and real-time updates
- **Data Persistence**: TinyDB-based storage for products and competitors

## 🏗️ Architecture

```
AmazonPriceCompetitor/
├── main.py                 # Streamlit application entry point
├── src/
│   ├── __init__.py
│   ├── db.py              # TinyDB database wrapper
│   ├── llm.py             # LangChain + OpenAI competitor analysis
│   ├── oxylab_client.py   # Oxylabs API client for scraping
│   └── services.py        # Business logic layer
├── data.json              # TinyDB database file (auto-generated)
├── requirements.txt       # Python dependencies
├── .env                   # Environment variables (create this!)
└── README.md             # This file
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- Oxylabs account with API credentials
- OpenAI API key

### Installation

1. **Clone the repository**
   ```bash
   cd AmazonPriceCompetitor
   ```

2. **Create a virtual environment** (recommended)
   ```powershell
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables**
   
   Create a `.env` file in the project root:
   ```env
   OXYLAB_USERNAME=your_oxylabs_username
   OXYLAB_PASSWORD=your_oxylabs_password
   OPENAI_API_KEY=your_openai_api_key
   ```

### Running the Application

```bash
streamlit run main.py
```

The application will open in your browser at `http://localhost:8501`

## 📖 Usage Guide

### 1. Scraping a Product

1. In the **sidebar**, enter:
   - **Product ASIN** (e.g., `B0CX23VSAS`)
   - **Region/Zip Code** (e.g., `us`, `83980`, or leave blank)
   - **Amazon Domain** (select from dropdown)

2. Click **🔍 Scrape Product**

3. The product will be added to your database and displayed in the main view

### 2. Find Competitors

1. Find your product in the **Products** tab
2. Click **🔎 Find Competitors** on the product card
3. The app scrapes and stores related competitor products automatically

### 3. Analyzing Competitors

1. Open the **Best Product** tab for a quick ranking view
2. Use the **Chatbot** tab to ask Alex for competitor insights and comparisons

### 4. Viewing Results

The analysis provides:
- **Market Summary**: Overview of the competitive landscape
- **Positioning Analysis**: How your product compares to competitors
- **Top Competitors**: List of main competitors with key differentiators
- **Recommendations**: Actionable strategies for improvement

## 🔧 Configuration

### Oxylabs Settings

The application uses the following Oxylabs endpoints:
- `amazon_product` - Product detail scraping
- `amazon_search` - Competitor discovery

Search strategies used:
- `featured` - Amazon's featured products
- `price_asc` - Lowest to highest price
- `price_desc` - Highest to lowest price
- `rating_desc` - Highest rated first
- `avg_rating_desc` - Best average ratings

## 🤖 Chatbot & LLM

The app includes a built-in assistant (Alex) that answers questions about your stored products and competitors.

- **Chatbot**: Uses a Streamlit chat UI to maintain conversation history and respond with database-aware answers.
- **LLM Engine**: Powered by LangChain + OpenAI (`gpt-4o-mini` by default) for natural language responses.
- **Context Source**: Pulls product data from TinyDB and formats it into a readable context for the model.
- **Use Cases**: Ask about best products, compare prices/ratings, or get usage help.

You can adjust the model or temperature in [src/chatbot.py](src/chatbot.py).

## 🗄️ Database

Product and competitor data is stored in a lightweight local JSON database using TinyDB.

- **Storage File**: [data.json](data.json) (auto-generated on first run)
- **Tables**: A single `products` table stores both parent products and competitors.
- **Fields**: ASIN, title, price, rating, brand, stock, domain, geo, categories, images, timestamps, and more.
- **Operations**: Insert, update, search, delete, and bulk delete via [src/db.py](src/db.py).

If you delete all products in the UI, the TinyDB table is truncated but the file remains.

## 🕸️ Web Scraping

Scraping is handled by the Oxylabs Real-Time API client in [src/oxylab_client.py](src/oxylab_client.py).

- **Endpoints**:
   - `amazon_product` for detailed product pages
   - `amazon_search` for competitor discovery
- **Normalization**: Raw API responses are cleaned into a consistent product schema.
- **Rate Limiting**: A short delay (0.1s) is applied between requests.
- **Geo/Domain Support**: Works across multiple Amazon marketplaces and regions.

## 🔍 Competitor Analysis

Once you've scraped your products, the app automatically finds and analyzes competitors.

### How It Works

1. **Competitor Discovery**: Click **🔎 Find Competitors** on any product card in the **Products** tab
2. **Smart Filtering**: The app filters competitors by:
   - **Product Type**: Only shampoos compete with shampoos (not conditioners or creams)
   - **Category**: Same product categories for accuracy
   - **Price Range**: Normalized unit prices (€/100ml) for fair comparison
   - **Marketplace**: Same Amazon domain/region
3. **Data Storage**: Found competitors are stored with category and type information
4. **Analysis Ready**: Competitors analyzed side-by-side with unit price normalization

### Competitor Matching Strategy

The algorithm scores potential competitors by:

- **Product Type Match** (70 pts): Must be same type (e.g., shampoo vs shampoo)
- **Category Match** (50 pts): Shared product categories
- **Normalized Price Proximity** (30 pts): Similar price per 100g/100ml
- **Brand Match** (20 pts): Same brand bonus

✅ **Example**: Searching for "Pantene Shampoo 500ml"
- ✓ Pantene Conditioner 500ml → Different type, FILTERED OUT
- ✓ Schwarzkopf Shampoo 250ml → Same type, same category → INCLUDED
- ✓ Pantene Hair Mask 500ml → Different type, FILTERED OUT

### Competitor Metrics

Each competitor is tracked with:
- **ASIN**: Unique Amazon identifier
- **Title & Brand**: Product name and manufacturer
- **Product Type**: Detected type (shampoo, conditioner, cream, etc.)
- **Price & Currency**: Regional pricing from Oxylabs
- **Unit Price**: Normalized price per 100g/100ml for fair comparison
- **Rating**: Customer rating on a 5.0 scale
- **Stock Status**: Availability information
- **Categories**: Product classification and hierarchy

### Analysis Views

- **Best Product Tab**: Automatically ranks all products (yours + competitors) by price, rating, and stock—see who's winning
- **Chatbot (Alex)**: Ask questions like "How do my competitors compare?" or "Which competitor is best?"
- **Chatbot Functions**: Get specific competitor details, pricing trends, or strategic recommendations

## 🧪 Development

### Project Structure

- **main.py**: Streamlit UI components and page rendering
- **src/db.py**: Database operations with error handling
- **src/llm.py**: LangChain integration for competitive analysis
- **src/oxylab_client.py**: Oxylabs API wrapper with rate limiting
- **src/services.py**: High-level business logic and orchestration

### Key Technologies

- **Frontend**: Streamlit
- **Web Scraping**: Oxylabs Real-Time API
- **AI/ML**: LangChain + OpenAI GPT-5 series
- **Database**: TinyDB (JSON-based)
- **Environment**: python-dotenv
- **Validation**: Pydantic

## 🛡️ Error Handling

The application includes comprehensive error handling:
- API timeout management (30s default)
- Rate limiting between requests (0.1s delay)
- Validation for all inputs (ASINs, domains, etc.)
- Graceful degradation when services are unavailable
- Detailed logging for debugging
