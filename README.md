# 📊 Amazon Competitor Analysis Tool

A powerful web application that scrapes Amazon product data and provides AI-powered competitive analysis using Oxylabs API and OpenAI's GPT models.

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

### 2. Analyzing Competitors

1. Find your product in the **Stored Products** section
2. Click **🔎 Analyze Competitors** on the product card
3. Click **🔄 Fetch Competitors** to scrape competitor data
4. Navigate to the **🤖 LLM Analysis** tab
5. Click **🚀 Run LLM Analysis** for AI-powered insights

### 3. Viewing Results

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

## 🎯 Future Enhancements

- [ ] Price history tracking and alerts
- [ ] Export analysis to PDF/Excel
- [ ] Scheduled automatic competitor monitoring
- [ ] Support for more marketplaces
- [ ] Advanced filtering and search
- [ ] Bulk product import via CSV
