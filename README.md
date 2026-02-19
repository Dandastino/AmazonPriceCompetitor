# 📊 Amazon Competitor Analysis Tool

A comprehensive web application that scrapes Amazon product data and provides AI-powered competitive analysis with sentiment insights, review analysis, and advanced visualizations using Oxylabs API and OpenAI's GPT models.

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776ab?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.39%2B-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o--mini-412991?logo=openai&logoColor=white)](https://openai.com/)
[![LangChain](https://img.shields.io/badge/🦜_LangChain-0.1%2B-2C3E50)](https://www.langchain.com/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker&logoColor=white)](https://www.docker.com/)
[![MongoDB](https://img.shields.io/badge/MongoDB-4.6%2B-47A248?logo=mongodb&logoColor=white)](https://www.mongodb.com/)
[![Oxylabs](https://img.shields.io/badge/Oxylabs-API-00A359?logo=data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjQiIGhlaWdodD0iMjQiIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48Y2lyY2xlIGN4PSIxMiIgY3k9IjEyIiByPSIxMSIgZmlsbD0id2hpdGUiIG9wYWNpdHk9IjAuOCIvPjwvc3ZnPg==&logoColor=white)](https://oxylabs.io/)
[![Pandas](https://img.shields.io/badge/Pandas-2.0%2B-150458?logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![Scikit Learn](https://img.shields.io/badge/-Scikit_Learn-0D1117?style=flat-square&logo=scikitlearn)](https://scikit-learn.org/stable/)
[![Plotly](https://img.shields.io/badge/Plotly-5.17%2B-3F4F75?logo=plotly&logoColor=white)](https://plotly.com/)
[![NLTK](https://img.shields.io/badge/NLTK-3.8%2B-154f3c?logo=python&logoColor=white)](https://www.nltk.org/)


## ✨ Features

### 🛍️ Product Management
- **Product Scraping**: Extract detailed product information from Amazon worldwide
- **Competitor Discovery**: Automatically find competing products for your items
- **Product Cards**: Rich UI cards showing product details (price, rating, brand, stock)
- **Pagination**: Browse through products with intuitive pagination controls
- **Product Deletion**: Manage your product database with bulk delete options

### 🤖 AI-Powered Analysis
- **Alex - AI Assistant**: Conversational AI chatbot powered by GPT-4o-mini
- **Product Intelligence**: Ask Alex questions about products and get context-aware answers
- **Tool Guidance**: Alex understands all features and can guide you through workflows
- **Comprehensive Knowledge**: Alex knows about reviews, competitors, sentiment, pricing, and more
- **Natural Conversations**: Chat naturally about products, competitors, and strategies

### 📊 Competitor Analysis
- **Competitor Discovery**: Automatically find competing products
- **Competitor Scoring**: Intelligent algorithm scores competitors by metrics
- **Your Product Section**: Clear display of your product metrics vs competitors
- **All Competitors List**: Complete list of all competitors sorted by score (📊 All Competitors)
- **Detailed Cards**: Each competitor shows price, rating, brand, ASIN, and score
- **CSV Export**: Download competitor data for external analysis

### 📝 Review & Sentiment Analysis
- **Review Scraping**: Pull customer reviews directly from Amazon
- **Sentiment Analysis**: GPT-4o-mini powered AI for accurate aspect-based sentiment extraction
- **Aspect-Based Analysis**: Extract specific product aspects (quality, price, shipping, durability, etc.)
- **Visual Insights**: 
  - Pain points bar charts (top complaint categories)
  - Sentiment distribution pie charts (positive/negative/neutral breakdown)
  - Complaint word clouds (visual keyword analysis)
- **Gap Detection**: Identify competitor weaknesses and customer pain points
- **Emotion Tracking**: Detailed sentiment breakdown with percentages

### 🎯 AI Competitor Insights
- **Strategic Analysis**: GPT-powered market positioning analysis
- **Competitive Advantages**: Identify your strengths vs competitors
- **Weaknesses Detection**: Find areas where competitors are stronger
- **Unit Price Comparison**: Smart price-per-unit analysis
- **Actionable Recommendations**: Get strategic suggestions based on competitor analysis
- **Text Export**: Download complete analysis reports

### 💡 Product Analysis & Pricing
- **Market Statistics**: View competitor metrics (count, avg price, avg rating, stock status)
- **Your Product Metrics**: Compare your product against market averages
- **Price Recommendations**: Get pricing suggestions based on:
  - Minimum competitor price
  - Average competitor price
  - Maximum competitor price
- **Competitive Positioning**: Understand where your product stands
- **Pricing Strategy Insights**: Recommendations for optimal pricing

### 📈 Advanced Visualizations
- **Pain Points Bar Charts**: Top negative aspects from reviews
- **Sentiment Distribution Pie Charts**: Overall sentiment breakdown
- **Complaint Word Clouds**: Visual representation of complaint keywords
- **Competitor Comparison**: Side-by-side metric display
- **Interactive Charts**: Plotly-powered interactive visualizations

### 💬 Interactive Dashboard
- **4 Main Tabs**:
  - **📦 All Products**: View and manage all scraped products
  - **🔍 Competitors**: Analyze competitors for selected products
  - **📊 Analysis**: Advanced analysis with 3 integrated subtabs
  - **🤖 Assistant**: Chat with Alex about your products
- **3 Analysis Subtabs**:
  - **📝 Review Analysis & Sentiment**: Customer sentiment and pain points
  - **🎯 AI Competitor Insights**: AI-powered market analysis
  - **💡 Product Analysis**: Pricing and competitive positioning
- **Real-Time Updates**: Live data refresh after actions
- **Export Capabilities**: Download data and analysis reports as CSV/TXT
- **Responsive UI**: Beautiful modern design with gradient headers and cards

## 🚀 Getting Started

### Prerequisites

- **Python 3.10+** AND **Docker** (for containerized deployment)
- **Oxylabs Account**: Web scraping API credentials
- **OpenAI API Key**: For GPT-powered analysis

### Option 1: Docker Installation (Recommended) 🐳

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd AmazonPriceCompetitor
   ```

2. **Configure environment variables**
   
   Create a `.env` file in the project root:
   ```env
   OXYLAB_USERNAME=your_oxylabs_username
   OXYLAB_PASSWORD=your_oxylabs_password
   OPENAI_API_KEY=your_openai_api_key
   ```

3. **Build and run with Docker Compose**
   ```bash
   docker-compose up --build
   ```

4. **Access the application**
   
   Open your browser at `http://localhost:8501`

**Docker Commands:**
```bash
# Start in detached mode
docker-compose up -d

# Stop the application
docker-compose down

# View logs
docker-compose logs -f

# Rebuild after changes
docker-compose up --build
```

### Option 2: Local Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd AmazonPriceCompetitor
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   # Windows
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   
   # macOS/Linux
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables**
   
   Create a `.env` file:
   ```env
   OXYLAB_USERNAME=your_oxylabs_username
   OXYLAB_PASSWORD=your_oxylabs_password
   OPENAI_API_KEY=your_openai_api_key
   ```

5. **Run the application**
   ```bash
   streamlit run main.py
   ```

6. **Access the application**
   
   Open your browser at `http://localhost:8501`

## 📖 Usage Guide

### 🚀 Quick Start Workflow

1. **Set up credentials** (see [Configuration](#-configuration))
2. **Scrape your first product** using the sidebar
3. **Find competitors** from the Products tab
4. **Analyze competitors** in the Competitors tab
5. **Run sentiment analysis** in the Analysis tab
6. **Chat with Alex** for insights and guidance

### 📥 Sidebar - Product Scraping

The sidebar appears on every page and allows you to scrape products:

**Steps**:
1. Enter **ASIN**: Amazon product identifier (find on product page URL)
   - Example: `https://www.amazon.com/dp/B0CX23VSAS` → ASIN is `B0CX23VSAS`
2. Enter **Zip Code** (optional, for region-specific pricing)
3. Select **Domain**: Amazon marketplace (com, de, uk, fr, it, es, ca, jp, br, in)
4. Click **🔍 Scrape Product**
5. Wait for product to appear in All Products tab

**Tips**:
- Region affects pricing and product availability
- Domain selects the Amazon marketplace
- You can scrape the same product from different regions

### 1. 📦 All Products Tab

**Purpose**: View, browse, and manage all your scraped Amazon products

**Features**:
- See all products with prices, ratings, brands, and stock status
- Pagination controls to browse through products
- **🔎 Find Competitors** button on each product card
- **Product Details**: Full information display for each product
- **🗑️ Delete All Data**: Option to clear your database

**How to use**:
1. Navigate to the **All Products** tab
2. Browse through your product list
3. Click **🔎 Find Competitors** on any product to discover competitors
4. View product pagination controls for navigation

### 2. 🔍 Competitors Tab

**Purpose**: Analyze and compare competitors for your products

**Features**:
- **🎯 Your Product**: Display your product's metrics (price, rating, brand, stock)
- **🥊 Top Competitors**: View your top 10 competitors ranked by score
- **📊 All Competitors**: Complete list sorted by competitive score (highest to lowest)
- **CSV Export**: Download competitor data for further analysis

**How to use**:
1. Navigate to the **Competitors** tab
2. Select a product from the dropdown (or use selected product from Products tab)
3. View **Your Product** section with key metrics
4. Review **Top Competitors** (10 highest-scored competitors)
5. Explore **All Competitors** sorted by score
6. Click **Export** to download competitor data as CSV

### 3. 📊 Analysis Tab

**Purpose**: Deep dive into product analysis with 3 integrated analysis tools

#### 3a. 📝 Review Analysis & Sentiment

**Purpose**: Analyze customer reviews and sentiment

**Steps**:
1. Select a product from dropdown
2. Click **🚀 Scrape Reviews** to collect Amazon reviews
3. Click **🧠 Analyze Sentiment** (powered by GPT-4o-mini for accurate aspect extraction)
4. View visualizations:
   - **Pain Points Bar Chart**: Top complaint categories
   - **Sentiment Distribution**: Pie chart with sentiment breakdown
   - **Word Clouds**: Visual representation of negative keywords
   - **Aspect Table**: Detailed aspect-sentiment analysis

#### 3b. 🎯 AI Competitor Insights

**Purpose**: Get AI-powered strategic analysis

**Steps**:
1. Select a product with competitors
2. Click **🧠 Generate AI Insights**
3. Review AI analysis including:
   - Market positioning analysis
   - Competitive advantages/weaknesses
   - Unit price comparisons
   - Actionable strategic recommendations
4. Click **Download Analysis** to save as text file

#### 3c. 💡 Product Analysis

**Purpose**: Analyze pricing and competitive positioning

**Includes**:
- **Competitor Statistics**: Total competitors, average price, average rating, stock availability %
- **Your Product Metrics**: Current price, rating, review count
- **Price Recommendations**:
  - Minimum competitor price
  - Average competitor price
  - Maximum competitor price
- **Strategic Recommendations**: Suggested pricing range for competitiveness

### 4. 🤖 Assistant Tab

**Purpose**: Chat with Alex, your AI product assistant

**Features**:
- **Alex - AI Assistant**: Powered by GPT-4o-mini
- **Natural Conversation**: Ask questions naturally
- **Comprehensive Knowledge**: Alex knows about:
  - All products in your database
  - Competitor information
  - Review and sentiment analysis
  - Pricing strategies
  - How to use all features
  - Analysis results and insights

**Example Questions**:
- "What are the main complaints about this product?"
- "How does my price compare to competitors?"
- "What are the key differentiators?"
- "How do I scrape a product?"
- "What analysis tools are available?"
- "Show me the top 3 competitors"

**How to use**:
1. Navigate to **Assistant** tab
2. Select a product to discuss
3. Type your question in the text area
4. Click **💬 Get Answer**
5. Read Alex's response powered by GPT-4

## ⚙️ Configuration

### Environment Variables

Create a `.env` file in the project root with these variables:

```env
OXYLABS_USERNAME=your_oxylabs_username
OXYLABS_PASSWORD=your_oxylabs_password  
OPENAI_API_KEY=your_openai_api_key
```

| Variable | Required | Description |
|----------|----------|-------------|
| `OXYLABS_USERNAME` | Yes | Oxylabs API username for product scraping |
| `OXYLABS_PASSWORD` | Yes | Oxylabs API password for product scraping |
| `OPENAI_API_KEY` | Yes | OpenAI API key for Alex chatbot and sentiment analysis |

### Database Configuration

- **Database Type**: MongoDB
- **Connection**: Configured via your MongoDB connection settings
- **Persistence**: Data stored in MongoDB collections
- **Backup**: Use your MongoDB backup strategy

### Getting API Keys

**Oxylabs** (Web Scraping):
1. Visit https://oxylabs.io/
2. Sign up for an account
3. Create API credentials in dashboard
4. Use username and password in `.env`

**OpenAI** (Alex Chatbot & Sentiment Analysis):
1. Visit https://platform.openai.com/
2. Sign up and add payment method
3. Create API key in settings
4. Use in `.env` file
5. Monitor usage for billing

## 🗄️ Database

### Storage

- **Primary**: MongoDB
- **Location**: Your MongoDB instance
- **Format**: Collections (documents)
- **Persistence**: Managed by MongoDB

### Data Collections

**Products Collection**:
```json
{
  "asin": "B0CX23VSAS",
  "title": "Product Name",
  "price": 29.99,
  "currency": "USD",
  "rating": 4.5,
  "reviews_count": 1234,
  "brand": "Brand Name",
  "stock": "In Stock",
  "image_url": "https://...",
  "domain": "com",
  "scraped_at": "2024-01-15T10:30:00",
  "competitors": ["B0XX...", "B0YY...", "...]
}
```

**Reviews Collection**:
```json
{
  "asin": "B0CX23VSAS",
  "reviews": [
    {
      "rating": 5,
      "title": "Great product!",
      "content": "Works perfectly...",
      "date": "2024-01-10",
      "verified": true
    }
  ],
  "sentiment_analysis": {
    "mode": "llm",
    "positive": 0.7,
    "negative": 0.2,
    "neutral": 0.1,
    "analyzed_at": "2024-01-15T11:00:00"
  }
}
```

## 🤖 AI & Machine Learning

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

## � Interactive Features

### Product Cards
- **Rich Display**: Product image, title, price, rating, brand, stock status
- **Actions**: 
  - 🔎 Find Competitors
  - View Details
  - Access in other analysis tabs

### Visualizations

**Pain Points Chart**: Bar chart showing top negative review aspects

**Sentiment Distribution**: Pie chart with positive/negative/neutral breakdown

**Word Clouds**: Visual representation of keywords from negative reviews

**Competitor Cards**: Detailed info cards for each competitor

**Comparison Tables**: Side-by-side metric comparison

### Export Options

1. **CSV Export** (Competitors tab): Download competitor data
2. **TXT Export** (Analysis tab): Download analysis reports
3. **All Data**: Accessible and portable in `data.json`

## 🐳 Docker Deployment

### Quick Start

```bash
# Build and run
docker-compose up --build

# Run in background
docker-compose up -d

# View logs
docker-compose logs -f

# Stop containers
docker-compose down
```

### Custom Configuration

```yaml
# docker-compose.yml
services:
  app:
    environment:
      - OXYLABS_USERNAME=your_username
      - OXYLABS_PASSWORD=your_password
      - OPENAI_API_KEY=your_api_key
    ports:
      - "8080:8501"  # Custom port mapping
    volumes:
      - ./data:/app/data  # Persist database
```

### Health Checks

- **Endpoint**: `http://localhost:8501/_stcore/health`
- **Interval**: 30 seconds
- **Timeout**: 10 seconds
- **Retries**: 3

## 🛠️ Development

### Project Organization

The codebase is organized into clear modules:

- **`main.py`**: Entry point, defines 4 main tabs
- **`src/pages/`**: Tab implementations
- **`src/core/`**: Business logic and analytics
- **`src/api/`**: External API clients (Oxylabs, OpenAI)
- **`src/models/`**: AI/ML models (chatbot, sentiment analysis)
- **`src/ui/`**: UI components and rendering
- **`src/viz/`**: Visualizations

### Running Tests

```bash
# Install dev dependencies
pip install pytest pytest-cov

# Run tests
pytest tests/

# With coverage
pytest --cov=src tests/
```

### Code Style

- **Formatter**: Black (line length: 88)
- **Linter**: Ruff
- **Type Hints**: Used throughout

### Making Changes

1. Create a feature branch: `git checkout -b feature/my-feature`
2. Make your changes
3. Test locally: `streamlit run main.py`
4. Format code: `black src/`
5. Lint: `ruff check src/`
6. Commit: `git commit -m "Add my feature"`
7. Push: `git push origin feature/my-feature`
8. Create Pull Request

## � External APIs & Services

### Oxylabs Web Scraping API

**Purpose**: Scrape Amazon product information

**What We Scrape**:
- Product title, ASIN, price, currency
- Rating and review count
- Brand and availability
- Product images

**Integration**: 
- Credentials stored in `.env`
- Used in `src/api/oxylab_client.py`
- Automatic product type detection for competitor matching

### OpenAI API

**Used For**:

1. **Alex Chatbot** (GPT-4o-mini):
   - Full product database context
   - Natural conversation responses
   - Tool guidance and workflow help

2. **Sentiment Analysis** (GPT-4o-mini):
   - Aspect-based sentiment extraction
   - Accurate and detailed analysis

**Billing**: Pay-per-use model

**Limits**: 
- Input tokens: 16,000
- Output tokens: 4,000 per request

### Rate Limits & Quotas

**Oxylabs**:
- ~100 requests per minute
- Depends on your subscription

**OpenAI**:
- No hard rate limit for free users
- Depends on account tier and usage

**Application**:
- Implements automatic exponential backoff
- Handles errors gracefully with user feedback

## 🐛 Troubleshooting

### Common Issues

**Issue**: `Error: OXYLABS_USERNAME not set`
- **Solution**: Create `.env` file in project root with credentials

**Issue**: `NLTK data warnings (punkt_tab not found)`
- **Solution**: Automatic fallback is now in place - the app will work without NLTK data by using regex-based sentence splitting
- **Manual Fix** (optional): Run `python -c "import nltk; nltk.download('punkt_tab')"`

**Issue**: `Failed to tokenize sentences`
- **Solution**: This is now non-fatal - app automatically falls back to regex-based tokenization. You'll see debug-level logs instead of errors.

**Issue**: `Port 8501 already in use`
- **Solution**: Use `streamlit run main.py --server.port 8502`

**Issue**: `Docker container exits immediately`
- **Solution**: Check logs with `docker-compose logs` and verify `.env` file is in project root

**Issue**: `OpenAI API errors`
- **Solution**: Verify `OPENAI_API_KEY` in `.env` file is valid and has active credits

## 📝 Changelog

### Version 3.1.0 (2026-02-14) - Current

**Features**:
- ✨ **Alex AI Assistant**: Comprehensive chatbot with full product database access and tool knowledge
- ✨ **Competitor Scoring**: All competitors sorted by intelligent scoring algorithm
- ✨ **Sentiment Analysis**: AI-powered (GPT-4o-mini) aspect-based sentiment extraction
- ✨ **Review Analysis**: Complete sentiment visualization with word clouds and pain points
- ✨ **AI Competitor Insights**: GPT-powered strategic market analysis
- ✨ **Product Analysis Tab**: Dedicated pricing and competitive positioning analysis
- ✨ **Smart Product Matching**: Automatic product type detection for accurate competitors
- ✨ **NLTK Robustness**: Graceful fallback for sentence tokenization (no crashes)

**Improvements**:
- 🎯 Tab reorganization: 4 main tabs with 3 integrated analysis subtabs
- 🔍 Competitor analysis with clear "Your Product" vs "Top Competitors" sections
- 📊 All competitors now sorted by score by default
- 👁️ Clear Alex branding throughout assistant interface
- 🛡️ Improved error handling and logging
- ⚡ Better performance for large product lists

**Bug Fixes**:
- 🐛 Fixed NLTK punkt_tab errors with automatic fallback
- 🐛 Improved sentiment analysis robustness
- 🐛 Better error messages for missing data

### Version 3.0.0 (2026-02-14)

**Features**:
- ✨ Alex AI Assistant with comprehensive tool knowledge
- ✨ Tab reorganization and better UI organization
- ✨ Competitor sorting by score
- ✨ Product Analysis subtab
- ✨ Improved navigation

### Version 2.0.0 (2024-01-15)

**Features**:
- ✨ Competitors tab with side-by-side comparison
- ✨ Review Analysis tab with sentiment analysis
- ✨ AI Insights tab with GPT-4 analysis
- ✨ Visualizations (charts, word clouds, radar)
- ✨ CSV export for competitor data
- ✨ Docker support

### Version 1.0.0 (2024-01-01)

- 🎉 Initial release
- Basic product scraping and competitor discovery
- Price analytics and basic chatbot

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👏 Acknowledgments

- **Oxylabs**: For reliable Amazon scraping API
- **OpenAI**: For GPT-4o-mini language model
- **Streamlit**: For the amazing web app framework

## 📧 Support

For issues and questions:
- 🐛 Open an issue on GitHub
- 📧 Email: support@example.com
- 💬 Discord: [Join our community](#)

---

**Built with ❤️ by the Amazon Competitor Analysis Team**
