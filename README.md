# parsion-ai-demo

## Description
This project is an AI-powered tool designed for comprehensive analysis of the Indian stock market. It utilizes AI agents to aggregate news, analyze market data, assess sentiment, and generate insightful reports for traders and investors.

## Features
- **News Aggregation:** Collects and processes market-moving news from multiple Indian financial sources.
- **Market Data Analysis:** Analyzes key market indices like Nifty 50 and Bank Nifty, including technical levels, volume, and trends.
- **Sentiment Analysis:** Evaluates overall and sector-specific market sentiment to identify trading signals.
- **Agent-Based Architecture:** Uses PraisonAIAgents to structure and automate the analysis workflow.
- **Report Generation:** Produces detailed reports on market news, analysis, and sentiment.

## Requirements
- Python 3.x
- The dependencies listed in `requirements.txt`

## Installation
1. Clone the repository:
   ```bash
   git clone <repository_url>
   cd parsion-ai-demo
   ```
2. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```
3. Ensure you have necessary API keys or credentials for any external services (like OpenAI, if used by PraisonAIAgents) configured, likely via a `.env` file.

## Usage
To run the market analysis workflow, execute the main application file:

```bash
python app.py
```

The application will run the defined agent tasks and generate output files in the `outputs` directory, categorized by news, market analysis, sentiment, and reports.

## Report Template
The project uses a comprehensive markdown template (`market_analysis_template.md`) for generating market analysis reports. The template includes the following sections:

1. **Market News Analysis**
   - Key Headlines
   - Sector-wise Updates
   - Economic Indicators
   - Corporate Announcements

2. **Market Analysis**
   - Indices Performance
   - Top Gainers/Losers
   - Sector Performance

3. **Market Sentiment Analysis**
   - Overall Market Sentiment
   - Sector-wise Sentiment
   - Risk Assessment

4. **Investment Insights**
   - Market Outlook
   - Buy/Sell Recommendations
   - Key Events to Watch
   - Technical Analysis

5. **FII/DII Activity**
   - Foreign Institutional Investors
   - Domestic Institutional Investors

6. **Market Statistics**
   - Trading Statistics
   - Sector-wise Statistics

The template is automatically populated with real-time data and analysis from various sources, providing a comprehensive view of the Indian stock market.

## Output Files
Analysis results and reports are saved in the `outputs` directory, structured as follows:
- `outputs/news/` - News aggregation and analysis
- `outputs/market/` - Market data and technical analysis
- `outputs/sentiment/` - Market sentiment analysis
- `outputs/reports/` - Generated market analysis reports
- `outputs/investor/` - Investment insights and recommendations

