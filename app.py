from dotenv import load_dotenv
load_dotenv()
from praisonaiagents import Agent, PraisonAIAgents, Task
import time
from datetime import datetime, timedelta
import json
import os
from IPython.display import display, Markdown
import glob
import yfinance as yf
import pandas as pd
import numpy as np

# Create output directories
os.makedirs('outputs/news', exist_ok=True)
os.makedirs('outputs/market', exist_ok=True)
os.makedirs('outputs/sentiment', exist_ok=True)
os.makedirs('outputs/reports', exist_ok=True)
os.makedirs('outputs/investor', exist_ok=True)

# Create specialized agents for Indian market analysis
news_aggregator = Agent(
    name="News Aggregator",
    role="Financial News Collector",
    goal="Collect and aggregate market-moving news from multiple sources",
    instructions="""Collect and analyze news from the following sources:
    1. MoneyControl (https://www.moneycontrol.com/stocksmarketsindia/)
    2. Economic Times (https://economictimes.indiatimes.com/markets)
    3. Angel One (https://www.angelone.in/share-market-today)
    4. CNBC TV18 (https://www.cnbctv18.com/)
    
    Return the analysis in this exact format:

# Market News

## High-Impact Headlines
- [List top 3-5 market-moving headlines]
- For each headline:
  * Source: [Source name]
  * Impact: [Market impact]
  * Affected: [Sectors/stocks]

## Sector Updates
[Only include sectors with significant news]

### [Sector Name]
- Key News: [List relevant news]
- Market Impact: [Analysis]
- Trading Impact: [Opportunities/Risks]

## Market Catalysts
- [List key events with timing]
- [Expected market impact]

## Trading Signals
- [List specific opportunities]
- [Include entry/exit points]
"""
)

news_analyzer = Agent(
    name="News Analyzer",
    role="Market News Processor",
    goal="Process and analyze significant market news for trading opportunities",
    instructions="""Analyze significant market news and return in this exact format:

# Market Analysis

## Key Events
- [List 3-5 most important events]
- For each event:
  * Impact: [Market impact]
  * Timeline: [Time horizon]
  * Trading Impact: [Opportunities/Risks]

## Sector Analysis
[Only analyze sectors with significant news]

### [Sector Name]
- Market Impact: [Analysis]
- Trading Opportunities:
  * [Specific opportunities]
  * [Entry/Exit points]
- Risk Factors:
  * [Key risks]
  * [Mitigation strategies]

## Market Structure
- Trend: [Current market structure]
- Key Levels: [Important price levels]
- Volume: [Unusual patterns]
- Breadth: [Advance/Decline analysis]

## Trading Opportunities
### Short-term (1-3 days)
- [List opportunities]
- [Entry/Exit points]

### Medium-term (1-2 weeks)
- [List opportunities]
- [Entry/Exit points]
"""
)

market_analyzer = Agent(
    name="Market Analyzer",
    role="Market Data Processor",
    goal="Analyze market data for trading opportunities",
    instructions="""Analyze market data and return in this exact format:

# Market Analysis

## Index Analysis
### Nifty 50
- Current: [Value]
- Change: [Value]%
- Technical Levels:
  * Support: [Levels]
  * Resistance: [Levels]
  * Moving Averages: [Key MAs]
- Volume: [Analysis]
- Range: [Expected range]

### Bank Nifty
- Current: [Value]
- Change: [Value]%
- Technical Levels:
  * Support: [Levels]
  * Resistance: [Levels]
  * Moving Averages: [Key MAs]
- Volume: [Analysis]
- Range: [Expected range]

## Market Structure
- Trend: [Direction and strength]
- Volume: [Analysis]
- Breadth: [Advance/Decline]

## Trading Opportunities
### Index Trading
- Bias: [Bullish/Bearish/Neutral]
- Levels: [Entry/Exit points]
- Stop Loss: [Levels]
- Target: [Levels]

### Options Trading
- Strategies: [List with rationale]
- Risk/Reward: [Analysis]
- Position Size: [Recommendations]
"""
)

sentiment_analyzer = Agent(
    name="Sentiment Analyzer",
    role="Market Sentiment Processor",
    goal="Analyze market sentiment for trading signals",
    instructions="""Analyze market sentiment and return in this exact format:

# Market Sentiment

## Overall Sentiment
- Current: [Positive/Negative/Neutral]
- Score: [Score between -1 and 1]
- Mood: [Analysis]
- Trading Impact: [How to trade]

## Sector Sentiment
[Only analyze sectors with significant changes]

### [Sector Name]
- Sentiment: [Positive/Negative/Neutral]
- Factors:
  * [Key factors]
  * [Impact analysis]
- Opportunities:
  * [Specific opportunities]
  * [Entry/Exit points]
- Risks:
  * [Key risks]
  * [Mitigation strategies]

## Market Psychology
- Fear & Greed: [Value and analysis]
- Positioning: [Analysis]
- Contrarian: [Signals if any]

## Trading Signals
### Short-term (1-3 days)
- [List signals]
- [Entry/Exit points]

### Medium-term (1-2 weeks)
- [List signals]
- [Entry/Exit points]
"""
)

investor_insights = Agent(
    name="Investor Insights",
    role="Investment Advisor",
    goal="Provide actionable trading and investment insights",
    instructions="""Analyze market data and provide insights in this exact format:

# Trading Insights

## Market Opportunities
### Day Trading
- [List 3-5 opportunities]
- Entry: [Levels]
- Stop Loss: [Levels]
- Target: [Levels]
- Risk/Reward: [Ratio]

### Swing Trading (1-5 days)
- [List 3-5 opportunities]
- Entry: [Levels]
- Stop Loss: [Levels]
- Target: [Levels]
- Risk/Reward: [Ratio]

### Position Trading (1-4 weeks)
- [List 3-5 opportunities]
- Entry: [Levels]
- Stop Loss: [Levels]
- Target: [Levels]
- Risk/Reward: [Ratio]

## Trading Strategies
### Index Trading
- Bias: [Bullish/Bearish/Neutral]
- Levels: [Entry/Exit points]
- Stop Loss: [Levels]
- Target: [Levels]

### Options Strategies
- Strategies: [List with rationale]
- Risk/Reward: [Analysis]
- Position Size: [Recommendations]

## Risk Management
### Position Sizing
- Index: [Recommendations]
- Stocks: [Recommendations]
- Options: [Recommendations]

### Stop Loss Management
- Index: [Levels]
- Stocks: [Levels]
- Options: [Levels]

## Action Items
### Today
- [List 3-5 actions]
- [Entry/Exit points]

### This Week
- [List 3-5 actions]
- [Entry/Exit points]
"""
)

report_generator = Agent(
    name="Report Generator",
    role="Report Creator",
    goal="Generate comprehensive trading and investment report",
    instructions="""Generate market analysis insights and return in this exact format:

# Market Report

## Market Overview
- State: [Current market state]
- Trends: [Key market trends]
- Opportunities: [List with rationale]

## Technical Analysis
### Nifty 50
- Support: [Levels with analysis]
- Resistance: [Levels with analysis]
- Trend: [Analysis]
- Volume: [Analysis]
- Range: [Expected range]

### Bank Nifty
- Support: [Levels with analysis]
- Resistance: [Levels with analysis]
- Trend: [Analysis]
- Volume: [Analysis]
- Range: [Expected range]

## Trading Opportunities
### Index Trading
- Bias: [Bullish/Bearish/Neutral]
- Levels: [Entry/Exit points]
- Stop Loss: [Levels]
- Target: [Levels]

### Options Trading
- Strategies: [List with rationale]
- Risk/Reward: [Analysis]
- Position Size: [Recommendations]

## Action Items
### Today
- [List specific actions]
- [Entry/Exit points]

### This Week
- [List specific actions]
- [Entry/Exit points]
"""
)

# Create tasks with routing logic
news_aggregation_task = Task(
    name="aggregate_news",
    description="Collect and aggregate market news from multiple sources",
    expected_output="Comprehensive news aggregation from multiple sources",
    agent=news_aggregator,
    is_start=True,
    next_tasks=["process_news"],
    output_file="outputs/news/news_aggregation.txt"
)

news_processing_task = Task(
    name="process_news",
    description="Process significant market news",
    expected_output="Structured news analysis in markdown",
    agent=news_analyzer,
    next_tasks=["analyze_market"],
    output_file="outputs/news/news_analysis.txt"
)

market_analysis_task = Task(
    name="analyze_market",
    description="Analyze market data and trends",
    expected_output="Market analysis in markdown",
    agent=market_analyzer,
    next_tasks=["analyze_sentiment"],
    output_file="outputs/market/market_analysis.txt"
)

sentiment_analysis_task = Task(
    name="analyze_sentiment",
    description="Analyze market sentiment",
    expected_output="Sentiment analysis in markdown",
    agent=sentiment_analyzer,
    next_tasks=["generate_investor_insights"],
    output_file="outputs/sentiment/sentiment_analysis.txt"
)

investor_insights_task = Task(
    name="generate_investor_insights",
    description="Generate investor-focused insights and recommendations",
    expected_output="Comprehensive investor insights in markdown",
    agent=investor_insights,
    next_tasks=["generate_report"],
    output_file="outputs/investor/investor_insights.txt"
)

report_generation_task = Task(
    name="generate_report",
    description="Generate final market report",
    expected_output="Market analysis insights in markdown",
    agent=report_generator,
    output_file="outputs/reports/final_report.txt"
)

def save_text_report(content, filename):
    """Save content to a text file"""
    try:
        # Create reports directory if it doesn't exist
        os.makedirs('reports', exist_ok=True)
        
        # Save the file
        filepath = os.path.join('reports', filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        
        return filepath
    except Exception as e:
        print(f"Error saving text file: {str(e)}")
        return None

def get_market_data(symbol="^NSEI"):
    """
    Fetch market data for technical analysis
    """
    try:
        # Fetch data for the last 30 days
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        
        # Get data from Yahoo Finance
        data = yf.download(symbol, start=start_date, end=end_date)
        
        if data.empty:
            return None
            
        # Calculate technical levels
        current_price = data['Close'][-1]
        high_30d = data['High'].max()
        low_30d = data['Low'].min()
        
        # Calculate moving averages
        ma20 = data['Close'].rolling(window=20).mean().iloc[-1]
        ma50 = data['Close'].rolling(window=50).mean().iloc[-1]
        
        # Calculate support and resistance levels
        support_levels = [
            round(low_30d, 2),
            round(current_price * 0.95, 2),
            round(current_price * 0.90, 2)
        ]
        
        resistance_levels = [
            round(high_30d, 2),
            round(current_price * 1.05, 2),
            round(current_price * 1.10, 2)
        ]
        
        # Calculate trading range
        trading_range = {
            'lower': round(low_30d, 2),
            'upper': round(high_30d, 2)
        }
        
        return {
            'current_price': round(current_price, 2),
            'support_levels': support_levels,
            'resistance_levels': resistance_levels,
            'moving_averages': {
                'MA20': round(ma20, 2),
                'MA50': round(ma50, 2)
            },
            'trading_range': trading_range
        }
    except Exception as e:
        print(f"Error fetching market data: {str(e)}")
        return None

def format_technical_levels(data):
    """
    Format technical levels into markdown
    """
    if not data:
        return None
        
    return f"""### Key Technical Levels
- Current Price: {data['current_price']}
- Support Levels: {', '.join(map(str, data['support_levels']))}
- Resistance Levels: {', '.join(map(str, data['resistance_levels']))}
- Moving Averages:
  * 20-day MA: {data['moving_averages']['MA20']}
  * 50-day MA: {data['moving_averages']['MA50']}
- Trading Range: {data['trading_range']['lower']} - {data['trading_range']['upper']}
"""

def combine_reports():
    """
    Combines all output files into a concise market report
    """
    try:
        # Get market data
        nifty_data = get_market_data("^NSEI")
        bank_nifty_data = get_market_data("^NSEBANK")
        
        # Define the sequence of analysis steps
        analysis_sequence = [
            {
                'title': 'Market News',
                'pattern': 'news_aggregation.txt'
            },
            {
                'title': 'Market Analysis',
                'pattern': 'market_analysis.txt'
            },
            {
                'title': 'Market Sentiment',
                'pattern': 'sentiment_analysis.txt'
            },
            {
                'title': 'Trading Insights',
                'pattern': 'investor_insights.txt'
            }
        ]
        
        # Create a comprehensive report
        report = f"""# Market Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}

"""
        
        # Add technical analysis section if data is available
        if nifty_data or bank_nifty_data:
            report += "## Technical Analysis\n"
            if nifty_data:
                report += "### Nifty 50\n"
                report += format_technical_levels(nifty_data)
            if bank_nifty_data:
                report += "### Bank Nifty\n"
                report += format_technical_levels(bank_nifty_data)
            report += "\n"
        
        # Process each step in sequence
        for step in analysis_sequence:
            # Find the most recent file matching the pattern
            matching_files = glob.glob(f'outputs/**/{step["pattern"]}', recursive=True)
            if matching_files:
                # Sort by modification time (newest first)
                latest_file = sorted(matching_files, key=os.path.getmtime, reverse=True)[0]
                
                try:
                    with open(latest_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # Skip empty sections or sections with placeholder values
                    if content and not all('[Value]' in line or '[Levels]' in line or '[Analysis]' in line for line in content.split('\n')):
                        # Clean up the content
                        content = content.replace('##', '###')  # Make all headers consistent
                        content = '\n'.join(line for line in content.split('\n') 
                                          if not any(x in line for x in ['[Value]', '[Levels]', '[Analysis]', '---']))
                        
                        report += f"## {step['title']}\n"
                        report += content.strip() + "\n\n"
                    
                except Exception as e:
                    print(f"Error reading file {latest_file}: {str(e)}")
        
        # Save the combined report as markdown
        combined_report_path = f"outputs/market_report_{datetime.now().strftime('%Y%m%d_%H%M')}.md"
        with open(combined_report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"\nReport saved to: {combined_report_path}")
        return combined_report_path
        
    except Exception as e:
        print(f"Error combining reports: {str(e)}")
        return None

def run_agentic_workflow():
    print(f"\n=== Starting Agentic Market Analysis - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===\n")
    
    try:
        # Create and run workflow
        workflow = PraisonAIAgents(
            agents=[news_aggregator, news_analyzer, market_analyzer, sentiment_analyzer, investor_insights, report_generator],
            tasks=[news_aggregation_task, news_processing_task, market_analysis_task, 
                   sentiment_analysis_task, investor_insights_task, report_generation_task],
            process="workflow",
            verbose=True
        )
        
        print("\nStarting Market Analysis Workflow...")
        print("=" * 50)
        
        # Run the workflow and get results
        workflow.start()
        
        # Combine all reports
        print("\nCombining all reports...")
        combined_report = combine_reports()
        
        if combined_report:
            print(f"\nAnalysis complete! Full report available at: {combined_report}")
        else:
            print("\nAnalysis complete but failed to combine reports.")
            
    except Exception as e:
        print(f"Error in workflow execution: {str(e)}")
        # Generate error report
        error_content = f"""Error Report
Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Error Details:
{str(e)}

Stack Trace:
{str(e.__traceback__)}
"""
        filename = f"error_report_{datetime.now().strftime('%Y%m%d_%H%M')}.txt"
        error_file = save_text_report(error_content, filename)
        if error_file:
            print(f"Error report generated: {error_file}")
        else:
            print("Failed to generate error report")

if __name__ == "__main__":
    run_agentic_workflow()
    print("\nDone")

