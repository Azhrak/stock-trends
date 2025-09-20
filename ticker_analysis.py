#!/usr/bin/env python3
"""
Ticker-specific analysis tool for SHAP explainability reports.
Allows you to filter and analyze predictions for specific stocks like AAPL.
"""

import json
import argparse
import pandas as pd
from typing import Dict, List, Any


def load_latest_report(reports_dir: str = "reports") -> Dict[str, Any]:
    """Load the most recent prediction explanation report."""
    import os
    import glob
    
    pattern = os.path.join(reports_dir, "prediction_explanation_report_*.json")
    report_files = glob.glob(pattern)
    
    if not report_files:
        raise FileNotFoundError(f"No prediction explanation reports found in {reports_dir}")
    
    # Get the most recent report
    latest_report = max(report_files, key=os.path.getmtime)
    print(f"Loading report: {latest_report}")
    
    with open(latest_report, 'r') as f:
        return json.load(f)


def analyze_ticker(report: Dict[str, Any], ticker: str) -> Dict[str, Any]:
    """Extract all information related to a specific ticker."""
    ticker = ticker.upper()
    results = {
        'ticker': ticker,
        'examples': [],
        'ticker_stats': None,
        'feature_insights': []
    }
    
    # Check sample explanations for this ticker
    sample_explanations = report.get('sample_explanations', {})
    
    for example_type, example_data in sample_explanations.items():
        if example_type == 'ticker_analysis':
            # Get ticker-specific statistics
            ticker_analysis = example_data.get(ticker, {})
            if ticker_analysis:
                results['ticker_stats'] = ticker_analysis
        elif isinstance(example_data, dict) and example_data.get('ticker') == ticker:
            # This is a specific prediction example for our ticker
            results['examples'].append({
                'type': example_type,
                'data': example_data
            })
    
    return results


def get_ticker_feature_importance(report: Dict[str, Any], ticker: str) -> List[Dict[str, Any]]:
    """Get feature importance specific to a ticker from ticker analysis."""
    ticker = ticker.upper()
    
    sample_explanations = report.get('sample_explanations', {})
    ticker_analysis = sample_explanations.get('ticker_analysis', {})
    ticker_data = ticker_analysis.get(ticker, {})
    
    return ticker_data.get('top_features', [])


def explain_prediction_value(prediction: float) -> str:
    """Explain what a prediction value means in investment terms."""
    percentage = prediction * 100
    if percentage > 10:
        return f"Strong Buy signal ({percentage:.1f}% expected return over 12 weeks)"
    elif percentage > 5:
        return f"Buy signal ({percentage:.1f}% expected return over 12 weeks)"
    elif percentage > 2:
        return f"Moderate Buy ({percentage:.1f}% expected return over 12 weeks)"
    elif percentage > -2:
        return f"Neutral/Hold ({percentage:.1f}% expected return over 12 weeks)"
    elif percentage > -5:
        return f"Moderate Sell ({percentage:.1f}% expected loss over 12 weeks)"
    else:
        return f"Sell signal ({percentage:.1f}% expected loss over 12 weeks)"


def explain_feature(feature_name: str) -> str:
    """Explain what each feature means in plain English."""
    explanations = {
        'obv': "On-Balance Volume (buying/selling pressure from volume)",
        'volatility_12w': "12-week price volatility (how much the stock price moves)",
        'price_position_26w': "Price position within 26-week range (near highs/lows)",
        'price_sma_26_ratio': "Price vs 26-week moving average (above/below trend)",
        'atr_pct_12w': "Average True Range (daily volatility measure)",
        'support_52w': "Distance from 52-week support level",
        'resistance_52w': "Distance from 52-week resistance level",
        'macd_signal': "MACD momentum indicator signal",
        'rsi': "Relative Strength Index (overbought/oversold)",
        'volume_sma_26': "Volume vs 26-week average (high/low trading activity)",
        'price_sma_52_ratio': "Price vs 52-week moving average (long-term trend)",
        'price_ema_26_ratio': "Price vs 26-week exponential moving average"
    }
    return explanations.get(feature_name, f"{feature_name} (technical indicator)")


def explain_model_confidence(std_dev: float) -> str:
    """Explain what the prediction standard deviation means."""
    if std_dev < 0.003:
        return "Very consistent predictions (high confidence)"
    elif std_dev < 0.006:
        return "Moderately consistent predictions (medium confidence)"
    else:
        return "Variable predictions (lower confidence, stock harder to predict)"


def compare_prediction_vs_actual(prediction: float, actual: float) -> str:
    """Compare predicted vs actual returns and explain the accuracy."""
    pred_pct = prediction * 100
    actual_pct = actual * 100
    diff = abs(pred_pct - actual_pct)
    
    if diff < 1:
        return f"✅ Very accurate (predicted {pred_pct:.1f}%, actual {actual_pct:.1f}%)"
    elif diff < 3:
        return f"✅ Good accuracy (predicted {pred_pct:.1f}%, actual {actual_pct:.1f}%)"
    elif diff < 5:
        return f"⚠️ Moderate accuracy (predicted {pred_pct:.1f}%, actual {actual_pct:.1f}%)"
    else:
        return f"❌ Low accuracy (predicted {pred_pct:.1f}%, actual {actual_pct:.1f}%)"


def print_ticker_analysis(analysis: Dict[str, Any], detailed: bool = False):
    """Print formatted ticker analysis with clear explanations."""
    ticker = analysis['ticker']
    
    print(f"\n{'='*60}")
    print(f"INVESTMENT ANALYSIS FOR {ticker}")
    print(f"{'='*60}")
    
    # Print ticker statistics with explanations
    if analysis['ticker_stats']:
        stats = analysis['ticker_stats']
        avg_pred = stats['avg_prediction']
        avg_actual = stats['avg_actual']
        std_dev = stats['prediction_std']
        
    print(f"\n📊 MODEL PREDICTION SUMMARY:")
    print(f"  {explain_prediction_value(avg_pred)}")
    print(f"  Based on {stats['sample_count']} predictions over the test period")
    
    # Display date range if available
    if 'date_range_start' in stats and 'date_range_end' in stats:
        date_start = stats['date_range_start']
        date_end = stats['date_range_end']
        print(f"  Analysis period: {date_start} to {date_end}")
        
        # Calculate approximate weeks
        try:
            from datetime import datetime
            start_date = datetime.strptime(date_start, '%Y-%m-%d')
            end_date = datetime.strptime(date_end, '%Y-%m-%d')
            weeks = (end_date - start_date).days // 7
            print(f"  Period length: ~{weeks} weeks")
        except:
            print(f"  Period length: ~12 weeks (estimated)")
    else:
        print(f"  Analysis period: Past 12 weeks (estimated)")
    
    print(f"  {explain_model_confidence(std_dev)}")
    
    print(f"\n📈 ACTUAL PERFORMANCE:")
    print(f"  {compare_prediction_vs_actual(avg_pred, avg_actual)}")
        
    print(f"\n🔍 WHAT DRIVES {ticker} PREDICTIONS:")
    print(f"  The model focuses on these key factors:")
    for i, feature in enumerate(stats['top_features'][:5], 1):
        feature_name = feature['feature']
        importance = feature['importance']
        explanation = explain_feature(feature_name)
        print(f"    {i}. {explanation}")
        if importance > 0.004:
            print(f"       ⭐ Very important for {ticker} predictions")
        elif importance > 0.002:
            print(f"       ⚠️ Moderately important for {ticker} predictions")
        else:
            print(f"       📍 Minor factor for {ticker} predictions")
    
    print(f"\n💡 INVESTMENT INSIGHTS:")
    insights = generate_investment_insights(ticker, stats, avg_pred, avg_actual)
    for insight in insights:
        print(f"  • {insight}")
    
    # Print specific examples with clear explanations
    if analysis['examples']:
        print(f"\n📋 SPECIFIC PREDICTION EXAMPLES:")
        for example in analysis['examples']:
            data = example['data']
            pred = data.get('predicted_impact', 0)
            actual = data.get('actual_return', 0)
            date = data.get('date', 'N/A')
            
            print(f"\n  {example['type'].replace('_', ' ').upper()}:")
            print(f"    Date: {date}")
            print(f"    {explain_prediction_value(pred)}")
            print(f"    {compare_prediction_vs_actual(pred, actual)}")
            
            if detailed and 'top_positive_contributors' in data:
                print(f"    🔼 Why the model was bullish:")
                for contrib in data['top_positive_contributors'][:3]:
                    feature_name = contrib['feature']
                    shap_val = contrib['shap_value']
                    impact_pct = shap_val * 100
                    explanation = explain_feature(feature_name)
                    print(f"      • {explanation}: +{impact_pct:.1f}% impact")
                    
                print(f"    🔽 Why the model was bearish:")
                for contrib in data['top_negative_contributors'][:3]:
                    feature_name = contrib['feature']
                    shap_val = contrib['shap_value']
                    impact_pct = abs(shap_val) * 100
                    explanation = explain_feature(feature_name)
                    print(f"      • {explanation}: -{impact_pct:.1f}% impact")


def generate_investment_insights(ticker: str, stats: Dict, avg_pred: float, avg_actual: float) -> List[str]:
    """Generate actionable investment insights based on the analysis."""
    insights = []
    
    # Overall model accuracy insight
    accuracy_diff = abs(avg_pred - avg_actual)
    if accuracy_diff < 0.01:
        insights.append(f"Model is very accurate for {ticker} - predictions are reliable")
    elif accuracy_diff < 0.03:
        insights.append(f"Model has good accuracy for {ticker} - use with confidence")
    else:
        insights.append(f"Model has mixed accuracy for {ticker} - use predictions cautiously")
    
    # Prediction magnitude insight
    if avg_pred > 0.05:
        insights.append(f"{ticker} shows strong positive momentum signals")
    elif avg_pred > 0.02:
        insights.append(f"{ticker} shows moderate positive momentum")
    elif avg_pred < -0.02:
        insights.append(f"{ticker} shows concerning negative momentum")
    else:
        insights.append(f"{ticker} shows neutral momentum - sideways movement expected")
    
    # Top feature insights
    top_features = [f['feature'] for f in stats['top_features'][:3]]
    if 'obv' in top_features:
        insights.append("Volume analysis is crucial - watch buying/selling pressure")
    if any('volatility' in f for f in top_features):
        insights.append("Price volatility is a key factor - expect price swings")
    if any('position' in f for f in top_features):
        insights.append("Price positioning matters - watch support/resistance levels")
    if any('sma' in f or 'ema' in f for f in top_features):
        insights.append("Moving averages are important - trend following strategy recommended")
    
    return insights


def list_available_tickers(report: Dict[str, Any]) -> List[str]:
    """List all available tickers in the report."""
    sample_explanations = report.get('sample_explanations', {})
    ticker_analysis = sample_explanations.get('ticker_analysis', {})
    return sorted(ticker_analysis.keys())


def main():
    parser = argparse.ArgumentParser(description='Analyze SHAP explainability for specific tickers')
    parser.add_argument('ticker', nargs='?', help='Ticker symbol to analyze (e.g., AAPL)')
    parser.add_argument('--list', action='store_true', help='List all available tickers')
    parser.add_argument('--detailed', action='store_true', help='Show detailed feature contributions')
    parser.add_argument('--reports-dir', default='reports', help='Directory containing reports')
    
    args = parser.parse_args()
    
    try:
        # Load the latest report
        report = load_latest_report(args.reports_dir)
        
        if args.list:
            tickers = list_available_tickers(report)
            print(f"\nAvailable tickers in the report:")
            for ticker in tickers:
                stats = report['sample_explanations']['ticker_analysis'][ticker]
                print(f"  {ticker}: {stats['sample_count']} samples")
            return
        
        if not args.ticker:
            print("Please specify a ticker to analyze, or use --list to see available tickers")
            print("Example: python ticker_analysis.py AAPL")
            return
        
        # Analyze the specified ticker
        analysis = analyze_ticker(report, args.ticker)
        
        if not analysis['ticker_stats'] and not analysis['examples']:
            print(f"No data found for ticker {args.ticker.upper()}")
            print("Use --list to see available tickers")
            return
        
        print_ticker_analysis(analysis, detailed=args.detailed)
        
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()