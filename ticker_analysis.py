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


def print_ticker_analysis(analysis: Dict[str, Any], detailed: bool = False):
    """Print formatted ticker analysis."""
    ticker = analysis['ticker']
    
    print(f"\n{'='*60}")
    print(f"ANALYSIS FOR {ticker}")
    print(f"{'='*60}")
    
    # Print ticker statistics
    if analysis['ticker_stats']:
        stats = analysis['ticker_stats']
        print(f"\nTICKER STATISTICS:")
        print(f"  Sample count: {stats['sample_count']}")
        print(f"  Average prediction: {stats['avg_prediction']:.4f}")
        print(f"  Average actual return: {stats['avg_actual']:.4f}")
        print(f"  Prediction std dev: {stats['prediction_std']:.4f}")
        
        print(f"\n  TOP 5 FEATURES FOR {ticker}:")
        for i, feature in enumerate(stats['top_features'], 1):
            print(f"    {i}. {feature['feature']:<20} importance: {feature['importance']:.4f}")
    
    # Print specific examples
    if analysis['examples']:
        print(f"\nSPECIFIC PREDICTION EXAMPLES FOR {ticker}:")
        for example in analysis['examples']:
            data = example['data']
            print(f"\n  {example['type'].upper()}:")
            print(f"    Date: {data.get('date', 'N/A')}")
            print(f"    Predicted impact: {data.get('predicted_impact', 0):.4f}")
            print(f"    Actual return: {data.get('actual_return', 0):.4f}")
            
            if detailed and 'top_positive_contributors' in data:
                print(f"    Top positive contributors:")
                for contrib in data['top_positive_contributors'][:3]:
                    print(f"      {contrib['feature']:<20} SHAP: {contrib['shap_value']:+.4f}")
                    
                print(f"    Top negative contributors:")
                for contrib in data['top_negative_contributors'][:3]:
                    print(f"      {contrib['feature']:<20} SHAP: {contrib['shap_value']:+.4f}")


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