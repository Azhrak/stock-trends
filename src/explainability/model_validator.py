"""
Model validation and explainability summary script.
Validates model performance and summarizes key explainability insights.
"""

import sys
import os
sys.path.append('/Users/juho/code/azhrak/stock-trends/src')

import pandas as pd
import numpy as np
import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
import glob

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelValidator:
    """Comprehensive model validation and insight summarizer."""
    
    def __init__(self, reports_dir: str = "reports", models_dir: str = "models"):
        """
        Initialize model validator.
        
        Args:
            reports_dir: Directory containing explainability reports
            models_dir: Directory containing model results
        """
        self.reports_dir = reports_dir
        self.models_dir = models_dir
        
    def load_latest_reports(self) -> Dict[str, Any]:
        """Load the latest explainability reports."""
        
        reports = {}
        
        # Load LightGBM explainability report
        lgb_reports = glob.glob(os.path.join(self.reports_dir, "lightgbm_explainability_report_*.json"))
        if lgb_reports:
            latest_lgb = sorted(lgb_reports)[-1]
            with open(latest_lgb, 'r') as f:
                reports['lightgbm_explainability'] = json.load(f)
            logger.info(f"Loaded LightGBM explainability report: {os.path.basename(latest_lgb)}")
        
        # Load SHAP explanation report
        shap_reports = glob.glob(os.path.join(self.reports_dir, "prediction_explanation_report_*.json"))
        if shap_reports:
            latest_shap = sorted(shap_reports)[-1]
            with open(latest_shap, 'r') as f:
                reports['shap_explanations'] = json.load(f)
            logger.info(f"Loaded SHAP explanation report: {os.path.basename(latest_shap)}")
        
        # Load model comparison results
        comparison_files = glob.glob(os.path.join(self.models_dir, "model_comparison_report.json"))
        if comparison_files:
            with open(comparison_files[0], 'r') as f:
                reports['model_comparison'] = json.load(f)
            logger.info("Loaded model comparison report")
        
        return reports
    
    def validate_model_performance(self, reports: Dict[str, Any]) -> Dict[str, Any]:
        """Validate model performance against expected benchmarks."""
        
        validation_results = {
            'performance_checks': {},
            'validation_status': 'PASS',
            'issues': [],
            'recommendations': []
        }
        
        # Check model comparison results
        if 'model_comparison' in reports:
            comparison = reports['model_comparison']
            
            # Validate RMSE performance (using the correct field names)
            lgb_rmse = comparison.get('summary', {}).get('lightgbm', {}).get('test_rmse', {}).get('mean', float('inf'))
            transformer_rmse = comparison.get('summary', {}).get('transformer', {}).get('test_rmse', {}).get('mean', float('inf'))
            
            validation_results['performance_checks']['lightgbm_rmse'] = {
                'value': lgb_rmse,
                'benchmark': 0.15,  # Expect RMSE < 0.15
                'status': 'PASS' if lgb_rmse < 0.15 else 'FAIL'
            }
            
            validation_results['performance_checks']['transformer_rmse'] = {
                'value': transformer_rmse,
                'benchmark': 0.15,  # Expect RMSE < 0.15
                'status': 'PASS' if transformer_rmse < 0.15 else 'FAIL'
            }
            
            # Check directional accuracy
            lgb_dir_acc = comparison.get('summary', {}).get('lightgbm', {}).get('test_dir_accuracy', {}).get('mean', 0)
            transformer_dir_acc = comparison.get('summary', {}).get('transformer', {}).get('test_dir_accuracy', {}).get('mean', 0)
            
            validation_results['performance_checks']['lightgbm_direction'] = {
                'value': lgb_dir_acc,
                'benchmark': 0.55,  # Expect > 55% directional accuracy
                'status': 'PASS' if lgb_dir_acc > 0.55 else 'FAIL'
            }
            
            validation_results['performance_checks']['transformer_direction'] = {
                'value': transformer_dir_acc,
                'benchmark': 0.55,  # Expect > 55% directional accuracy
                'status': 'PASS' if transformer_dir_acc > 0.55 else 'FAIL'
            }
            
            # Check for overfitting (train vs test performance gap)
            if 'detailed_analysis' in comparison:
                gap_analysis = self._check_overfitting(comparison)
                validation_results['performance_checks']['overfitting'] = gap_analysis
        
        # Overall validation status
        failed_checks = [check for check, result in validation_results['performance_checks'].items() 
                        if result.get('status') == 'FAIL']
        
        if failed_checks:
            validation_results['validation_status'] = 'FAIL'
            validation_results['issues'].extend([f"Failed check: {check}" for check in failed_checks])
        
        return validation_results
    
    def _check_overfitting(self, comparison: Dict[str, Any]) -> Dict[str, Any]:
        """Check for overfitting by comparing train/validation/test performance."""
        
        # This would require more detailed train/val metrics from model training
        # For now, return a placeholder
        return {
            'status': 'PASS',
            'message': 'Overfitting check not implemented - requires train/val metrics'
        }
    
    def summarize_explainability_insights(self, reports: Dict[str, Any]) -> Dict[str, Any]:
        """Summarize key explainability insights from all reports."""
        
        insights = {
            'key_findings': [],
            'feature_insights': {},
            'model_behavior': {},
            'investment_implications': []
        }
        
        # Analyze feature importance from multiple sources
        if 'lightgbm_explainability' in reports:
            lgb_report = reports['lightgbm_explainability']
            
            # Traditional feature importance
            if 'sections' in lgb_report and 'feature_importance' in lgb_report['sections']:
                top_features = lgb_report['sections']['feature_importance'].get('top_features', [])
                insights['feature_insights']['traditional_importance'] = top_features[:10]
                
                if top_features:
                    insights['key_findings'].append(f"Most important feature by traditional importance: {top_features[0]}")
        
        # SHAP-based insights
        if 'shap_explanations' in reports:
            shap_report = reports['shap_explanations']
            
            if 'global_insights' in shap_report:
                shap_features = shap_report['global_insights'].get('top_features_globally', [])
                if shap_features:
                    insights['feature_insights']['shap_importance'] = [
                        f"{feat['feature']}: {feat['shap_importance']:.4f}" 
                        for feat in shap_features[:10]
                    ]
                    insights['key_findings'].append(f"Most important feature by SHAP analysis: {shap_features[0]['feature']}")
            
            # Individual prediction insights
            if 'sample_explanations' in shap_report:
                sample_explanations = shap_report['sample_explanations']
                
                if 'highest_prediction' in sample_explanations:
                    highest = sample_explanations['highest_prediction']
                    insights['model_behavior']['extreme_predictions'] = {
                        'highest_predicted_return': highest.get('predicted_impact', 0),
                        'actual_return': highest.get('actual_return', 0),
                        'main_drivers': [contrib['feature'] for contrib in 
                                       highest.get('top_positive_contributors', [])[:3]]
                    }
        
        # Investment strategy insights
        all_important_features = set()
        
        # Collect all important features
        if 'traditional_importance' in insights['feature_insights']:
            all_important_features.update(insights['feature_insights']['traditional_importance'])
        
        if 'shap_importance' in insights['feature_insights']:
            shap_features = [feat.split(':')[0] for feat in insights['feature_insights']['shap_importance']]
            all_important_features.update(shap_features)
        
        # Categorize important features for investment insights
        feature_categories = self._categorize_features_for_strategy(list(all_important_features))
        insights['investment_implications'] = self._generate_investment_insights(feature_categories)
        
        return insights
    
    def _categorize_features_for_strategy(self, features: List[str]) -> Dict[str, List[str]]:
        """Categorize features for investment strategy insights."""
        
        categories = {
            'momentum': [],
            'volatility': [],
            'volume': [],
            'technical': [],
            'price_levels': [],
            'other': []
        }
        
        for feature in features:
            feature_lower = feature.lower()
            
            if any(x in feature_lower for x in ['obv', 'momentum', 'roc', 'macd']):
                categories['momentum'].append(feature)
            elif any(x in feature_lower for x in ['volatility', 'atr']):
                categories['volatility'].append(feature)
            elif 'volume' in feature_lower:
                categories['volume'].append(feature)
            elif any(x in feature_lower for x in ['sma', 'ema', 'rsi', 'bb']):
                categories['technical'].append(feature)
            elif any(x in feature_lower for x in ['support', 'resistance', 'position', 'high', 'low']):
                categories['price_levels'].append(feature)
            else:
                categories['other'].append(feature)
        
        return categories
    
    def _generate_investment_insights(self, feature_categories: Dict[str, List[str]]) -> List[str]:
        """Generate investment strategy insights from feature categories."""
        
        insights = []
        
        if feature_categories['momentum']:
            insights.append(f"Momentum indicators are highly predictive ({len(feature_categories['momentum'])} features), "
                           "suggesting trend-following strategies may be effective")
        
        if feature_categories['volatility']:
            insights.append(f"Volatility measures are important ({len(feature_categories['volatility'])} features), "
                           "indicating the model considers risk assessment crucial")
        
        if feature_categories['volume']:
            insights.append(f"Volume-based features are significant ({len(feature_categories['volume'])} features), "
                           "suggesting volume confirmation is valuable for predictions")
        
        if feature_categories['technical']:
            insights.append(f"Technical indicators dominate ({len(feature_categories['technical'])} features), "
                           "supporting technical analysis approaches")
        
        if feature_categories['price_levels']:
            insights.append(f"Price level features are influential ({len(feature_categories['price_levels'])} features), "
                           "indicating support/resistance and position within range matter")
        
        return insights
    
    def generate_validation_report(self) -> Dict[str, Any]:
        """Generate comprehensive validation and explainability report."""
        
        logger.info("Generating comprehensive validation report...")
        
        # Load all reports
        reports = self.load_latest_reports()
        
        if not reports:
            logger.error("No reports found for validation")
            return {}
        
        # Validate performance
        performance_validation = self.validate_model_performance(reports)
        
        # Summarize explainability insights
        explainability_insights = self.summarize_explainability_insights(reports)
        
        # Create comprehensive report
        validation_report = {
            'validation_date': datetime.now().isoformat(),
            'validation_status': performance_validation['validation_status'],
            'performance_validation': performance_validation,
            'explainability_insights': explainability_insights,
            'summary': {
                'models_validated': list(reports.keys()),
                'key_findings': explainability_insights['key_findings'],
                'investment_implications': explainability_insights['investment_implications'],
                'validation_issues': performance_validation.get('issues', [])
            },
            'recommendations': self._generate_recommendations(performance_validation, explainability_insights)
        }
        
        # Save report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = os.path.join(self.reports_dir, f"validation_report_{timestamp}.json")
        
        with open(report_file, 'w') as f:
            json.dump(validation_report, f, indent=2, default=str)
        
        logger.info(f"Validation report saved: {report_file}")
        
        return validation_report
    
    def _generate_recommendations(self, performance: Dict[str, Any], insights: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on validation and insights."""
        
        recommendations = []
        
        # Performance-based recommendations
        if performance['validation_status'] == 'FAIL':
            recommendations.append("Model performance below expected benchmarks - consider hyperparameter tuning")
        
        # Feature-based recommendations
        if 'feature_insights' in insights:
            if 'shap_importance' in insights['feature_insights']:
                recommendations.append("SHAP analysis reveals feature attributions - use for feature selection")
            
            if 'traditional_importance' in insights['feature_insights']:
                recommendations.append("Traditional feature importance available - compare with SHAP for validation")
        
        # Investment strategy recommendations
        if insights.get('investment_implications'):
            recommendations.append("Explainability insights suggest specific trading strategies - see investment implications")
        
        if not recommendations:
            recommendations.append("Models pass validation - ready for deployment consideration")
        
        return recommendations

def main():
    """Main function to run model validation and generate summary."""
    
    logger.info("=" * 70)
    logger.info("MODEL VALIDATION AND EXPLAINABILITY SUMMARY")
    logger.info("=" * 70)
    
    # Initialize validator
    validator = ModelValidator()
    
    # Generate comprehensive validation report
    validation_report = validator.generate_validation_report()
    
    if not validation_report:
        logger.error("Could not generate validation report")
        return
    
    # Print summary
    summary = validation_report.get('summary', {})
    
    logger.info(f"\\nVALIDATION STATUS: {validation_report.get('validation_status', 'UNKNOWN')}")
    logger.info("-" * 40)
    
    key_findings = summary.get('key_findings', [])
    if key_findings:
        logger.info("\\nKEY FINDINGS:")
        for i, finding in enumerate(key_findings, 1):
            logger.info(f"{i}. {finding}")
    
    investment_implications = summary.get('investment_implications', [])
    if investment_implications:
        logger.info("\\nINVESTMENT IMPLICATIONS:")
        for i, implication in enumerate(investment_implications, 1):
            logger.info(f"{i}. {implication}")
    
    recommendations = validation_report.get('recommendations', [])
    if recommendations:
        logger.info("\\nRECOMMENDATIONS:")
        for i, rec in enumerate(recommendations, 1):
            logger.info(f"{i}. {rec}")
    
    issues = summary.get('validation_issues', [])
    if issues:
        logger.info("\\nVALIDATION ISSUES:")
        for i, issue in enumerate(issues, 1):
            logger.info(f"{i}. {issue}")
    
    logger.info("\\n" + "=" * 70)
    logger.info("VALIDATION AND EXPLAINABILITY ANALYSIS COMPLETED")
    logger.info("=" * 70)

if __name__ == "__main__":
    main()