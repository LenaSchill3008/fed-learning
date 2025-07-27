#!/usr/bin/env python3
"""
Compare federated and centralized ML results.
Analyzes performance differences and trade-offs.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional


class MLComparison:
    def __init__(self, result_dir: str = "results"):
        self.result_dir = Path(result_dir)
        self.fed_file = self.result_dir / "results.csv"
        self.cent_file = self.result_dir / "centralized_results.csv"
        self.comparison_file = self.result_dir / "comparison_results.csv"
        self.analysis_file = self.result_dir / "analysis_results.csv"
    
    def load_results(self) -> Optional[pd.DataFrame]:
        """Load and combine federated and centralized results."""
        if not self.fed_file.exists():
            print("Federated results not found. Run federated experiments first.")
            return None
        
        if not self.cent_file.exists():
            print("Centralized results not found. Run centralized experiments first.")
            return None
        
        # Load both result files
        fed_df = pd.read_csv(self.fed_file)
        cent_df = pd.read_csv(self.cent_file)
        
        # Add approach column
        fed_df['approach'] = 'federated'
        cent_df['approach'] = 'centralized'
        
        # Combine results
        combined_df = pd.concat([fed_df, cent_df], ignore_index=True)
        
        # Sort for better comparison
        combined_df = combined_df.sort_values(['model', 'dataset', 'approach'])
        
        return combined_df
    
    def calculate_differences(self, combined_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate performance differences between federated and centralized."""
        metrics = ['loss', 'accuracy', 'precision', 'recall', 'f1_score']
        
        analysis_results = []
        
        # Group by model and dataset
        for (model, dataset), group in combined_df.groupby(['model', 'dataset']):
            if len(group) != 2:
                continue
                
            fed_row = group[group['approach'] == 'federated'].iloc[0]
            cent_row = group[group['approach'] == 'centralized'].iloc[0]
            
            analysis = {
                'model': model,
                'dataset': dataset,
            }
            
            for metric in metrics:
                fed_val = fed_row[metric]
                cent_val = cent_row[metric]
                
                # Calculate absolute difference
                diff = fed_val - cent_val
                
                # Calculate percentage difference
                if cent_val != 0:
                    pct_diff = (diff / cent_val) * 100
                else:
                    pct_diff = 0
                
                analysis[f'{metric}_fed'] = fed_val
                analysis[f'{metric}_cent'] = cent_val
                analysis[f'{metric}_diff'] = diff
                analysis[f'{metric}_pct_diff'] = pct_diff
            
            analysis_results.append(analysis)
        
        return pd.DataFrame(analysis_results)
    
    def print_summary(self, combined_df: pd.DataFrame, analysis_df: pd.DataFrame):
        """Print comprehensive comparison summary."""
        print("\n" + "=" * 80)
        print("FEDERATED vs CENTRALIZED ML COMPARISON")
        print("=" * 80)
        
        # Side-by-side comparison
        print("\nSIDE-BY-SIDE RESULTS:")
        print("-" * 60)
        print(combined_df.to_string(index=False))
        
        print("\n\nPERFORMANCE ANALYSIS:")
        print("-" * 60)
        
        # Key insights for each model-dataset combination
        for _, row in analysis_df.iterrows():
            print(f"\n{row['model'].upper()} on {row['dataset'].upper()}:")
            
            # Accuracy comparison
            acc_diff = row['accuracy_pct_diff']
            if acc_diff > 0:
                print(f"  ✅ Federated WINS: +{acc_diff:.2f}% accuracy vs centralized")
            elif acc_diff < -2:
                print(f"  ❌ Federated LOSES: {acc_diff:.2f}% accuracy vs centralized")
            else:
                print(f"  ⚖️  Similar performance: {acc_diff:.2f}% accuracy difference")
            
            # F1 score comparison
            f1_diff = row['f1_score_pct_diff']
            print(f"  📊 F1-Score difference: {f1_diff:+.2f}%")
            
            # Loss comparison (lower is better)
            loss_diff = row['loss_pct_diff']
            if loss_diff < 0:
                print(f"  🎯 Federated has {abs(loss_diff):.2f}% lower loss")
            else:
                print(f"  🎯 Centralized has {abs(loss_diff):.2f}% lower loss")
            
    
    def save_results(self, combined_df: pd.DataFrame, analysis_df: pd.DataFrame):
        """Save comparison and analysis results."""
        # Save combined results
        combined_df.to_csv(self.comparison_file, index=False)
        print(f"\n💾 Combined results saved to: {self.comparison_file}")
        
        # Save analysis results
        analysis_df.to_csv(self.analysis_file, index=False)
        print(f"💾 Analysis results saved to: {self.analysis_file}")
    
    def run_comparison(self):
        """Run complete comparison analysis."""
        print("🔬 Starting ML approach comparison...")
        
        # Load results
        combined_df = self.load_results()
        if combined_df is None:
            return
        
        # Calculate differences
        analysis_df = self.calculate_differences(combined_df)
        
        # Print comprehensive summary
        self.print_summary(combined_df, analysis_df)
        
        # Save results
        self.save_results(combined_df, analysis_df)
        
        print(f"\n Comparison complete!")


def main():
    """Main function to run ML comparison."""
    comparison = MLComparison()
    comparison.run_comparison()


if __name__ == "__main__":
    main()