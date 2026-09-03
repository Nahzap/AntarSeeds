"""
Visualize batch evaluation results with charts and tables.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse

def create_visualizations(csv_path, output_dir='batch_evaluation_results'):
    """Create visualizations from batch evaluation results"""
    
    # Load results
    df = pd.read_csv(csv_path)
    output_path = Path(output_dir)
    
    print("="*70)
    print("BATCH EVALUATION RESULTS VISUALIZATION")
    print("="*70)
    
    # 1. Overall Statistics
    print(f"\nTotal Images Evaluated: {len(df)}")
    print(f"Overall Accuracy: {df['correct'].mean():.2%}")
    print(f"Average Confidence: {df['confidence'].mean():.2%}")
    
    # 2. Per-Class Summary Table
    print("\n" + "="*70)
    print("PER-CLASS SUMMARY")
    print("="*70)
    
    summary = df.groupby('ground_truth').agg({
        'correct': ['sum', 'count', 'mean'],
        'confidence': 'mean',
        'mean_distance': 'mean'
    }).round(4)
    
    summary.columns = ['Correct', 'Total', 'Accuracy', 'Avg_Confidence', 'Avg_Distance']
    summary['Accuracy'] = summary['Accuracy'].apply(lambda x: f"{x:.2%}")
    summary['Avg_Confidence'] = summary['Avg_Confidence'].apply(lambda x: f"{x:.2%}")
    
    print(summary)
    
    # 3. Confusion Matrix Heatmap
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Confusion matrix (counts)
    confusion = pd.crosstab(df['ground_truth'], df['prediction'])
    sns.heatmap(confusion, annot=True, fmt='d', cmap='Blues', ax=axes[0], cbar_kws={'label': 'Count'})
    axes[0].set_title('Confusion Matrix (Counts)', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Ground Truth', fontsize=12)
    axes[0].set_xlabel('Prediction', fontsize=12)
    
    # Confusion matrix (normalized)
    confusion_norm = confusion.div(confusion.sum(axis=1), axis=0)
    sns.heatmap(confusion_norm, annot=True, fmt='.2%', cmap='Greens', ax=axes[1], cbar_kws={'label': 'Proportion'})
    axes[1].set_title('Confusion Matrix (Normalized)', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('Ground Truth', fontsize=12)
    axes[1].set_xlabel('Prediction', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(output_path / 'confusion_matrix_visualization.png', dpi=300, bbox_inches='tight')
    print(f"\nConfusion matrix visualization saved to: {output_path / 'confusion_matrix_visualization.png'}")
    plt.close()
    
    # 4. Accuracy Bar Chart
    fig, ax = plt.subplots(figsize=(10, 6))
    
    class_acc = df.groupby('ground_truth')['correct'].mean() * 100
    bars = ax.bar(range(len(class_acc)), class_acc.values, color=['#2ecc71', '#3498db', '#e74c3c'])
    ax.set_xticks(range(len(class_acc)))
    ax.set_xticklabels(class_acc.index, rotation=45, ha='right')
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Per-Class Accuracy', fontsize=14, fontweight='bold')
    ax.set_ylim([95, 101])
    ax.axhline(y=100, color='gray', linestyle='--', alpha=0.5, label='100%')
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, class_acc.values)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.2f}%',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path / 'accuracy_by_class.png', dpi=300, bbox_inches='tight')
    print(f"Accuracy chart saved to: {output_path / 'accuracy_by_class.png'}")
    plt.close()
    
    # 5. Confidence Distribution
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Overall confidence distribution
    axes[0].hist(df['confidence'], bins=20, color='skyblue', edgecolor='black', alpha=0.7)
    axes[0].set_xlabel('Confidence', fontsize=12)
    axes[0].set_ylabel('Frequency', fontsize=12)
    axes[0].set_title('Overall Confidence Distribution', fontsize=14, fontweight='bold')
    axes[0].axvline(df['confidence'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {df["confidence"].mean():.2%}')
    axes[0].legend()
    axes[0].grid(axis='y', alpha=0.3)
    
    # Confidence by correctness
    correct_conf = df[df['correct']]['confidence']
    incorrect_conf = df[~df['correct']]['confidence']
    
    axes[1].hist([correct_conf, incorrect_conf], bins=20, label=['Correct', 'Incorrect'], 
                 color=['green', 'red'], alpha=0.6, edgecolor='black')
    axes[1].set_xlabel('Confidence', fontsize=12)
    axes[1].set_ylabel('Frequency', fontsize=12)
    axes[1].set_title('Confidence Distribution by Correctness', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path / 'confidence_distribution.png', dpi=300, bbox_inches='tight')
    print(f"Confidence distribution saved to: {output_path / 'confidence_distribution.png'}")
    plt.close()
    
    # 6. Sample of Results Table (first 20 rows)
    print("\n" + "="*70)
    print("SAMPLE RESULTS (First 20 images)")
    print("="*70)
    
    sample_df = df[['image_name', 'ground_truth', 'prediction', 'confidence', 'correct']].head(20)
    sample_df['confidence'] = sample_df['confidence'].apply(lambda x: f"{x:.2%}")
    sample_df['correct'] = sample_df['correct'].apply(lambda x: '✓' if x else '✗')
    
    print(sample_df.to_string(index=False))
    
    # 7. Misclassified Images
    if (~df['correct']).any():
        print("\n" + "="*70)
        print("MISCLASSIFIED IMAGES")
        print("="*70)
        
        misclassified = df[~df['correct']][['image_name', 'ground_truth', 'prediction', 'confidence', 'mean_distance']]
        misclassified['confidence'] = misclassified['confidence'].apply(lambda x: f"{x:.2%}")
        misclassified['mean_distance'] = misclassified['mean_distance'].apply(lambda x: f"{x:.4f}")
        
        print(f"\nTotal Misclassified: {len(misclassified)}")
        print(misclassified.to_string(index=False))
    else:
        print("\n" + "="*70)
        print("NO MISCLASSIFIED IMAGES - PERFECT ACCURACY!")
        print("="*70)
    
    print("\n" + "="*70)
    print("VISUALIZATION COMPLETE")
    print("="*70)


def main():
    parser = argparse.ArgumentParser(description='Visualize Batch Evaluation Results')
    parser.add_argument('--csv', type=str, required=True,
                        help='Path to detailed results CSV file')
    parser.add_argument('--output_dir', type=str, default='batch_evaluation_results',
                        help='Directory to save visualizations')
    args = parser.parse_args()
    
    create_visualizations(args.csv, args.output_dir)


if __name__ == '__main__':
    main()
