import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from typing import List, Dict
from pathlib import Path

# Setup paths
SCRIPT_DIR = Path(__file__).parent
ROOT_DIR = SCRIPT_DIR.parent
VISUALIZATIONS_DIR = SCRIPT_DIR / 'visualizations'
DATA_FILE = ROOT_DIR / '3_LLMs_Responses_Metrics_Calculation' / 'processed_results.csv'

# Load data and setup
df = pd.read_csv(DATA_FILE)
os.makedirs(VISUALIZATIONS_DIR, exist_ok=True)

# Create consistent color mapping for models
MODEL_COLORS = {
    'aya': '#1f77b4',  # blue
    'gemma': '#ff7f0e',  # orange
    'llama3.2': '#2ca02c',  # green
    'mistral': '#d62728',  # red
    'mixtral': '#9467bd',  # purple
    'phi': '#8c564b',  # brown
    'qwen': '#e377c2',  # pink
    'deepseek-r1': '#7f7f7f',  # gray
    'falcon3': '#bcbd22'  # olive
}

def setup_style():
    """Set up the plotting style for all visualizations"""
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['axes.labelsize'] = 12

def save_plot(fig, filename: str):
    """Save the plot with consistent settings"""
    plt.savefig(VISUALIZATIONS_DIR / f'{filename}.png', bbox_inches='tight', dpi=300)
    plt.close()

def create_category_performance():
    """Create heatmaps for both English and Arabic performance by category"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # English heatmap
    en_heatmap = df.pivot_table(index='Model', columns='Category', values='EN_Response Accuracy Rate')
    sns.heatmap(en_heatmap, annot=True, cmap='YlGnBu', fmt='.1f', linewidths=.5, 
                cbar_kws={'label': 'Accuracy Rate (%)'}, ax=ax1)
    ax1.set_title('English Performance by Category')
    
    # Arabic heatmap
    ar_heatmap = df.pivot_table(index='Model', columns='Category', values='AR_Response Accuracy Rate')
    sns.heatmap(ar_heatmap, annot=True, cmap='YlGnBu', fmt='.1f', linewidths=.5, 
                cbar_kws={'label': 'Accuracy Rate (%)'}, ax=ax2)
    ax2.set_title('Arabic Performance by Category')
    
    plt.tight_layout()
    save_plot(fig, 'category_performance')

def create_language_comparison():
    """Create comprehensive language comparison visualizations"""
    # Direct comparison bar chart
    lang_comp = df.groupby('Model')[['EN_Response Accuracy Rate', 'AR_Response Accuracy Rate']].mean()
    plt.figure(figsize=(12, 6))
    x = np.arange(len(lang_comp))
    width = 0.35
    plt.bar(x - width/2, lang_comp['EN_Response Accuracy Rate'], width, label='English', color='royalblue')
    plt.bar(x + width/2, lang_comp['AR_Response Accuracy Rate'], width, label='Arabic', color='tomato')
    plt.xlabel('Model')
    plt.ylabel('Accuracy Rate (%)')
    plt.title('English vs Arabic Performance Comparison')
    plt.xticks(x, lang_comp.index, rotation=45)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.tight_layout()
    save_plot(plt.gcf(), 'language_comparison')

def create_performance_distribution():
    """Create distribution plots for both languages"""
    plt.figure(figsize=(12, 6))
    data_to_plot = df[['EN_Response Accuracy Rate', 'AR_Response Accuracy Rate']].melt()
    sns.boxplot(x='variable', y='value', data=data_to_plot)
    plt.title('Distribution of Performance Scores')
    plt.xlabel('Language')
    plt.ylabel('Accuracy Rate (%)')
    plt.xticks([0, 1], ['English', 'Arabic'])
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.tight_layout()
    save_plot(plt.gcf(), 'performance_distribution')

def create_model_consistency():
    """Create consistency analysis for both languages"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 6))
    
    # English consistency
    en_var = df.groupby('Model')['EN_Response Accuracy Rate'].agg(['mean', 'std']).reset_index()
    en_var['coefficient_of_variation'] = (en_var['std'] / en_var['mean']) * 100
    en_var = en_var.sort_values('coefficient_of_variation')
    
    sns.barplot(x='Model', y='coefficient_of_variation', data=en_var, 
                palette=MODEL_COLORS, ax=ax1)
    ax1.set_title('English Model Consistency\n(Lower values indicate more consistent performance)')
    ax1.set_ylabel('Coefficient of Variation (%)')
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(axis='y', linestyle='--', alpha=0.3)
    
    # Arabic consistency
    ar_var = df.groupby('Model')['AR_Response Accuracy Rate'].agg(['mean', 'std']).reset_index()
    ar_var['coefficient_of_variation'] = (ar_var['std'] / ar_var['mean']) * 100
    ar_var = ar_var.sort_values('coefficient_of_variation')
    
    sns.barplot(x='Model', y='coefficient_of_variation', data=ar_var, 
                palette=MODEL_COLORS, ax=ax2)
    ax2.set_title('Arabic Model Consistency\n(Lower values indicate more consistent performance)')
    ax2.set_ylabel('Coefficient of Variation (%)')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(axis='y', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    save_plot(fig, 'model_consistency')

def create_correlation_analysis():
    """Create correlation analysis for both languages"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # English correlations
    en_metrics = [col for col in df.columns if col.startswith('EN_') and any(x in col for x in 
                ['Accuracy', 'Sensitivity', 'Quality', 'Relevance'])]
    mask_en = np.triu(np.ones_like(df[en_metrics].corr(), dtype=bool))
    sns.heatmap(df[en_metrics].corr(), mask=mask_en, cmap='coolwarm', center=0,
                annot=True, fmt='.2f', square=True, linewidths=.5, ax=ax1)
    ax1.set_title('English Metrics Correlation')
    
    # Arabic correlations
    ar_metrics = [col for col in df.columns if col.startswith('AR_') and any(x in col for x in 
                ['Accuracy', 'Sensitivity', 'Quality', 'Relevance'])]
    mask_ar = np.triu(np.ones_like(df[ar_metrics].corr(), dtype=bool))
    sns.heatmap(df[ar_metrics].corr(), mask=mask_ar, cmap='coolwarm', center=0,
                annot=True, fmt='.2f', square=True, linewidths=.5, ax=ax2)
    ax2.set_title('Arabic Metrics Correlation')
    
    plt.tight_layout()
    save_plot(fig, 'correlation_analysis')

def create_cultural_sensitivity_analysis():
    """Create analysis of cultural sensitivity scores"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 6))
    
    # English cultural sensitivity
    en_sens = df.groupby('Model')['EN_Cultural_Sensitivity'].mean().sort_values(ascending=False)
    sns.barplot(x=en_sens.index, y=en_sens.values, palette=MODEL_COLORS, ax=ax1)
    ax1.set_title('English Cultural Sensitivity Scores')
    ax1.set_ylabel('Cultural Sensitivity Score')
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(axis='y', linestyle='--', alpha=0.3)
    
    # Arabic cultural sensitivity
    ar_sens = df.groupby('Model')['AR_Cultural_Sensitivity'].mean().sort_values(ascending=False)
    sns.barplot(x=ar_sens.index, y=ar_sens.values, palette=MODEL_COLORS, ax=ax2)
    ax2.set_title('Arabic Cultural Sensitivity Scores')
    ax2.set_ylabel('Cultural Sensitivity Score')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(axis='y', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    save_plot(fig, 'cultural_sensitivity_analysis')

def create_language_quality_analysis():
    """Create analysis of language quality scores"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 6))
    
    # English language quality
    en_qual = df.groupby('Model')['EN_Language_Quality'].mean().sort_values(ascending=False)
    sns.barplot(x=en_qual.index, y=en_qual.values, palette=MODEL_COLORS, ax=ax1)
    ax1.set_title('English Language Quality Scores')
    ax1.set_ylabel('Language Quality Score')
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(axis='y', linestyle='--', alpha=0.3)
    
    # Arabic language quality
    ar_qual = df.groupby('Model')['AR_Language_Quality'].mean().sort_values(ascending=False)
    sns.barplot(x=ar_qual.index, y=ar_qual.values, palette=MODEL_COLORS, ax=ax2)
    ax2.set_title('Arabic Language Quality Scores')
    ax2.set_ylabel('Language Quality Score')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(axis='y', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    save_plot(fig, 'language_quality_analysis')

def create_language_disparity_analysis():
    """Create analysis of language disparity between English and Arabic responses"""
    plt.figure(figsize=(12, 6))
    disparity = df.groupby('Model')['Language Disparity'].mean().sort_values(ascending=False)
    sns.barplot(x=disparity.index, y=disparity.values, palette='viridis')
    plt.title('Language Disparity Between English and Arabic Responses')
    plt.ylabel('Disparity Score')
    plt.xlabel('Model')
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.tight_layout()
    save_plot(plt.gcf(), 'language_disparity_analysis')

def create_category_breakdown():
    """Create detailed breakdown of performance by category and model"""
    categories = df['Category'].unique()
    n_categories = len(categories)
    
    # Calculate grid dimensions
    n_cols = min(2, n_categories)
    n_rows = (n_categories + 1) // 2
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5*n_rows))
    if n_categories == 1:
        axes = np.array([axes])
    axes = axes.ravel()
    
    for idx, category in enumerate(categories):
        category_data = df[df['Category'] == category]
        en_means = category_data.groupby('Model')['EN_Response Accuracy Rate'].mean()
        ar_means = category_data.groupby('Model')['AR_Response Accuracy Rate'].mean()
        
        x = np.arange(len(en_means))
        width = 0.35
        
        axes[idx].bar(x - width/2, en_means, width, label='English', color='royalblue')
        axes[idx].bar(x + width/2, ar_means, width, label='Arabic', color='tomato')
        
        axes[idx].set_title(f'Performance in {category.title()} Category')
        axes[idx].set_ylabel('Accuracy Rate (%)')
        axes[idx].set_xticks(x)
        axes[idx].set_xticklabels(en_means.index, rotation=45)
        axes[idx].legend()
        axes[idx].grid(axis='y', linestyle='--', alpha=0.3)
    
    # Hide empty subplots if any
    for idx in range(n_categories, len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    save_plot(fig, 'category_breakdown')

def main():
    """Main function to create all visualizations"""
    print("Setting up visualization style...")
    setup_style()
    
    print("Creating category performance heatmaps...")
    create_category_performance()
    
    print("Creating language comparison...")
    create_language_comparison()
    
    print("Creating performance distribution...")
    create_performance_distribution()
    
    print("Creating model consistency analysis...")
    create_model_consistency()
    
    print("Creating correlation analysis...")
    create_correlation_analysis()
    
    print("Creating cultural sensitivity analysis...")
    create_cultural_sensitivity_analysis()
    
    print("Creating language quality analysis...")
    create_language_quality_analysis()
    
    print("Creating language disparity analysis...")
    create_language_disparity_analysis()
    
    print("Creating category breakdown...")
    create_category_breakdown()
    
    print("All visualizations complete! Images saved in the '4_LLMs_Responses_Analysis/visualizations' folder.")

if __name__ == "__main__":
    main()