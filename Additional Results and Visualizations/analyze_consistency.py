import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
from pathlib import Path
import json

# Setup paths
SCRIPT_DIR = Path(__file__).parent
VISUALIZATIONS_DIR = SCRIPT_DIR / 'visualizations'
os.makedirs(VISUALIZATIONS_DIR, exist_ok=True)

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

def load_data():
    """Load and prepare the data"""
    # Load responses
    df = pd.read_csv(SCRIPT_DIR / 'test_llm_responses.csv')
    
    # Load questions
    with open(SCRIPT_DIR / 'test_questions.json', 'r') as f:
        questions = json.load(f)
    
    return df, questions

def calculate_response_similarity(responses):
    """Calculate similarity between responses using TF-IDF and cosine similarity"""
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(responses)
    similarity_matrix = cosine_similarity(tfidf_matrix)
    return similarity_matrix

def create_similarity_heatmap(df):
    """Create heatmap showing similarity between responses for the same questions"""
    # Group responses by question
    question_groups = df.groupby('Question (EN)')
    
    # Create figure with subplots for each question
    n_questions = len(question_groups)
    fig, axes = plt.subplots(1, n_questions, figsize=(6*n_questions, 5))
    if n_questions == 1:
        axes = [axes]
    
    for idx, (question, group) in enumerate(question_groups):
        # Calculate similarity matrix
        similarity_matrix = calculate_response_similarity(group['Response (EN)'].values)
        
        # Create heatmap
        sns.heatmap(similarity_matrix, 
                   annot=True, 
                   fmt='.2f',
                   cmap='YlGnBu',
                   xticklabels=[f'Response {i+1}' for i in range(len(group))],
                   yticklabels=[f'Response {i+1}' for i in range(len(group))],
                   ax=axes[idx])
        
        # Truncate question text for title
        title = question[:50] + '...' if len(question) > 50 else question
        axes[idx].set_title(f'Response Similarity\n{title}')
    
    plt.tight_layout()
    save_plot(fig, 'response_similarity_heatmap')

def create_response_length_comparison(df):
    """Create bar plot comparing response lengths for the same questions"""
    # Calculate response lengths
    df['Response Length'] = df['Response (EN)'].str.len()
    
    # Create figure
    plt.figure(figsize=(12, 6))
    
    # Plot response lengths
    sns.barplot(data=df, 
                x='Question (EN)', 
                y='Response Length',
                hue='Category')
    
    plt.title('Response Length Comparison Across Different Orders')
    plt.xlabel('Question')
    plt.ylabel('Response Length (characters)')
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Category', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    save_plot(plt.gcf(), 'response_length_comparison')

def create_category_consistency(df):
    """Create visualization showing consistency across categories"""
    # Calculate response lengths and group by category
    df['Response Length'] = df['Response (EN)'].str.len()
    category_stats = df.groupby('Category')['Response Length'].agg(['mean', 'std']).reset_index()
    
    # Create figure
    plt.figure(figsize=(10, 6))
    
    # Plot mean response length with error bars
    plt.bar(category_stats['Category'], category_stats['mean'], 
            yerr=category_stats['std'], capsize=10)
    
    plt.title('Response Length Consistency by Category')
    plt.xlabel('Category')
    plt.ylabel('Mean Response Length (characters)')
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    save_plot(plt.gcf(), 'category_consistency')

def main():
    """Main function to run the analysis"""
    print("Setting up visualization style...")
    setup_style()
    
    print("Loading data...")
    df, questions = load_data()
    
    print("Creating similarity heatmap...")
    create_similarity_heatmap(df)
    
    print("Creating response length comparison...")
    create_response_length_comparison(df)
    
    print("Creating category consistency visualization...")
    create_category_consistency(df)
    
    print("Analysis complete! Visualizations have been saved to the 'visualizations' directory.")

if __name__ == "__main__":
    main() 