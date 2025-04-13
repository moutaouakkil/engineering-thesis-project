import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load data and setup
df = pd.read_csv('../3_LLMs_Responses_Metrics_Calculation/processed_results.csv')
os.makedirs('visualizations', exist_ok=True)

def create_visualizations():
    # Set style
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    # 1. Radar Chart
    metrics = ['EN_Response Accuracy Rate', 'EN_Cultural_Sensitivity', 'EN_Language_Quality', 'EN_Contextual_Relevance']
    model_perf = df.groupby('Model')[metrics].mean()
    categories = ['Accuracy', 'Cultural Sensitivity', 'Language Quality', 'Relevance']
    angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(12, 12), subplot_kw=dict(polar=True))
    for i, (model, row) in enumerate(model_perf.iterrows()):
        values = row.values.tolist() + [row.values[0]]
        ax.plot(angles, values, 'o-', linewidth=2, label=model, color=plt.cm.tab10(i))
        ax.fill(angles, values, alpha=0.1, color=plt.cm.tab10(i))
    plt.xticks(angles[:-1], categories, size=12)
    plt.legend(loc='center left', bbox_to_anchor=(1.2, 0.5))
    plt.title('Model Performance Comparison', size=15, pad=20)
    ax.set_ylim(0, 100)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.set_yticklabels(['0%', '20%', '40%', '60%', '80%', '100%'])
    ax.grid(True, alpha=0.3)
    ax.set_rgrids([20, 40, 60, 80, 100], angle=0)
    plt.savefig('visualizations/model_radar_chart.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 2. Language Disparity
    disparity = df.groupby('Model').apply(lambda x: abs(x['EN_Response Accuracy Rate'].mean() - x['AR_Response Accuracy Rate'].mean()))
    disparity = disparity.sort_values(ascending=True)  # Sort in ascending order
    plt.figure(figsize=(12, 6))
    sns.barplot(x=disparity.index, y=disparity.values, palette='viridis')  # Restore viridis palette
    plt.title('Language Performance Disparity by Model\n(Lower is better - shows absolute difference between EN and AR accuracy)', fontsize=14)
    plt.ylabel('Accuracy Difference (%)')
    plt.xlabel('Model')
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    for i, v in enumerate(disparity.values):
        plt.text(i, v + 0.5, f'{v:.1f}%', ha='center', fontsize=10)
    plt.tight_layout()
    plt.savefig('visualizations/language_disparity.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 3. Category Heatmap
    heatmap_data = df.pivot_table(index='Model', columns='Category', values='EN_Response Accuracy Rate')
    plt.figure(figsize=(14, 8))
    sns.heatmap(heatmap_data, annot=True, cmap='YlGnBu', fmt='.1f', linewidths=.5, cbar_kws={'label': 'Accuracy Rate (%)'})
    plt.title('Model Performance by Category', fontsize=15)
    plt.tight_layout()
    plt.savefig('visualizations/category_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 4. Language Comparison
    lang_comp = df.groupby('Model')[['EN_Response Accuracy Rate', 'AR_Response Accuracy Rate']].mean()
    plt.figure(figsize=(14, 7))
    x = np.arange(len(lang_comp))
    width = 0.35
    plt.bar(x - width/2, lang_comp['EN_Response Accuracy Rate'], width, label='English', color='royalblue')
    plt.bar(x + width/2, lang_comp['AR_Response Accuracy Rate'], width, label='Arabic/Darija', color='tomato')
    plt.xlabel('Model')
    plt.ylabel('Accuracy Rate (%)')
    plt.title('English vs Arabic/Darija Performance by Model', fontsize=15)
    plt.xticks(x, lang_comp.index, rotation=45)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig('visualizations/language_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 5. Sensitivity vs Accuracy
    model_metrics = df.groupby('Model').agg({
        'EN_Response Accuracy Rate': 'mean',
        'EN_Cultural_Sensitivity': 'mean',
        'EN_Contextual_Relevance': 'mean'
    })
    plt.figure(figsize=(10, 8))
    plt.scatter(model_metrics['EN_Cultural_Sensitivity'], model_metrics['EN_Response Accuracy Rate'],
               s=model_metrics['EN_Contextual_Relevance'] * 5, alpha=0.7, c=range(len(model_metrics)), cmap='viridis')
    for model in model_metrics.index:
        plt.annotate(model, (model_metrics.loc[model, 'EN_Cultural_Sensitivity'],
                           model_metrics.loc[model, 'EN_Response Accuracy Rate']),
                    xytext=(7, -5), textcoords='offset points')
    plt.xlabel('Cultural Sensitivity Score')
    plt.ylabel('Accuracy Rate (%)')
    plt.title('Cultural Sensitivity vs Accuracy by Model\n(Bubble size = Contextual Relevance)', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('visualizations/sensitivity_accuracy.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 6. Correlation Matrix
    metric_cols = [col for col in df.columns if any(x in col for x in ['Accuracy', 'Sensitivity', 'Quality', 'Relevance', 'Coverage', 'Agreement'])]
    plt.figure(figsize=(12, 10))
    mask = np.triu(np.ones_like(df[metric_cols].corr(), dtype=bool))
    sns.heatmap(df[metric_cols].corr(), mask=mask, cmap='coolwarm', center=0,
                annot=True, fmt='.2f', square=True, linewidths=.5)
    plt.title('Correlation Matrix of Evaluation Metrics', fontsize=16)
    plt.tight_layout()
    plt.savefig('visualizations/correlation_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 7. Category Difficulty
    cat_diff = df.groupby('Category')['EN_Response Accuracy Rate'].agg(['mean', 'std']).reset_index()
    cat_diff = cat_diff.sort_values('mean', ascending=True)  # Sort in ascending order
    plt.figure(figsize=(12, 6))
    plt.errorbar(x=cat_diff['Category'], y=cat_diff['mean'],
                yerr=cat_diff['std'], fmt='o', capsize=5, ecolor='red', capthick=2,
                color='royalblue', markersize=8)  # Add color and marker size
    plt.title('Category Difficulty Ranking with Uncertainty', fontsize=15)
    plt.ylabel('Average Accuracy Rate (%)')
    plt.xlabel('Category')
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('visualizations/category_difficulty.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 8. Model Performance by Category
    fig, axes = plt.subplots(4, 2, figsize=(18, 20))
    for i, (category, ax) in enumerate(zip(df['Category'].unique(), axes.flatten())):
        cat_data = df[df['Category'] == category]
        model_perf = cat_data.groupby('Model')['EN_Response Accuracy Rate'].mean().sort_values()
        sns.barplot(x=model_perf.index, y=model_perf.values, palette='viridis', ax=ax)  # Restore viridis palette
        ax.set_title(f'Model Performance on {category} Questions', fontsize=12)
        ax.set_ylabel('Accuracy Rate (%)')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(axis='y', linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig('visualizations/model_by_category.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 9. Cross-Model Agreement
    try:
        agreement = df.groupby('Category')['EN_Cross-Model Agreement'].mean().sort_values()
        plt.figure(figsize=(12, 6))
        sns.barplot(x=agreement.index, y=agreement.values, palette='Blues_d')  # Restore Blues_d palette
        plt.title('Cross-Model Agreement by Category', fontsize=15)
        plt.ylabel('Average Agreement Score (%)')
        plt.xticks(rotation=45)
        plt.grid(axis='y', linestyle='--', alpha=0.3)
        plt.tight_layout()
        plt.savefig('visualizations/cross_model_agreement.png', dpi=300, bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(f"Error creating cross-model agreement visualization: {e}")

    # 10. Model Consistency
    try:
        model_var = df.groupby('Model')['EN_Response Accuracy Rate'].agg(['mean', 'std']).reset_index()
        model_var['coefficient_of_variation'] = (model_var['std'] / model_var['mean']) * 100
        model_var = model_var.sort_values('coefficient_of_variation')
        
        plt.figure(figsize=(12, 6))
        sns.barplot(x='Model', y='coefficient_of_variation', data=model_var, palette='viridis')  # Restore viridis palette
        
        plt.title('Model Consistency Analysis\n(Lower values indicate more consistent performance)', 
                 fontsize=14)
        plt.ylabel('Coefficient of Variation (%)')
        plt.xticks(rotation=45)
        plt.grid(axis='y', linestyle='--', alpha=0.3)
        plt.tight_layout()
        plt.savefig('visualizations/model_consistency.png', dpi=300, bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(f"Error creating model consistency visualization: {e}")

print("Creating visualizations...")
create_visualizations()
print("All visualizations complete! Images saved in the 'visualizations' folder.")