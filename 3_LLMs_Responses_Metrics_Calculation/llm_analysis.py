import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import json
import os

# Load knowledge and sensitivity markers
def load_data():
    knowledge = {}
    for lang, file in [('english', 'english_knowledge.json'), ('arabic', 'arabic_knowledge.json')]:
        with open(os.path.join('reference_knowledge_en_ar', file), 'r', encoding='utf-8') as f:
            knowledge[lang] = json.load(f)[f'reference_knowledge_{lang}']
    
    sensitivity = {
        'english': {
            'positive': ['diverse', 'nuanced', 'regional variation', 'cultural context'],
            'negative': ['all Arabs', 'Middle Eastern', 'like in Egypt', 'primitive']
        },
        'arabic': {
            'positive': ['متنوع', 'دقيق', 'اختلاف إقليمي', 'سياق ثقافي'],
            'negative': ['كل العرب', 'شرق أوسطي', 'مثل مصر', 'بدائي']
        }
    }
    return knowledge, sensitivity

def calculate_metrics(response, knowledge, sensitivity, is_arabic=False):
    metrics = {}
    prefix = 'AR_' if is_arabic else 'EN_'
    
    # Keyword and fact matching
    keywords = knowledge['keywords']
    facts = knowledge['facts']
    response_lower = response.lower()
    
    keyword_ratio = sum(1 for kw in keywords if re.search(r'\b' + re.escape(kw.lower()) + r'\b', response_lower)) / len(keywords) if keywords else 0
    fact_ratio = sum(1 for fact in facts if any(re.search(re.escape(term.lower()), response_lower) for term in fact.split())) / len(facts) if facts else 0
    
    # Cultural sensitivity
    pos_markers = sum(1 for term in sensitivity['positive'] if term.lower() in response_lower)
    neg_markers = sum(1 for term in sensitivity['negative'] if term.lower() in response_lower)
    
    # Calculate all metrics
    metrics[f'{prefix}Set Coverage'] = keyword_ratio * 100
    metrics[f'{prefix}Response Accuracy Rate'] = fact_ratio * 100
    metrics[f'{prefix}Accuracy_Scale'] = min(5, max(1, int(fact_ratio * 5) + 1))
    metrics[f'{prefix}Cultural_Sensitivity'] = min(5, max(1, 3 + pos_markers - neg_markers))
    metrics[f'{prefix}Language_Quality'] = min(5, max(1, 3 + (1 if 10 <= len(response.split()) / len(re.split(r'[.!?]', response)) <= 25 else 0) - (1 if neg_markers > 0 else 0)))
    metrics[f'{prefix}Contextual_Relevance'] = min(5, max(1, (keyword_ratio * 2.5) + (fact_ratio * 2.5)))
    
    return metrics

def analyze_responses(df):
    knowledge, sensitivity = load_data()
    
    # Calculate cross-model agreement and language disparity
    for q_id in df['Question ID'].unique():
        q_df = df[df['Question ID'] == q_id]
        for lang, col in [('EN', 'Response (EN)'), ('AR', 'Response (MA)')]:
            try:
                vectorizer = TfidfVectorizer()
                vectors = vectorizer.fit_transform(q_df[col])
                sim_matrix = cosine_similarity(vectors)
                for i, model in enumerate(q_df['Model'].values):
                    df.loc[(df['Question ID'] == q_id) & (df['Model'] == model), f'{lang}_Cross-Model Agreement'] = (np.sum(sim_matrix[i]) - 1) / (len(q_df) - 1) * 100
            except: continue
    
    # Calculate language disparity
    for idx, row in df.iterrows():
        if pd.notna(row['Response (EN)']) and pd.notna(row['Response (MA)']):
            try:
                vectors = TfidfVectorizer().fit_transform([str(row['Response (EN)']), str(row['Response (MA)'])])
                df.loc[idx, 'Language Disparity'] = (1 - cosine_similarity(vectors)[0][1]) * 100
            except: continue
    
    # Calculate response metrics
    for idx, row in df.iterrows():
        category = row['Category']
        for lang, col, is_arabic in [('english', 'Response (EN)', False), ('arabic', 'Response (MA)', True)]:
            if category in knowledge[lang]:
                metrics = calculate_metrics(str(row[col]), knowledge[lang][category], sensitivity[lang], is_arabic)
                for metric, value in metrics.items():
                    df.loc[idx, metric] = value
    
    return df

# Process and save results
df = pd.read_csv('../2_LLMs_Responses/llm_responses.csv')
processed_df = analyze_responses(df)

# Reorder and format columns
base_cols = ['Question ID', 'Category', 'Question (EN)', 'Question (MA)', 'Model', 'Response (EN)', 'Response (MA)']
en_metrics = [col for col in processed_df.columns if col.startswith('EN_')]
ar_metrics = [col for col in processed_df.columns if col.startswith('AR_')]
other_metrics = [col for col in processed_df.columns if col not in base_cols + en_metrics + ar_metrics]

processed_df = processed_df[base_cols + en_metrics + ar_metrics + other_metrics]
for col in en_metrics + ar_metrics + ['Language Disparity']:
    if col in processed_df.columns:
        processed_df[col] = processed_df[col].round().astype('Int64')

processed_df.to_csv('processed_results.csv', index=False)