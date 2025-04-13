import ollama
import csv
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional
import logging
import sys
import time
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ollama_queries.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

def load_questions(json_file: str) -> List[Dict]:
    """
    Load and flatten categorized questions from JSON file.
    
    Returns:
        List of questions with category information added
    """
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    questions = []
    for category, category_questions in data['categorized_questions'].items():
        for question in category_questions:
            question['category'] = category
            questions.append(question)
    questions.sort(key=lambda x: x['id'])
    return questions

models = ["aya", "deepseek-r1", "llama3.2", "falcon3", "phi", "qwen", "gemma"]

def warm_up_model(model: str, max_retries: int = 3) -> bool:
    """
    Warm up a model with a simple query to ensure it's loaded.
    """
    logging.info(f"Warming up model: {model}")
    for attempt in range(max_retries):
        try:
            start_time = time.time()
            response = ollama.generate(model=model, prompt="Hi", options={"num_predict": 1})
            if response and "response" in response:
                elapsed = time.time() - start_time
                logging.info(f"Model {model} warmed up successfully in {elapsed:.2f} seconds")
                return True
        except Exception as e:
            logging.warning(f"Warm-up attempt {attempt + 1} failed for {model}: {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(2)
    logging.error(f"Failed to warm up model {model} after {max_retries} attempts")
    return False

def query_model(model: str, prompt: str) -> Optional[str]:
    """
    Query an Ollama model with error handling and timeouts.
    """
    try:
        start_time = time.time()
        response = ollama.generate(model=model, prompt=prompt, options={"num_predict": 500})
        elapsed = time.time() - start_time
        
        if response and "response" in response:
            logging.info(f"Query to {model} completed in {elapsed:.2f} seconds")
            return response["response"]
        else:
            logging.error(f"Invalid response format from {model}: {response}")
            return None
    except Exception as e:
        logging.error(f"Error querying {model}: {str(e)}")
        return None

def process_batch(questions: List[Dict], models: List[str], output_file: str) -> None:
    """
    Process a batch of questions across multiple models in parallel.
    """
    warmed_up_models = [m for m in models if warm_up_model(m)]
    if not warmed_up_models:
        logging.error("No models successfully warmed up. Exiting.")
        return
    
    logging.info(f"Successfully warmed up {len(warmed_up_models)} models")
    
    with open(output_file, 'w', newline='', encoding='utf-8-sig') as file:
        writer = csv.writer(file)
        writer.writerow([
            "Question ID", "Category", "Question (EN)", "Question (MA)", 
            "Model", "Response (EN)", "Response (MA)"
        ])

        # Calculate total number of queries
        total_queries = len(questions) * len(warmed_up_models) * 2  # *2 for EN and MA
        progress_bar = tqdm(total=total_queries, desc="Processing queries")

        responses = {}
        with ThreadPoolExecutor(max_workers=3) as executor:
            future_to_params = {}
            
            # Submit all queries
            for model in warmed_up_models:
                for q in questions:
                    future_en = executor.submit(query_model, model, q["text_en"])
                    future_ma = executor.submit(query_model, model, q["text_ma"])
                    future_to_params[future_en] = {**q, "model": model, "is_english": True}
                    future_to_params[future_ma] = {**q, "model": model, "is_english": False}

            # Track responses
            for future in as_completed(future_to_params):
                params = future_to_params[future]
                q_id, model = params["id"], params["model"]
                
                if (q_id, model) not in responses:
                    responses[(q_id, model)] = {"en": None, "ma": None}
                
                try:
                    result = future.result()
                    key = "en" if params["is_english"] else "ma"
                    responses[(q_id, model)][key] = result
                    
                    progress_bar.update(1)
                    
                    # Write to CSV if we have both responses
                    if all(responses[(q_id, model)].values()):
                        writer.writerow([
                            q_id,
                            params["category"],
                            params["text_en"],
                            params["text_ma"],
                            model,
                            responses[(q_id, model)]["en"],
                            responses[(q_id, model)]["ma"]
                        ])
                        file.flush()
                except Exception as e:
                    logging.error(f"Error processing results for {model} on question {q_id}: {str(e)}")
                    progress_bar.update(1)

        progress_bar.close()

if __name__ == "__main__":
    try:
        logging.info("Starting LLM query process...")
        questions = load_questions('questions.json')
        total_questions = len(questions)
        total_categories = len(set(q['category'] for q in questions))
        logging.info(f"Loaded {total_questions} questions across {total_categories} categories")
        
        if total_questions != 49:
            logging.warning(f"Expected 49 questions but found {total_questions}")
        if total_categories != 7:
            logging.warning(f"Expected 7 categories but found {total_categories}")
        
        # Save to the existing 2_LLMs_Responses folder outside of 1_LLMs_Bulk_Queries
        output_file = os.path.join('..', '2_LLMs_Responses', 'llm_responses.csv')
        process_batch(questions, models, output_file)
        logging.info("Query process completed!")
    except KeyboardInterrupt:
        logging.info("Process interrupted by user")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Process failed: {str(e)}")
        sys.exit(1)
