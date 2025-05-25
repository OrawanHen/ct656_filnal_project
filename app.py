from flask import Flask, render_template, request, jsonify, session
import pandas as pd
import os
import requests
from sklearn.metrics.pairwise import cosine_similarity
import json
import uuid
from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_community.chat_message_histories import ChatMessageHistory
from typing import List

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'  # Change this to a secure random key in production
DEEPSEEK_API_KEY = 'sk-d499803f76e44b148d1abc38b97b3da4'

# Initialize embeddings and vector store
embeddings = OpenAIEmbeddings(openai_api_key=DEEPSEEK_API_KEY)
vector_store = None

# Initialize conversation memory storage
conversation_memories = {}

def load_documents():
    global vector_store
    if os.path.exists("rag_data") and os.listdir("rag_data"):
        loader = DirectoryLoader("rag_data", glob="**/*.txt")
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        texts = text_splitter.split_documents(documents)
        vector_store = FAISS.from_documents(texts, embeddings)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

@app.route('/')
def home():
    return render_template('index.html')

def generate_synthesis(prompt, data=None):
    headers = {
        'Authorization': f'Bearer {DEEPSEEK_API_KEY}',
        'Content-Type': 'application/json'
    }
    
    content = prompt
    if data:
        content = f"Here is the dataset:\n{data}\n\n{prompt}"
    
    payload = {
        "model": "deepseek-chat",
        "messages": [
            {
                "role": "system", 
                "content": "You are an expert data analysis and synthesis assistant. Your tasks are:\n1. Thoroughly analyze datasets\n2. Identify key patterns/trends\n3. Generate structured reports\n4. Provide actionable recommendations\n5. Format clearly with headings\n6. Include relevant statistics\n7. Explain concepts simply\n8. Highlight data quality issues"
            },
            {
                "role": "user",
                "content": content
            }
        ],
        "temperature": 0.7
    }
    
    response = requests.post(
        "https://api.deepseek.com/v1/chat/completions",
        headers=headers,
        json=payload
    )
    
    if response.status_code == 200:
        return response.json()['choices'][0]['message']['content']
    else:
        return f"Error from DeepSeek API: {response.text}"

@app.route('/chat', methods=['POST'])
def chat():
    # Initialize or retrieve conversation memory
    if 'session_id' not in session:
        session['session_id'] = str(uuid.uuid4())
    
    if session['session_id'] not in conversation_memories:
        conversation_memories[session['session_id']] = ChatMessageHistory()
    
    memory = conversation_memories[session['session_id']]
    user_input = request.form.get('message')
    file = request.files.get('file')
    
    response = ""
    
    # First try to get relevant context from RAG
    context = ""
    if vector_store and user_input:
        docs = vector_store.similarity_search(user_input, k=3)
        context = "\n\nRelevant context:\n" + "\n---\n".join([doc.page_content for doc in docs])
    
    # Get conversation history
    history = "\n".join([f"{msg.type}: {msg.content}" for msg in memory.messages])
    
    if file and file.filename.endswith('.xlsx'):
        try:
            # Read Excel file directly from memory without saving
            df = pd.read_excel(file)
            columns = df.columns.tolist()
            sample_data = df.head().to_dict(orient='records')
            
            # Generate synthetic data prompt
            num_rows = 10  # Default
            if user_input and "rows" in user_input.lower():
                try:
                    num_rows = int(user_input.split()[-1])
                except:
                    pass
            
            prompt = f"""
            Generate {num_rows} synthetic data rows matching this Excel structure:
            Columns: {columns}
            Sample data patterns: {sample_data}
            
            Strict Requirements:
            1. Maintain EXACT data types from sample (dates as dates, numbers as numbers, etc)
            2. Preserve value ranges and distributions from sample
            3. Generate realistic but artificial data that follows patterns
            4. Return ONLY a properly formatted Python list of dictionaries
            5. Include all original columns
            6. Maintain any categorical/enumerated values
            
            Example format: [{{"col1": val1, "col2": val2}}, ...]
            
            Important: Do NOT include any explanations or markdown - ONLY the list of dictionaries.
            """
            
            response = generate_synthesis(prompt)
            try:
                # Parse the generated data into a DataFrame
                # Safely evaluate the response to get the data
                try:
                    synthetic_data = eval(response)
                    if not isinstance(synthetic_data, list):
                        raise ValueError("Response was not a list of dictionaries")
                    if not all(isinstance(item, dict) for item in synthetic_data):
                        raise ValueError("Not all items in the list are dictionaries")
                    
                    # Validate all original columns are present
                    missing_cols = [col for col in columns if col not in synthetic_data[0]]
                    if missing_cols:
                        raise ValueError(f"Missing columns in generated data: {missing_cols}")
                        
                    synthetic_df = pd.DataFrame(synthetic_data)
                except Exception as e:
                    raise ValueError(f"Error validating generated data: {str(e)}")
                
                # Create response with markdown preview
                # Generate statistics for both datasets
                def get_stats(df):
                    stats = {}
                    for col in df.columns:
                        col_stats = {
                            'type': str(df[col].dtype),
                            'unique': df[col].nunique(),
                            'nulls': df[col].isnull().sum(),
                            'values': df[col].values
                        }
                        if pd.api.types.is_numeric_dtype(df[col]):
                            col_stats.update({
                                'mean': df[col].mean(),
                                'std': df[col].std(),
                                'min': df[col].min(),
                                '25%': df[col].quantile(0.25),
                                '50%': df[col].quantile(0.5),
                                '75%': df[col].quantile(0.75),
                                'max': df[col].max()
                            })
                        stats[col] = col_stats
                    return stats

                original_stats = get_stats(df)
                synthetic_stats = get_stats(synthetic_df)

                # Format comparison report
                response = f"## Original Data Statistics ({len(df)} rows)\n\n"
                for col, stats in original_stats.items():
                    response += f"### {col}\n"
                    response += f"- Type: {stats['type']}\n"
                    response += f"- Unique values: {stats['unique']}\n"
                    response += f"- Null values: {stats['nulls']}\n"
                    if 'mean' in stats:
                        response += f"- Mean: {stats['mean']:.2f}\n"
                        response += f"- Std Dev: {stats['std']:.2f}\n"
                        response += f"- Range: {stats['min']:.2f} to {stats['max']:.2f}\n"
                    response += "\n"

                response += f"\n## Synthetic Data Statistics ({len(synthetic_df)} rows)\n\n"
                response += synthetic_df.to_markdown() + "\n\n"
                response += "### Statistical Comparison\n"
                for col, stats in synthetic_stats.items():
                    response += f"#### {col}\n"
                    if 'mean' in stats:
                        # Calculate differences
                        mean_diff = abs(stats['mean'] - original_stats[col]['mean'])
                        std_diff = abs(stats['std'] - original_stats[col]['std'])
                        min_diff = abs(stats['min'] - original_stats[col]['min'])
                        max_diff = abs(stats['max'] - original_stats[col]['max'])
                        
                        # Format comparisons
                        response += f"- Mean difference: {mean_diff:.2f} ({mean_diff/original_stats[col]['mean']*100:.1f}%)\n"
                        response += f"- Std Dev difference: {std_diff:.2f}\n"
                        response += f"- Range difference: {min_diff:.2f} (min), {max_diff:.2f} (max)\n"
                        response += f"- Distribution similarity: {min(stats['std']/original_stats[col]['std'], original_stats[col]['std']/stats['std']):.2f}\n"
                        
                        # Calculate cosine similarity for numeric columns
                        if pd.api.types.is_numeric_dtype(df[col]):
                            # Reshape and normalize the data
                            orig_values = original_stats[col]['values'].reshape(-1, 1)
                            synth_values = stats['values'].reshape(-1, 1)
                            
                            try:
                                # Calculate cosine similarity
                                similarity = cosine_similarity(orig_values, synth_values)[0][0]
                                response += f"- Cosine similarity: {similarity:.3f}\n"
                            except Exception as e:
                                response += f"- Cosine similarity calculation failed: {str(e)}\n"
                    
                    # Compare unique values for categorical data
                    if not pd.api.types.is_numeric_dtype(df[col]):
                        unique_ratio = stats['unique'] / original_stats[col]['unique']
                        response += f"- Unique values ratio: {unique_ratio:.2f}\n"
                    
                    response += "\n"
            except Exception as e:
                response = f"Error processing generated data: {str(e)}\nOriginal response:\n{response}"
        except Exception as e:
            response = f"Error processing Excel file: {str(e)}"
    elif user_input:
        full_prompt = f"{history}\n\n{user_input + context}"
        response = generate_synthesis(full_prompt)
    else:
        response = "Please provide either text input or upload an Excel file"
    
    # Save conversation to memory
    if user_input:
        memory.add_user_message(user_input)
        memory.add_ai_message(response)
    
    result = {
        'response': response,
        'columns': columns if file and file.filename.endswith('.xlsx') else None,
        'data': sample_data if file and file.filename.endswith('.xlsx') else None
    }
    
    # if file and file.filename.endswith('.xlsx') and 'excel_b64' in locals():
    #     result['excel_data'] = excel_b64
    #     result['csv_data'] = csv_data
    
    return jsonify(result)

# @app.route('/upload_doc', methods=['POST'])
# def upload_doc():
#     file = request.files.get('file')
#     if file and file.filename.endswith('.txt'):
#         filepath = os.path.join("rag_data", file.filename)
#         file.save(filepath)
#         load_documents()
#         return jsonify({'status': 'success'})
#     return jsonify({'status': 'error', 'message': 'Invalid file type'})

if __name__ == '__main__':
    # Load documents on startup
    if not os.path.exists("rag_data"):
        os.makedirs("rag_data")
    load_documents()
    app.run(debug=True)
