#!/usr/bin/env python3
import os
import json
import glob
import numpy as np
from http.server import HTTPServer, SimpleHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
import mimetypes

class DashboardHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        # Cache for loaded JSON files to avoid reloading
        self.file_cache = {}
        super().__init__(*args, **kwargs)
    
    def do_GET(self):
        parsed_path = urlparse(self.path)
        query_params = parse_qs(parsed_path.query)
        
        if parsed_path.path == '/api/files':
            self.handle_files_api()
        elif parsed_path.path == '/api/questions':
            file_path = query_params.get('file', [None])[0]
            if file_path:
                self.handle_questions_api(file_path)
            else:
                self.send_error(400, "Missing file parameter")
        elif parsed_path.path == '/api/question-data':
            file_path = query_params.get('file', [None])[0]
            question_index = query_params.get('question', [None])[0]
            if file_path and question_index is not None:
                try:
                    question_index = int(question_index)
                    self.handle_question_data_api(file_path, question_index)
                except ValueError:
                    self.send_error(400, "Invalid question index")
            else:
                self.send_error(400, "Missing file or question parameter")
        elif parsed_path.path == '/api/file-summary':
            file_path = query_params.get('file', [None])[0]
            if file_path:
                self.handle_file_summary_api(file_path)
            else:
                self.send_error(400, "Missing file parameter")
        else:
            # Serve static files (HTML, CSS, JS)
            super().do_GET()
    
    def load_json_file(self, file_path):
        """Load and cache JSON file"""
        if file_path in self.file_cache:
            return self.file_cache[file_path]
        
        # Look for file relative to parent directory (outside dashboard_tokens)
        full_path = os.path.join('..', file_path)
        if not os.path.exists(full_path):
            # Fallback: try current directory
            full_path = file_path
        
        try:
            with open(full_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.file_cache[file_path] = data
                return data
        except Exception as e:
            raise Exception(f"Error loading file {file_path}: {str(e)}")
    
    def handle_files_api(self):
        """Return list of available JSON files in the generation_comparison directory"""
        try:
            files = []
            
            # Look for JSON files in the generation_comparison directory structure
            # Search in parent directory (outside dashboard_tokens)
            pattern = "../generation_comparison/**/generation_comparison_T_e_*.json"
            json_files = glob.glob(pattern, recursive=True)
            
            for file_path in json_files:
                # Convert back to relative path from parent directory
                relative_path = file_path[3:]  # Remove '../' prefix
                
                # Extract meaningful name from path
                dir_parts = relative_path.split('/')
                if len(dir_parts) >= 3:
                    experiment_dir = dir_parts[1]  # e.g., "answer_directly"
                    print(experiment_dir)
                    experiment_parameters = dir_parts[2]
                    temperature = experiment_parameters.split('_')[2]
                    num_examples = experiment_parameters.split('_')[1]
                    num_rounds = experiment_parameters.split('_')[0]
                    filename = dir_parts[-1]       # e.g., "generation_comparison_T_e_50_k_4.json"
                    
                    # Extract parameters from filename
                    import re
                    match = re.search(r'T_e_(\d+)_k_(\d+)', filename)
                    if match:
                        t_e, k = match.groups()
                        
                        display_name = f"{experiment_dir} - T_e={t_e}, k={k}, T={num_rounds}, N={num_examples}, T={temperature}"
                    else:
                        display_name = f"{experiment_dir} - {filename}"
                else:
                    display_name = relative_path
                
                files.append({
                    'path': relative_path,
                    'name': display_name
                })
            
            # Sort files by name for better organization
            files.sort(key=lambda x: x['name'])
            
            print(f"Found {len(files)} files")
            
            # Send JSON response
            self.send_json_response(files)
            
        except Exception as e:
            self.send_error(500, f"Error listing files: {str(e)}")
    
    def handle_questions_api(self, file_path):
        """Return list of questions for a specific file"""
        try:
            data = self.load_json_file(file_path)
            
            if not (data and 'embedding_mixture' in data and 'results' in data['embedding_mixture']):
                raise Exception("Invalid file format - missing embedding_mixture.results")
            
            questions = []
            results = data['embedding_mixture']['results']
            
            for index, result in enumerate(results):
                questions.append({
                    'index': index,
                    'question': result.get('question', ''),
                    'ground_truth': result.get('ground_truth'),
                    'predicted_answer': result.get('predicted_answer'),
                    'is_correct': result.get('is_correct', False)
                })
            
            self.send_json_response({
                'questions': questions,
                'total_count': len(questions)
            })
            
        except Exception as e:
            self.send_error(500, f"Error loading questions: {str(e)}")
    
    def handle_question_data_api(self, file_path, question_index):
        """Return detailed data for a specific question"""
        try:
            data = self.load_json_file(file_path)
            
            if not (data and 'embedding_mixture' in data and 'results' in data['embedding_mixture']):
                raise Exception("Invalid file format - missing embedding_mixture.results")
            
            results = data['embedding_mixture']['results']
            
            if question_index >= len(results):
                raise Exception(f"Question index {question_index} out of range (max: {len(results)-1})")
            
            result = results[question_index]
            phase_info = result.get('phase_info', {})
            
            # Prepare response data
            response_data = {
                'question': result.get('question', ''),
                'ground_truth': result.get('ground_truth'),
                'predicted_answer': result.get('predicted_answer'),
                'is_correct': result.get('is_correct', False),
                'answer_text': result.get('answer_text', ''),
                'phase_info': {
                    'total_tokens': phase_info.get('total_tokens', 0),
                    'phase1_tokens': phase_info.get('phase1_tokens', 0),
                    'phase2_tokens': phase_info.get('phase2_tokens', 0),
                    'transition_tokens': phase_info.get('transition_tokens', 0),
                    'phase1_rounds_completed': phase_info.get('phase1_rounds_completed', 0),
                    'phase1_rounds_requested': phase_info.get('phase1_rounds_requested', 0),
                    'phase2_rounds_requested': phase_info.get('phase2_rounds_requested', 0),
                    'phase1_token_ids': phase_info.get('phase1_token_ids', []),
                    'phase2_token_ids': phase_info.get('phase2_token_ids', []),
                    'transition_token_ids': phase_info.get('transition_token_ids', [])
                }
            }
            
            self.send_json_response(response_data)
            
        except Exception as e:
            self.send_error(500, f"Error loading question data: {str(e)}")
    
    def handle_file_summary_api(self, file_path):
        """Return statistical summary for a specific file"""
        try:
            data = self.load_json_file(file_path)
            
            if not (data and 'embedding_mixture' in data):
                raise Exception("Invalid file format - missing embedding_mixture")
            
            embedding_mixture = data['embedding_mixture']
            results = embedding_mixture.get('results', [])
            
            if not results:
                raise Exception("No results found in file")
            
            # Calculate basic statistics
            total_questions = len(results)
            correct_answers = sum(1 for r in results if r.get('is_correct', False))
            accuracy = correct_answers / total_questions if total_questions > 0 else 0
            
            # Get token counts
            token_counts = embedding_mixture.get('token_counts', [])
            if not token_counts:
                # Fallback: try to get from phase_info
                token_counts = []
                for result in results:
                    phase_info = result.get('phase_info', {})
                    total_tokens = phase_info.get('total_tokens', 0)
                    if total_tokens > 0:
                        token_counts.append(total_tokens)
            
            avg_tokens = np.mean(token_counts) if token_counts else 0
            
            # Bootstrap confidence intervals (similar to print_generation.py)
            n_boot = 1000  # Reduced for faster response
            
            # Accuracy confidence interval
            is_correct_list = [1 if r.get('is_correct', False) else 0 for r in results]
            boot_acc = []
            for _ in range(n_boot):
                sample = np.random.choice(is_correct_list, size=len(is_correct_list), replace=True)
                boot_acc.append(np.mean(sample))
            
            acc_ci_lower, acc_ci_upper = np.percentile(boot_acc, [2.5, 97.5])
            
            # Token count confidence interval
            tok_ci_lower, tok_ci_upper = 0, 0
            if token_counts:
                boot_tok = []
                for _ in range(n_boot):
                    sample_tok = np.random.choice(token_counts, size=len(token_counts), replace=True)
                    boot_tok.append(np.mean(sample_tok))
                
                tok_ci_lower, tok_ci_upper = np.percentile(boot_tok, [2.5, 97.5])
            
            # Calculate additional statistics
            phase1_tokens = []
            phase2_tokens = []
            phase1_rounds = []
            
            for result in results:
                phase_info = result.get('phase_info', {})
                if phase_info.get('phase1_tokens', 0) > 0:
                    phase1_tokens.append(phase_info['phase1_tokens'])
                if phase_info.get('phase2_tokens', 0) > 0:
                    phase2_tokens.append(phase_info['phase2_tokens'])
                if phase_info.get('phase1_rounds_completed', 0) > 0:
                    phase1_rounds.append(phase_info['phase1_rounds_completed'])
            
            # Extract experiment parameters from filename
            import re
            filename = os.path.basename(file_path)
            match = re.search(r'T_e_(\d+)_k_(\d+)', filename)
            experiment_params = {}
            if match:
                experiment_params['T_e'] = int(match.group(1))
                experiment_params['k'] = int(match.group(2))
            
            # Prepare summary data
            summary_data = {
                'experiment_params': experiment_params,
                'filename': filename,
                'total_questions': total_questions,
                'correct_answers': correct_answers,
                'accuracy': accuracy,
                'accuracy_ci': {
                    'lower': acc_ci_lower,
                    'upper': acc_ci_upper
                },
                'avg_tokens': avg_tokens,
                'token_ci': {
                    'lower': tok_ci_lower,
                    'upper': tok_ci_upper
                },
                'phase_stats': {
                    'avg_phase1_tokens': np.mean(phase1_tokens) if phase1_tokens else 0,
                    'avg_phase2_tokens': np.mean(phase2_tokens) if phase2_tokens else 0,
                    'avg_phase1_rounds': np.mean(phase1_rounds) if phase1_rounds else 0,
                    'phase1_token_std': np.std(phase1_tokens) if phase1_tokens else 0,
                    'phase2_token_std': np.std(phase2_tokens) if phase2_tokens else 0
                },
                'token_distribution': {
                    'min_tokens': int(np.min(token_counts)) if token_counts else 0,
                    'max_tokens': int(np.max(token_counts)) if token_counts else 0,
                    'median_tokens': int(np.median(token_counts)) if token_counts else 0,
                    'std_tokens': np.std(token_counts) if token_counts else 0
                }
            }
            
            self.send_json_response(summary_data)
            
        except Exception as e:
            self.send_error(500, f"Error generating file summary: {str(e)}")
    
    def send_json_response(self, data):
        """Send JSON response with proper headers"""
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        
        response = json.dumps(data, indent=2)
        self.wfile.write(response.encode('utf-8'))
    
    def end_headers(self):
        # Add CORS headers
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        super().end_headers()

def run_server(port=8000):
    """Run the dashboard server"""
    server_address = ('0.0.0.0', port)
    httpd = HTTPServer(server_address, DashboardHandler)
    
    print(f"Dashboard server running on all interfaces at port {port}")
    print(f"If using VSCode port forwarding, the server should be accessible through the forwarded port")
    print(f"Open your browser and navigate to http://localhost:{port}/token_generation_dashboard.html")
    print("Press Ctrl+C to stop the server")
    
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")
        httpd.server_close()

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Run the Token Generation Dashboard server')
    parser.add_argument('--port', type=int, default=8000, help='Port to run the server on (default: 8000)')
    args = parser.parse_args()
    
    run_server(args.port) 