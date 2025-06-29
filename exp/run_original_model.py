import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import json
import os
from common import setup_env
from models import build_tokenizer, build_model
from utils.data_utils import load_and_prepare_dataset

from tqdm import tqdm
from utils.eval_utils import check_response_accuracy, load_formatting_prompt, load_previous_results

from transformers import logging
logging.set_verbosity_error()

class Evaluator:
    def __init__(self, init_prompt, ds_args, prm_model_args, gen_model_args, root_path='./records'):
        """
        Evaluator class to evaluate the performance of a model on a dataset with or wihout latent shifting.

        Args:
            init_prompt (str): Initial prompt to be used for the evaluation.
            ds_args (dict): Dictionary containing dataset arguments such as dataset name and split.
            prm_model_args (dict): Dictionary containing PRM model arguments such as model name and device.
            gen_model_args (dict): Dictionary containing generation model arguments such as model type, size, GPUs, seed, and 8-bit flag.
            root_path (str, optional): Path to save the evaluation records. Defaults to './records'.
        """
        if not os.path.exists(root_path):
            os.makedirs(root_path)

        self.init_prompt = init_prompt
        self.root_path = root_path
        self.prm_model_args = prm_model_args
        self.ds_args = ds_args
        self.gen_model_args = gen_model_args
        self.dataset = load_and_prepare_dataset(ds_args)
        # Initialize PRM tokenizer and model
        self.tokenizer = build_tokenizer(
            gen_model_args['model_type'], gen_model_args['model_size'], padding_side="right"
        )
        self.gen_model = build_model(
            gen_model_args['model_type'], gen_model_args['model_size'], gen_model_args['in_8bit']
        )

        if 'llama' in gen_model_args['model_type']:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        torch.autograd.set_grad_enabled(False)

    def get_exp_name(self, model_type, dataset_name, start_sample, n_samples, icv_mode):
        return f"{model_type.replace('/','-')}_{dataset_name.replace('/','-')}_{start_sample}_{n_samples}_icv_type_{icv_mode}"

    def formatting_final_answers(self, best_generation):
        """
        Formats the final answers from the generated text.

        Args:
            query (str): The original query.
            best_generation (list): List of generated text.
        """
        if self.gen_model_args['model_type'] == 'R1Qwen' and self.ds_args['dataset_name'] == 'HuggingFaceH4/MATH-500':
            return best_generation
        final_answers = []
        for response_each in best_generation:
            try:
                final_answers.append('Answer: ' + response_each.split("Answer:")[1].strip())
            except:
                try:
                    final_answers.append('Answer: ' + response_each.split("ANSWER:")[1].strip())
                except:
                    final_answers.append('Answer: ' + 'null')
        return final_answers
    
        
    
    def generate(self, input_ids, num_return_sequences=5):
        if 'Qwen' in self.gen_model_args['model_type']:
            terminators = self.tokenizer.eos_token_id
        elif 'llama' in self.gen_model_args['model_type']:
            terminators = [
                self.tokenizer.eos_token_id,
                self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
            ]
        if 'R1Qwen' in self.gen_model_args['model_type']:
            generation_outputs = self.gen_model.generate(
                                input_ids=input_ids,
                                max_new_tokens=8192,
                                temperature=0.6,
                                top_k=40,
                                top_p=0.95,
                                num_return_sequences=num_return_sequences,
                                eos_token_id=terminators,
                            )
        else:
            generation_outputs = self.gen_model.generate(
                                    input_ids=input_ids,
                                    max_new_tokens=2048*2,
                                    temperature=0.7,
                                    num_return_sequences=num_return_sequences,
                                    eos_token_id=terminators,
                                )
        decoded_output = self.tokenizer.batch_decode(
            generation_outputs[:, input_ids.size(-1):], skip_special_tokens=True
        )
        return decoded_output


    def generate_without_icv(self, input_ids, num_return_sequences=5):
        """
        Generates text without ICV applied.

        Args:
            input_ids (torch.Tensor): Input IDs tensor.
            num_return_sequences (int, optional): The number of sequences to return. Defaults to 5.

        Returns:
            List[str]: List of generated text.
        """
        decoded_output = self.generate(input_ids, num_return_sequences=num_return_sequences)
        return decoded_output
    
    

    def get_exp_name(self, model_type, dataset_name, start_sample, n_samples, num_trials):
        return f"{model_type.replace('/','-')}_{dataset_name.replace('/','-')}_{start_sample}_{n_samples}_ntrials_{num_trials}"

    def run_evaluation_without_icv(self, start_sample=0, n_samples=200, num_return_sequences=5, num_trials=20):
        """
        Executes the evaluation of the model without applying ICV.

        Args:
            start_sample (int, optional): The index of the first sample to evaluate. Defaults to 0.
            n_samples (int, optional): The total number of samples to evaluate. Defaults to 200.
            num_return_sequences (int, optional): The number of sequences to return for each input. Defaults to 5.
            num_trials (int, optional): The number of trials for the evaluation. Defaults to 20.
        """
        
        exp_name = self.get_exp_name(self.gen_model_args['model_type'], self.ds_args['dataset_name'], start_sample, n_samples, num_trials)
        output_file_path = os.path.join(self.root_path, exp_name + "_Original_Model.json")
        correctness, correctness_v2, total_samples, start_sample = load_previous_results(output_file_path)

        print('=' * 20)
        print('Current experimentoutput file path: ', output_file_path)
        print('=' * 20)

        if n_samples > 0:
            end_sample = start_sample + n_samples
        else:
            end_sample = len(self.dataset)

        with torch.no_grad():
            with open(output_file_path, 'a' if os.path.exists(output_file_path) else 'w') as f:
                for query_id in tqdm(range(start_sample, end_sample), desc="Processing hard query"): 
                    query = self.dataset[query_id]['question']
                    answer = self.dataset[query_id]['answer']

                    if 'R1Qwen' in self.gen_model_args['model_type']:
                        query += "\n\nAssistant: <think>\n"
                        messages_query = [
                        {"role": "user", "content": init_prompt + ' ' + query },
                    ]
                        input_ids = self.tokenizer.apply_chat_template(
                            messages_query, add_generation_prompt=False, return_tensors="pt"
                        ).to(self.gen_model.device)
                    else:
                        messages_query = [
                        {"role": "user", "content": init_prompt + ' ' + query },
                    ]
                        input_ids = self.tokenizer.apply_chat_template(
                            messages_query, add_generation_prompt=True, return_tensors="pt"
                        ).to(self.gen_model.device)

                    generated_sentences = self.generate_without_icv(input_ids, num_return_sequences=1)
                    formatted_answers = self.formatting_final_answers(generated_sentences)  
                     
                    # formating the final answer into Answer: #number to calucalte the accuracy
                    correct, correct_count, answer_majority_index = check_response_accuracy(self.ds_args['dataset_name'], formatted_answers, answer, weights=None)
                    correctness += correct
                    total_samples += 1

                    json.dump({
                        'acc_so_far': correctness / total_samples if total_samples else 0,
                        'query_id': query_id,
                        'query': query,
                        'final_answers': generated_sentences[answer_majority_index],
                        'formatted_answers': formatted_answers[answer_majority_index],
                        'answer': answer,
                        'correct_count': correct_count,
                        'correct': correct,
                        }, f)
                    f.write("\n")
                    f.flush()

        accuracy = correctness / total_samples if total_samples else 0
        return accuracy

if __name__ == "__main__":
    # Mocked global functions from your environment setup:
    # ----------------------------------------------------
    # (Replace these stubs with actual implementations)
    import argparse

    parser = argparse.ArgumentParser(description='ICV Arguments')
    parser.add_argument('--num_trials', type=int, default=1, help='Number of trials for generation')
    parser.add_argument('--model_type', type=str, default='llama-3', help='model to use')
    parser.add_argument('--model_size', type=str, default='8b', help='model size to use')
    parser.add_argument('--dataset', type=str, default='gsm8k', choices=['gsm8k', 'HuggingFaceH4/MATH-500', 'Idavidrein/gpqa'], help='dataset to use')
    parser.add_argument('--split', type=str, default='test', help='split of the dataset to use')
    args = parser.parse_args()

    # Original single-process usage
    gen_model_args = {
        "model_type": args.model_type,
        "model_size": args.model_size,
        "gpus": 1,
        "seed": 42,
        "in_8bit": True
    }

    prm_model_args = {
        "prm_model_name": "RLHFlow/Llama3.1-8B-PRM-Deepseek-Data",
        "prm_device": "auto"
    }

    dataset_args = {
        "dataset_name": args.dataset,
        "split": args.split
    }

    start_sample = 0
    n_samples = -1

    print(torch.cuda.is_available())

    # Initialize generation model and tokenizer
    setup_env(gpu_s=gen_model_args['gpus'], seed=gen_model_args['seed'])

    init_prompt = 'Solve the problem.'
    if args.dataset == 'HuggingFaceH4/MATH-500' and args.model_type == 'R1Qwen':
        init_prompt = 'Solve the problem. Put your final answer within \\boxed{{}}".'
    if args.dataset == 'musr' or args.dataset == 'Idavidrein/gpqa':
        init_prompt = 'Derive Your Final Answer by Choosing the Correct Option for a Multi-Choice Question. Select only one option. Put your final answer (alphabet only) in format of "Answer: selected option".'
    evaluator = Evaluator(init_prompt, dataset_args, prm_model_args, gen_model_args)

    # Run optimization on a subset of the dataset
    accuracy = evaluator.run_evaluation_without_icv(start_sample=start_sample, n_samples=n_samples, num_trials=args.num_trials)
    print(f"Test accuracy: {accuracy:.4f}")