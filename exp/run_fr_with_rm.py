import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import json
import os
from common import setup_env
from models import build_tokenizer, build_model
from tasks import load_task 
from utils.llm_layers import add_icv_layers, remove_icv_layers
from utils.data_utils import load_and_prepare_dataset

from tqdm import tqdm
from utils.tools import tokenize_each_demonstration, apply_template
from utils.eval_utils import obtain_majority_answers, check_response_accuracy, load_formatting_prompt, load_previous_results, extract_answer
from utils.score_outputs import get_output_win_rate

from utils.score_outputs import get_prm_score


from transformers import logging
logging.set_verbosity_error()

class Evaluator:
    def __init__(self, args, init_prompt, ds_args, prm_model_args, gen_model_args, root_path='./records'):
        """
        Evaluator class to evaluate the performance of a model on a dataset with or wihout latent shifting.

        Args:
            args (dict): Dictionary containing arguments such as num_trials, alpha_mode, model_type, model_size, dataset, split.
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
        self.alpha_mode = args.alpha_mode
        # Initialize PRM tokenizer and model
        self.tokenizer = build_tokenizer(
            gen_model_args['model_type'], gen_model_args['model_size'], padding_side="right"
        )
        self.gen_model = build_model(
            gen_model_args['model_type'], gen_model_args['model_size'], gen_model_args['in_8bit']
        )

        if 'R1' in gen_model_args['model_type']:
            self.eval_model = build_model(
                'Qwen2.5', '7b', False
            )
            self.eval_tokenizer = build_tokenizer(
                'Qwen2.5', '7b', padding_side="left"
            )
        else:
            self.eval_model = self.gen_model
            self.eval_tokenizer = self.tokenizer

        if 'llama' in gen_model_args['model_type']:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        torch.autograd.set_grad_enabled(False)

        self.formatting_prompt = load_formatting_prompt(ds_args['dataset_name'])

        if 'musr' in ds_args['dataset_name']:
            dataset_demo = [{'question': item['final_question'], 'answer': item['answer']} for item in self.dataset]
        else:
            dataset_demo = [{'question': item['question'], 'answer': item['answer']} for item in self.dataset]
        # Apply templates to the demo dataset
        demos_with_templates = apply_template(dataset_demo, self.tokenizer)
        self.icvs_to_shift = self.prepare_icv(demos_with_templates, rank=1, n=256)
        
        if prm_model_args != None:
            self.prm_tokenizer = AutoTokenizer.from_pretrained(prm_model_args['prm_model_name'])
            self.prm_model = AutoModelForCausalLM.from_pretrained(
                prm_model_args['prm_model_name'], device_map=prm_model_args['prm_device'], torch_dtype=torch.bfloat16
            ).eval()
            
            self.prm_tokenizer.padding_side = "right"
            self.prm_tokenizer.pad_token = self.prm_tokenizer.eos_token
            self.prm_model.config.pad_token_id = self.prm_model.config.eos_token_id
        else:
            self.prm_model = None

        
    def prepare_icv(self, demos_with_templates, n=256, rank=1):
        """
        Prepares the ICV (Inverse Concept Variance) by sampling and tokenizing data.
        """
        # Load TaskHandler and apply templates
        TaskHandler = load_task('demo')
        task_agent = TaskHandler('default')
        task_agent.set_seed(0)

        demos_with_templates = demos_with_templates[0:n]
        # Ensure no ICV layers remain in the model
        try:
            while True:
                remove_icv_layers(self.gen_model)
        except Exception:
            pass
        
        print('==================calculating icv==================')
        icv_cot, _ = task_agent.obtain_icv(
            self.gen_model,
            tokenize_each_demonstration(demos_with_templates, self.tokenizer, prefix=("", "")),
            rank=rank
        )

        icv_cot = icv_cot[1:]

        return torch.stack([icv_cot], dim=1) 
    
    def get_exp_name(self, model_type, dataset_name, start_sample, n_samples, icv_mode):
        return f"{model_type.replace('/','-')}_{dataset_name.replace('/','-')}_{start_sample}_{n_samples}_icv_type_{icv_mode}"
        
    
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
                                    max_new_tokens=2048,
                                    temperature=0.7,
                                    num_return_sequences=num_return_sequences,
                                    eos_token_id=terminators,
                                )
        decoded_output = self.tokenizer.batch_decode(
            generation_outputs[:, input_ids.size(-1):], skip_special_tokens=True
        )
        return decoded_output
    
    def clean_icv_layers(self):
        try:
            while True:
                remove_icv_layers(self.gen_model)
        except:
            pass

    def generate_without_icv(self, input_ids, num_return_sequences=5):
        """
        Generates text without ICV applied.

        Args:
            input_ids (torch.Tensor): Input IDs tensor.
            num_return_sequences (int, optional): The number of sequences to return. Defaults to 5.

        Returns:
            List[str]: List of generated text.
        """
        self.clean_icv_layers()
        decoded_output = self.generate(input_ids, num_return_sequences=num_return_sequences)
        return decoded_output
    
    def generate_with_icv(self, input_ids, alpha, num_return_sequences=5):
        """
        Generates text with ICV applied.

        Args:
            input_ids (torch.Tensor): Input IDs tensor.
            alpha (float or list): Alpha value(s) for ICV.
        """

        self.clean_icv_layers()
        if not isinstance(alpha, list):
            alpha = [alpha]
        add_icv_layers(self.gen_model, self.icvs_to_shift.cuda(), alpha)
        decoded_output = self.generate(input_ids, num_return_sequences=num_return_sequences)
        return decoded_output

    def formatting_final_answers(self, eval_model, eval_tokenizer, query, best_generation, formatting_prompt):
        """
        Formats the final answers from the generated text.

        Args:
            query (str): The original query.
            best_generation (list): List of generated text.
        """
        final_answers = []
        for response_each in best_generation:
            messages_query_final_answer = [
            {"role": "system", "content": "You job is to extract the final short answer from the more detailed answer."},
            {"role": "user", "content": formatting_prompt.format(query=query, reasoning=response_each)},
        ]
        
            input_ids_final_answer = eval_tokenizer.apply_chat_template(
                messages_query_final_answer, add_generation_prompt=True, return_tensors="pt"
            ).to(eval_model.device)

            current_data_final = eval_model.generate(input_ids_final_answer, num_return_sequences=1)[0]
            current_data_final = eval_tokenizer.decode(current_data_final[len(input_ids_final_answer[0]):], skip_special_tokens=True)
            if 'gsm8k' in self.ds_args['dataset_name']:
                current_data_final = current_data_final.replace(',', '')
            final_answers.append(current_data_final)
        return final_answers

    def get_exp_name(self, model_type, dataset_name, start_sample, n_samples, num_trials, alpha_mode, alpha_a, alpha_b):
        return f"{model_type.replace('/','-')}_{dataset_name.replace('/','-')}_{start_sample}_{n_samples}_ntrials_{num_trials}_alpha_mode_{alpha_mode}_alpha_a_{alpha_a}_alpha_b_{alpha_b}"

    def run_evaluation(self, start_sample=0, n_samples=200, num_return_sequences=5, num_trials=20, alpha_a = 0, alpha_b = 0.15):
        """
        Executes the evaluation of the model without applying ICV.

        Args:
            start_sample (int, optional): The index of the first sample to evaluate. Defaults to 0.
            n_samples (int, optional): The total number of samples to evaluate. Defaults to 200.
            num_return_sequences (int, optional): The number of sequences to return for each input. Defaults to 5.
            num_trials (int, optional): The number of trials for the evaluation. Defaults to 20.
            alpha_a (float, optional): The lower bound of the alpha range. Defaults to 0.
            alpha_b (float, optional): The upper bound of the alpha range. Defaults to 0.15.
        """
        
        exp_name = self.get_exp_name(self.gen_model_args['model_type'], self.ds_args['dataset_name'], start_sample, n_samples, num_trials, self.alpha_mode, alpha_a, alpha_b)
        output_file_path = os.path.join(self.root_path, exp_name + "_FR_reward.json")
        print('=' * 20)
        print('Current experimentoutput file path: ', output_file_path)
        print('=' * 20)
        correctness, correctness_v2, total_samples, start_sample = load_previous_results(output_file_path)

        if n_samples > 0:
            end_sample = start_sample + n_samples
        else:
            end_sample = len(self.dataset)

        with torch.no_grad():
            with open(output_file_path, 'a' if os.path.exists(output_file_path) else 'w') as f:
                for query_id in tqdm(range(start_sample, end_sample), desc="Processing hard query"): 
                    query = self.dataset[query_id]['question']
                    answer = self.dataset[query_id]['answer']

                    messages_query = [
                        {"role": "user", "content": init_prompt + ' ' + query},
                    ]

                    input_ids = self.tokenizer.apply_chat_template(
                        messages_query, add_generation_prompt=True, return_tensors="pt"
                    ).to(self.gen_model.device)

                    ans = []
                    ans_solutions = []
                    ans_scores = []
                    all_scores = []
                    ans_all = []
                    

                    for _ in tqdm(range(num_trials), desc="Processing trials"):
                        if self.alpha_mode == 'uniform':
                            alpha = np.random.uniform(alpha_a, alpha_b)
                        elif self.alpha_mode == 'normal':
                            alpha = np.random.normal((alpha_a + alpha_b)*0.5, (alpha_b - alpha_a)*0.25)
                        elif self.alpha_mode == 'zero':
                            alpha = 0

                        generated_sentences = self.generate_with_icv(input_ids, alpha, num_return_sequences=num_return_sequences)
                        formatted_answers = self.formatting_final_answers(self.eval_model, self.eval_tokenizer, query, generated_sentences, self.formatting_prompt)
                        # Create a list of answer frequencies
                        if num_trials > 1:
                            scores = get_prm_score(query, generated_sentences, self.prm_model, self.prm_tokenizer)
                        else:
                            scores = [1]*len(generated_sentences)
                        ans_all.extend(formatted_answers)
                        all_scores.extend(scores)
                        


                        _, ans_majority_index = obtain_majority_answers(self.ds_args['dataset_name'], formatted_answers, weights=scores)
                        ans.append(formatted_answers[ans_majority_index])
                        
                        ans_solutions.append(generated_sentences[ans_majority_index])
                        ans_scores.append(scores[ans_majority_index])
                     
                    # Check if ans is a list of lists and reshape to a flat list if needed
                    if ans and isinstance(ans[0], list):
                        # Flatten the list of lists into a single list
                        ans = [item for sublist in ans for item in sublist]   
                                              
                        
                    # formating the final answer into Answer: #number to calucalte the accuracy
                    binary_scores = [1 if score == max(ans_scores) else 0 for score in all_scores]
                    correct, correct_count, answer_majority_index = check_response_accuracy(self.ds_args['dataset_name'], ans, answer, weights=ans_scores)
                    correct_v2, _, _ = check_response_accuracy(self.ds_args['dataset_name'], ans_all, answer, weights=binary_scores)
                    correctness += correct
                    correctness_v2 += correct_v2
                    total_samples += 1

                    json.dump({
                        'acc_so_far': correctness / total_samples if total_samples else 0,
                        'acc_so_far_v2': correctness_v2 / total_samples if total_samples else 0,
                        'query_id': query_id,
                        'query': query,
                        'generations': ans_solutions[answer_majority_index],
                        'final_answers': ans[answer_majority_index],
                        'predictions': ans,
                        'all_predictions': ans_all,
                        'scores': ans_scores,
                        'all_scores': all_scores,
                        'answer': answer,
                        'correct_count': correct_count,
                        'correct': correct,
                        'alpha': 0,
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
    parser.add_argument('--num_trials', type=int, default=20, help='Number of trials for generation')
    parser.add_argument('--model_type', type=str, default='llama-3', help='model to use')
    parser.add_argument('--model_size', type=str, default='8b', help='model size to use')
    parser.add_argument('--dataset', type=str, default='gsm8k', choices=['gsm8k', 'HuggingFaceH4/MATH-500', 'Idavidrein/gpqa', 'mmlu', 'aime2024', 'musr'], help='dataset to use')
    parser.add_argument('--split', type=str, default='test', help='split of the dataset to use')
    parser.add_argument('--alpha_mode', type=str, default='uniform', help='uniform or normal')
    parser.add_argument('--root_path', type=str, default='./records/rewarded_majority_vote', help='root path to save the records')
    parser.add_argument('--num_samples', type=int, default=-1, help='number of samples to use, if -1, use all samples')
    parser.add_argument('--alpha_a', type=float, default=0, help='lower bound of the alpha range')
    parser.add_argument('--alpha_b', type=float, default=0.15, help='upper bound of the alpha range')

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
    print(torch.cuda.is_available())

    # Initialize generation model and tokenizer
    setup_env(gpu_s=gen_model_args['gpus'], seed=gen_model_args['seed'])

    init_prompt = 'Solve the problem.'
    if args.dataset == 'musr':
        init_prompt = ''# if args.dataset == 'HuggingFaceH4/MATH-500' and args.model_type == 'R1Qwen':
    #     init_prompt = 'Solve the problem. Put your final answer within \\boxed{{}}.'
    # if args.dataset == 'mmlu' and args.model_type == 'llama-3':
    #     init_prompt = 'Answer the question with A/B/C/D in the end.'
    evaluator = Evaluator(args, init_prompt, dataset_args, prm_model_args, gen_model_args, root_path=args.root_path)

    # Run optimization on a subset of the dataset
    accuracy = evaluator.run_evaluation(start_sample=start_sample, n_samples=args.num_samples, num_trials=args.num_trials, alpha_a = args.alpha_a, alpha_b = args.alpha_b)
    print(f"Test accuracy: {accuracy:.4f}")