import torch
import numpy as np
from utils.eval_utils import extract_answer

def get_prm_score(query, output, prm_model, prm_tokenizer):
        """
        Calculates the reward score for the generated text.

        Args:
            query (str): The original query.
            output (list): List of generated text.
            prm_model (torch.nn.Module): The PRM model.
            prm_tokenizer (torch.nn.Module): The PRM tokenizer.
        """

        if output is None or sum(len(out) for out in output) / len(output) <= 10:
            return [0.0]

        scores_all = []
        if not isinstance(output, list):
            output = [output]

        prm_model.to('cuda')

        for out in output:
            plus_tag_id = prm_tokenizer.encode('+')[-1]
            minus_tag_id = prm_tokenizer.encode('-')[-1]
            candidate_tokens = [plus_tag_id, minus_tag_id]
            text = query + '  ' + out
            conversation = [
                {"content": text, "role": "user"},
                {"content": "+", "role": "assistant"}
            ]

            input_ids_prm = prm_tokenizer.apply_chat_template(
                conversation, return_tensors="pt"
            ).to('cuda')

            with torch.no_grad():
                logits = prm_model(input_ids_prm).logits[:, -3, candidate_tokens]
                scores = logits.softmax(dim=-1)[:, 0]
                scores_all.append(scores[0].detach().to('cpu', dtype=torch.float32).item())
        prm_model.to('cpu')

        return scores_all


def get_output_win_rate(model, tokenizer, query, generated_sentences, formatted_answers, dataset_name):
        """
        Compares each solution with every other solution in a pairwise manner.

        Args:
            model (torch.nn.Module): The model to use.
            tokenizer (torch.nn.Module): The tokenizer to use.
            query (str): The original query.
            generated_sentences (list): List of generated solutions.
            formatted_answers (list): List of formatted answers in string format.
            dataset_name (str): The name of the dataset.
        Returns:
            wins (list): List containing the number of wins for each solution.
            win_output (list): List containing the winning output.
            win_answer (str): The winning answer.
            unique_wins (dict): Dictionary containing the number of wins for each unique answer.
        """
        extracted_answers = extract_answer(dataset_name, formatted_answers) 

        # Extract unique answers
        unique_answers = {}
        for i, answer in enumerate(extracted_answers):
            if answer not in unique_answers:
                unique_answers[answer] = []
            unique_answers[answer].append(i)

        # Initialize wins for each unique answer
        unique_wins = {answer: 0 for answer in unique_answers}

        # Compare only unique answers
        unique_answer_list = list(unique_answers.keys())
        num_unique_solutions = len(unique_answer_list)

        for i in range(num_unique_solutions):
            for j in range(num_unique_solutions):
                if i != j:
                    # Use the original generated sentences for comparison
                    solution_1 = generated_sentences[unique_answers[unique_answer_list[i]][0]]
                    solution_2 = generated_sentences[unique_answers[unique_answer_list[j]][0]]

                    prompt_for_solution_score = f"""
Here are two solutions to the a question. You must determine which one is more likely to be correct. 
Please think extremely carefully. Do not leap to conclusions. You should find out where the solutions disagree, trace them back to the source of their disagreement, and figure out which one is correct and more consistent with the original question.

The original question:
{query}

Solution 1:
{solution_1}
Solution 2:
{solution_2}

Think carefully. In the end, your should summarize your response in a JSON format:
{{
"better_solution": 1 or 2
}}
                    """
                    input_ids_solution_score = tokenizer.apply_chat_template(
                        [{"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": prompt_for_solution_score}], add_generation_prompt=True, return_tensors="pt"
                    ).to(model.device)
                    solution_score = model.generate(input_ids_solution_score, do_sample=True, max_new_tokens=1024,num_return_sequences=10)
                    solution_score = [tokenizer.decode(solution_score[len(input_ids_solution_score[0]):], skip_special_tokens=True) for solution_score in solution_score]
                    
                    # Count votes for solution 1 and solution 2
                    solution_1_votes = sum(1 for score in solution_score if "\"better_solution\": 1" in score)
                    solution_2_votes = sum(1 for score in solution_score if "\"better_solution\": 2" in score)

                    # Apply majority voting
                    if solution_1_votes > solution_2_votes:
                        unique_wins[unique_answer_list[i]] += 1
                    elif solution_2_votes > solution_1_votes:
                        unique_wins[unique_answer_list[j]] += 1
                    # If tied, no increment is applied

        # Assign win rates to all outputs based on their unique answer
        wins = [0] * len(generated_sentences)
        for answer, indices in unique_answers.items():
            for index in indices:
                wins[index] = unique_wins[answer]
        
        win_indices = np.where(np.array(wins) == max(wins))[0]
        win_output = np.array(generated_sentences)[win_indices]
        win_answer = extracted_answers[np.argmax(wins)]
        return wins, win_output.tolist(), win_answer, unique_wins