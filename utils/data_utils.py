from datasets import load_dataset

def load_and_prepare_dataset(ds_args):
    """
    Loads and prepares the dataset based on the provided arguments.

    Args:
        ds_args (dict): Dictionary containing dataset arguments such as dataset name and split.

    Returns:
        Dataset: The loaded and prepared dataset.
    """
    if ds_args['dataset_name'] == 'Idavidrein/gpqa':
        import random
        ds = load_dataset(ds_args['dataset_name'], "gpqa_diamond")['train']
        all_data = []
        for i in range(len(ds)):
            correct_answer = ds[i]['Correct Answer'].strip()
            incorrect_answers = []
            incorrect_answers.append(ds[i]['Incorrect Answer 1'].strip())
            incorrect_answers.append(ds[i]['Incorrect Answer 2'].strip())
            incorrect_answers.append(ds[i]['Incorrect Answer 3'].strip())
            # Get shuffled choices 
            shuffled_choices = incorrect_answers + [correct_answer]
            random.shuffle(shuffled_choices)

            # Add Options to the question
            question = ds[i]['Question']
            question += f"\n\nOptions:\n"
            for i, choice in enumerate(shuffled_choices):
                question += f"{chr(65 + i)}. {choice}\n"
            correct_choice = chr(65 + shuffled_choices.index(correct_answer))
            all_data.append({
                "question": question,
                "answer": correct_choice,
            })

                # if num_tokens < -1:
                #     question = f"{question}"+"\n\nLet's think step by step and output the final answer (eg, A, B, C, D) within \\boxed{}." + f" Think for maximum {num_tokens} tokens."
                # else:
                #     question = f"{question}"+"\n\nLet's think step by step and output the final answer (eg, A, B, C, D) within \\boxed{}." + f" Think for {num_tokens} tokens."

        return all_data
    if ds_args['dataset_name'] == 'mmlu':
        import random
        ds = load_dataset("edinburgh-dawg/mmlu-redux", 'professional_law', split="test")
        
        all_data = []
        for i in range(len(ds)):
            correct_answer = ds[i]['choices'][ds[i]['answer']].strip()
            # Get shuffled choices 
            shuffled_choices = ds[i]['choices']
            random.shuffle(shuffled_choices)
            shuffled_choices = [choice.strip() for choice in shuffled_choices]

            # Add Options to the question
            question = ds[i]['question']
            question += f"\n\nOptions:\n"
            for i, choice in enumerate(shuffled_choices):
                question += f"{chr(65 + i)}. {choice}\n"
            correct_choice = chr(65 + shuffled_choices.index(correct_answer))
            all_data.append({
                "question": question,
                "answer": correct_choice,
            })
            
                # if num_tokens < -1:
                #     question = f"{question}"+"\n\nLet's think step by step and output the final answer (eg, A, B, C, D) within \\boxed{}." + f" Think for maximum {num_tokens} tokens."
                # else:
                #     question = f"{question}"+"\n\nLet's think step by step and output the final answer (eg, A, B, C, D) within \\boxed{}." + f" Think for {num_tokens} tokens."

        return all_data
    
    if ds_args['dataset_name'] == 'musr':
        import random
        ds = load_dataset("TAUR-Lab/MuSR", split="murder_mysteries")
        
        all_data = []
        for i in range(len(ds)):
            # Convert string representation of list to actual list
            choices = eval(ds[i]['choices'])
            correct_answer = choices[ds[i]['answer_index']].strip()
            # Get shuffled choices 
            shuffled_choices = choices
            
            random.shuffle(shuffled_choices)
            shuffled_choices = [choice.strip() for choice in shuffled_choices]

            # Add Options to the question
            question = "Narrative:\n"
            question += ds[i]['narrative'] + "\n\n"
            question += "Question:\n"
            question += ds[i]['question'] + "\n\n"
            question += "Options:\n"

            for i, choice in enumerate(shuffled_choices):
                question += f"{chr(65 + i)}. {choice}\n"
            correct_choice = chr(65 + shuffled_choices.index(correct_answer))
            all_data.append({
                "question": question,
                "answer": correct_choice,
                "final_question": ds[i]['question'],
            })
        return all_data
    
    
    if ds_args['dataset_name'] == 'aime2024':
        dataset = load_dataset("Maxwell-Jia/AIME_2024", 'default')['train']
        dataset = dataset.map(lambda data: {
            'question': data['Problem'],
            'solution': data['Solution'],
            'answer': f"Answer: {data['Answer']}"
        })
        return dataset
        
    
    try:
        dataset = load_dataset(ds_args['dataset_name'], "main")[ds_args['split']]
    except ValueError:
        dataset = load_dataset(ds_args['dataset_name'])[ds_args['split']]
    
    if ds_args['dataset_name'] == 'HuggingFaceH4/MATH-500':
        dataset = dataset.map(lambda data: {
            'question': data['problem']
        })
    return dataset