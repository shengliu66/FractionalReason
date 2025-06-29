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

        return all_data
    
    try:
        dataset = load_dataset(ds_args['dataset_name'], "main")[ds_args['split']]
    except ValueError:
        dataset = load_dataset(ds_args['dataset_name'])[ds_args['split']]
    
    if ds_args['dataset_name'] == 'HuggingFaceH4/MATH-500':
        dataset = dataset.map(lambda data: {
            'question': data['problem']
        })
    
    return dataset