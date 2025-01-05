import json
import jsonlines
import requests
import os
import tqdm
import random
import threading
from openai import OpenAI

def read_file(file_path):
    if file_path.endswith('.jsonl'):
        return read_jsonl_file(file_path)
    elif file_path.endswith('.json'):
        return read_json(file_path)
    else:
        raise ValueError("Unsupported file format. Please provide a JSON or JSONL file.")

def read_jsonl_file(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data.append(json.loads(line.strip()))
    return data

def read_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)
    
class GPT4:
    def __init__(self, model_name='gpt-4o') -> None:
        self.key_ind = 0
        self.max_wrong_time = 10
        self.model_name = model_name
        self.url = "YOUR_URL"
        self.keys = [['YOUR-KEY', '']]

        assert len(self.keys) > 0, 'No API keys available'
        self.wrong_time = [0] * len(self.keys)
        random.shuffle(self.keys)

    def get_api_key(self):
        self.key_ind = (self.key_ind + 1) % len(self.keys)
        return self.keys[self.key_ind]

    def call(self, content, args={}, showkeys=False):
        api_key, organization = self.get_api_key()
        if showkeys:
            print(api_key, organization)
        if organization == 'None':
            organization = ''

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
            "OpenAI-Organization": organization,
        }

        parameters = {
            "model": self.model_name,
            "messages": [{'role': 'user', 'content': content}],
            **args,
        }

        response = requests.post(self.url, headers=headers, json=parameters)
        response = json.loads(response.content.decode("utf-8"))
        if 'error' in response:
            self.wrong_time[self.key_ind] += 1
            if self.wrong_time[self.key_ind] > self.max_wrong_time:
                print(response)
                print(f'Removing key: {self.keys[self.key_ind]}')
            raise Exception(str(response))
        return response['choices'][0]['message']['content']


def read_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)


def save_to_json(data_item, filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data_item, f, ensure_ascii=False, indent=2)


def save_to_jsonl(data_list, filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w', encoding='utf-8') as f:
        for item in data_list:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


def classify_item(gpt_instance, item, prompt, output_dir, helpful_list, mid_help_list, helpless_list, index_offset):
    question = item['question']
    if "answers" in item:
        answer = item["answers"][0]
    else:
        answer = item["answer"]
    ctxs = item['ctxs'][:10]
    evidences = ["[{}] ".format(i + 1) + ctx["title"] + "\n" + (ctx["text"] if "text" in ctx else ctx["paragraph_text"])
                 for i, ctx in enumerate(ctxs)]
    evidences = "\n".join(evidences)
    input_text = f"Question: {question}\nAnswer: {answer}\nDocuments: {evidences}\n{prompt}"

    output = call_gpt4_with_retry(gpt_instance, input_text)

    if "(1)" in output or "1" in output:
        helpful_list.append(item)
        save_to_json(item, f'{output_dir}/helpful/helpful_{index_offset}.json')
    elif "(2)" in output or "2" in output:
        mid_help_list.append(item)
        save_to_json(item, f'{output_dir}/mid_help/mid_help_{index_offset}.json')
    elif "(3)" in output or "3" in output:
        helpless_list.append(item)
        save_to_json(item, f'{output_dir}/helpless/helpless_{index_offset}.json')


def classify_dataset(gpt_instance, dataset, prompt, output_dir, max_threads=40):
    helpful_list, mid_help_list, helpless_list = [], [], []
    threads = []

    for i, item in tqdm.tqdm(enumerate(dataset)):
        thread = threading.Thread(target=classify_item, args=(
        gpt_instance, item, prompt, output_dir, helpful_list, mid_help_list, helpless_list, i))
        threads.append(thread)
        thread.start()

        if len(threads) >= max_threads:
            for t in threads:
                t.join()
            threads = []

    for t in threads:
        t.join()

    save_to_jsonl(helpful_list, f'{output_dir}/helpful_list.jsonl')
    save_to_jsonl(mid_help_list, f'{output_dir}/mid_help_list.jsonl')
    save_to_jsonl(helpless_list, f'{output_dir}/helpless_list.jsonl')


def call_gpt4_with_retry(gpt_instance, content, max_retries=10):
    attempts = 0
    while attempts < max_retries:
        try:
            return gpt_instance.call(content)
        except Exception as e:
            attempts += 1
            print(f"Attempt {attempts}/{max_retries} failed: {e}")

    raise Exception("Failed to get response after maximum retries.")


# Prompts
PROMPT_SINGLE = '''Based on the question and its answer, along with the provided documents, carefully review the documents to assess their overall usefulness in answering the question. Avoid evaluating each document individually; instead, consider the documents as a whole. Choose the most accurate option based on how much the documents contribute to the answer:
1. Very helpful: The answer is directly provided in the documents.
2. Partially helpful: The documents offer supporting information or clues but do not provide an explicit answer.
3. Not helpful: The documents do not contribute to answering the question.
Please directly respond with only the chosen option (1, 2, or 3).'''  # (Single-hop prompt as provided)

PROMPT_MULTI = '''Based on the question and answer provided, carefully review the given documents and assess their overall usefulness in addressing the question. Avoid evaluating each document individually; instead, consider the documents as a whole. Choose the most accurate option based on how much the documents contribute to the answer:
1. Very helpful: The answer can be directly derived from multiple documents.
2. Partially helpful: The documents offer supporting information or clues but do not provide an explicit answer. It needs further reasoning or more knowledge.
Please directly respond with only the chosen option (1, or 2).'''  # (Multi-hop prompt as provided)


# Main logic
def main(dataset_type='HotpotQA', file_path='data.json'):
    gpt = GPT4(model_name='gpt-4o')

    # Select prompt and set output directory
    if dataset_type == 'TriviaQA':
        prompt = PROMPT_SINGLE
        output_dir = '../split_data/triviaqa'
    elif dataset_type == 'HotpotQA':
        prompt = PROMPT_MULTI
        output_dir = '../split_data/hotpotqa'
    else:
        raise ValueError("Unsupported dataset type. Choose 'TriviaQA' or 'HotpotQA'.")

    dataset = read_file(file_path)
    classify_dataset(gpt, dataset, prompt, output_dir)


# Execute main function
if __name__ == "__main__":
    main(dataset_type='TriviaQA', file_path='')


