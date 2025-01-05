import json
from concurrent.futures import ThreadPoolExecutor
import random
import requests
import os


gpt_check_prompt = '''{question}\n Please determine if retrieving external information would help answer the above question. If it helps, answer "True", otherwise answer "False".'''

class GPT4:
    def __init__(self, model_name='gpt-4-turbo') -> None:
        self.key_ind = 0
        self.max_wrong_time = 10
        self.model_name = model_name
        self.url = "your-url"
        self.keys = [['your-key','']]

        assert len(self.keys) > 0, 'have no key'
        self.wrong_time = [0] * len(self.keys)
        print(f'keys: {self.keys}')
        print(f'use model of {self.model_name}')

    def init_api_keys(self):
        self.keys = []
        with open('gpt_key.txt', encoding="utf-8", mode="r") as fr:
            for l in fr:
                cols = l.split('---')
                if len(cols[0]) < 45 or len(cols[0]) > 55:
                    continue
                if len(cols) == 1:
                    cols.append('None')
                self.keys.append((cols[0], cols[1]))
        assert len(self.keys) > 0, 'have no key'
        print(f'keys: {self.keys}')
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

        response = requests.post(
            self.url,
            headers=headers,
            json=parameters
        )
        response = json.loads(response.content.decode("utf-8"))
        if 'error' in response:
            self.wrong_time[self.key_ind] += 1
            if self.wrong_time[self.key_ind] > self.max_wrong_time:
                print(response)
                print(f'del {self.keys[self.key_ind]}')
                # del self.keys[self.key_ind]
                # del self.wrong_time[self.key_ind]
            assert False, str(response)
        return response['choices'][0]['message']['content']

    def test(self):
        for _ in range(len(self.keys)):
            try:
                print(self.call('你好', showkeys=True))
            except Exception as e:
                print(e)

    # @retry(wait_fixed=500, stop_max_attempt_number=5)
    def retry_call(self, content, args={"max_tokens": 4096}):
        return self.call(content, args)
    

def call_gpt4_with_retry(gpt_instance, content, max_retries=10):
    attempts = 0
    while attempts < max_retries:
        try:
            output = gpt_instance.call(content)
            if 'error' in output:
                raise Exception(f"API Error: {output['error']}")
            else:
                return output
        except:
            attempts += 1
            print(f"Attempt {attempts}/{max_retries} failed: generation error")
    
    raise Exception("Failed to get response after maximum retries.")


def call_gpt4_with_retry_with_json(gpt_instance, content, max_retries=10):
    attempts = 0
    while attempts < max_retries:
        output = gpt_instance.call(content).strip('```json\n')
        try:
            output = json.loads(output)
            if 'error' in output:
                raise Exception(f"API Error: {output['error']}")
            else:
                return output

        except json.JSONDecodeError:
            attempts += 1
            print(f"Attempt {attempts}/{max_retries} failed: JSON decode error. Retrying...")
    
    raise Exception("Failed to parse JSON response after maximum retries.")


def process_data_task(data_task, gpt_check_prompt, target_num, gpt):
    data_final = []
    # Open and read the data file
    with open(data_task, "r", encoding="utf-8") as f:
        data = json.load(f)
        for i, item in enumerate(data):
            question = item['question']
            # Construct GPT input
            gpt_input = gpt_check_prompt.format(question=question)
            output = call_gpt4_with_retry(gpt, gpt_input)
            # Check GPT output
            if "True" in output or "true" in output:
                data_final.append(item)
                if len(data_final) >= target_num:
                    print(f"Task {data_task}: Reached target num at idx {i}")
                    # Save data to a JSON file and modify the file name
                    save_data(data_final, data_task)
                    break
    # Save data if target_num is not reached
    if len(data_final) > 0:
        print(f"Task {data_task}: NOT Reached target num at idx {i}")
        save_data(data_final, data_task)

# Define a function to save the processed data
def save_data(data_final, data_task):
    # Generate a new file name by appending gpt_processed
    base_name = os.path.basename(data_task)
    new_file_name = base_name.replace('.json', '_gpt_processed.json')
    # Save as a new JSON file
    with open(new_file_name, "w", encoding="utf-8") as outfile:
        json.dump(data_final, outfile, ensure_ascii=False, indent=4)
    print(f"Saved processed data to {new_file_name}")

# Main function, process all tasks using multithreading
def main(data_list, gpt_check_prompt, target_num, gpt):
    with ThreadPoolExecutor(max_workers=len(data_list)) as executor:
        futures = [executor.submit(process_data_task, data_task, gpt_check_prompt, target_num, gpt) for data_task in data_list]
        # Wait for all threads to complete
        for future in futures:
            future.result()

# Example usage
data_list = [
    './data_normal/EvolInstruct_70k_processed.json', 
    './data_normal/lmsys_processed.json', 
    './data_normal/platypus_processed.json', 
    './data_normal/ShareGPT_V3_processed.json', 
    './data_normal/slim_ocar_processed.json'
]

target_num = 3000
gpt = GPT4("gpt-4o")
# Call the main function
main(data_list, gpt_check_prompt, target_num, gpt)
