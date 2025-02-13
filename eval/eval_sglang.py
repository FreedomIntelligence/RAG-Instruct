import argparse
import numpy as np
from tqdm import tqdm
import argparse
from utils import TASK_INST
import openai
from jinja2 import Template
import os
import json
from transformers import AutoTokenizer
from scorer import score
from jinja2 import Template


def postprocess_output(pred):
    pred = pred.replace("</s>", "")

    if len(pred) > 0 and pred[0] == " ":
        pred = pred[1:]
    return pred

def load_file(input_fp):
    with open(input_fp, 'r') as f:
        data = json.load(f)
    input_data = []
    for k,v in data.items():
        for da in v:
            da['source'] = k
        input_data.extend(v)
    return input_data

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str,
                        default="meta-llama/Llama-2-7b-chat-hf")
    parser.add_argument('--input_file', type=str, required=True)
    parser.add_argument('--retrieval_file', type=str, default=None)
    parser.add_argument('--mode', type=str, default="retrieval")
    parser.add_argument('--model_type', type=str, default="sft")
    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--max_new_tokens', type=int, default=15)
    parser.add_argument('--use_template', action="store_true")
    parser.add_argument('--int8bit', action="store_true")
    parser.add_argument('--metric', type=str)
    parser.add_argument('--top_n', type=int, default=10,
                        help="number of paragraphs to be considered.")
    parser.add_argument('--result_fp', type=str)
    parser.add_argument('--task', type=str)
    parser.add_argument('--prompt_name', type=str, default="prompt_no_input")
    parser.add_argument('--port', type=int, default=30000)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument("--dtype",  type=str, default=None,
                        help="world size to use multiple GPUs.")
    parser.add_argument("--world_size",  type=int, default=1,
                        help="world size to use multiple GPUs.")
    parser.add_argument("--choices",  type=str, default=None,
                        help="space-separated answer candidates")
    parser.add_argument("--instruction",  type=str,
                        default=None, help="task instructions")
    parser.add_argument('--download_dir', type=str, help="specify download dir",
                        default=".cache")
    parser.add_argument('--api_key', type=str, default=None)


    args = parser.parse_args()

    client = openai.Client(
    base_url=f"http://127.0.0.1:{args.port}/v1", api_key="EMPTY")

    if args.use_template:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True, padding_side='left')
        template = Template(tokenizer.chat_template)

    def call_model(prompts, model, max_new_tokens=50, print_example =False):
        temperature = 0.5
        if print_example:
            print("Example:")
            print(prompts[1])
        preds = []
        if args.use_template:
            prompts = [template.render(messages=[{"role": "user", "content": prom}],bos_token= tokenizer.bos_token,add_generation_prompt=True) for prom in prompts]

        response = client.completions.create(
            model="default",
            prompt=prompts,
            temperature=temperature, top_p=0.9, max_tokens=max_new_tokens
        )
        preds = [x.text for x in response.choices]
        postprocessed_preds = [postprocess_output(pred) for pred in preds]
        return postprocessed_preds, preds


    model = None


    input_data = load_file(args.input_file)

    # For baseline scripts, we simply load pre-retrieved documents from `retrieval_file` option.
    if args.mode == "retrieval":
        for id, item in enumerate(input_data):
            if "retrieval_ctxs" in item:
                item["ctxs"] = item["retrieval_ctxs"]
                del item["retrieval_ctxs"]
            retrieval_result = item["ctxs"][:args.top_n]
            evidences = ["[{}] ".format(i+1) + (ctx["text"] if "text" in ctx else ctx["paragraph_text"]) for i, ctx in enumerate(retrieval_result)] # 检索文档名和文档text
            item["paragraph"] = "\n".join(evidences)

    for item in input_data:

        if args.instruction is not None:
            item["instruction"] = args.instruction + \
                "\n\n### Input:\n" + item["instruction"]

        if 'health_claims_processed' in item['source']:
            task_prompt = "Determine whether this statement is True or False:\n{}"
            item['instruction'] = task_prompt.format(item['instruction'])

        if "ConvFin" in item['source']:
            instruction = item["instruction"]
            context = item["context"]
            previous_questions = item["previous_question"]
            
            if len(previous_questions) > 0:
                item["instruction"] = instruction + "\n\n### Context:\n" + context +\
                 "\n\n### Previous Questions and Answers:\n" + "\n".join(previous_questions) + "\n\n### Question:\n" + item["question"]
            else:
                item["instruction"] = instruction + "\n\n### Context:\n" + context +\
                 "\n\n### Previous Questions and Answers:" + "\n\n### Question:\n" + item["question"]
                
        if 'pubmedqa' in item['source']:
            instruction = TASK_INST['pubmedqa']
            context = item["context"]
            item["instruction"] = instruction + "\n\n### Question:\n" + item["question"]
            item["paragraph"] = f'[1] {context}'

        if 'casehold' in item['source']:
            instruction = TASK_INST['casehold']
            holding_1 = item["holding_1"]
            holding_2 = item["holding_2"]
            holding_3 = item["holding_3"]
            holding_4 = item["holding_4"]
            choices = "\n\nholding_1: {0}\nholding_2: {1}\nholding_3: {2}\nholding_4: {3}".format(
                holding_1, holding_2, holding_3, holding_4)
            item["instruction"] = instruction + \
                "\n\n### Input:\n" + item["question"] + choices

        if 'arc_challenge_processed' in item['source']:
            choices = item["choices"]
            answer_labels = {}
            instruction = TASK_INST['arc_c']
            for i in range(len(choices["label"])):
                answer_key = choices["label"][i]
                text = choices["text"][i]
                if answer_key == "1":
                    answer_labels["A"] = text
                if answer_key == "2":
                    answer_labels["B"] = text
                if answer_key == "3":
                    answer_labels["C"] = text
                if answer_key == "4":
                    answer_labels["D"] = text
                if answer_key in ["A", "B", "C", "D"]:
                    answer_labels[answer_key] = text

            if "D" not in answer_labels:
                answer_labels["D"] = ""
            choices = "\nA: {0}\nB: {1}\nC: {2}\nD: {3}".format(
                answer_labels["A"], answer_labels["B"], answer_labels["C"], answer_labels["D"])
            if "E" in answer_labels:
                choices += "\nE: {}".format(answer_labels["E"])
            item["instruction"] = instruction + \
                "\n\n### Input:\n" + item["question"] + choices
            item["answers"] = [item["answerKey"]]
        
        if 'medqa_test_en_retrieved' in item['source']:
            item["golds"]  = item["answer_idx"]
            instruction = TASK_INST['medqa']
            answer_labels = item["options"]
            if "D" not in answer_labels:
                answer_labels["D"] = ""
            choices = "\nA: {0}\nB: {1}\nC: {2}\nD: {3}".format(
                answer_labels["A"], answer_labels["B"], answer_labels["C"], answer_labels["D"])
            if "E" in answer_labels:
                choices += "\nE: {}".format(answer_labels["E"])
            item["instruction"] = instruction + \
                "\n\n### Input:\n" + item["question"] + choices
            item["answers"] = [item["answer_idx"]]

        if 'openbookqa' in item['source']:
            item["golds"]  = item["answer_idx"]
            instruction = TASK_INST['openbookqa']
            answer_labels = item["options"]
            if "D" not in answer_labels:
                answer_labels["D"] = ""
            choices = "\nA: {0}\nB: {1}\nC: {2}\nD: {3}".format(
                answer_labels["A"], answer_labels["B"], answer_labels["C"], answer_labels["D"])
            if "E" in answer_labels:
                choices += "\nE: {}".format(answer_labels["E"])
            item["instruction"] = instruction + \
                "\n\n### Input:\n" + item["question"] + choices
            item["answers"] = [item["answer_idx"]]


    final_results = []

    if args.model_type == 'sft':
        prompt = "### Instruction:\n{}\n\n### Response:\n"
        inner_prompt = """Reference Document:
{paragraph}

Please refer to the document above and answer the following question:
{instruction}"""
    else:
        prompt = "{}"
        inner_prompt = "### Paragraph:\n{paragraph}\n\n### Instruction:\n{instruction}\n\n### Response:\n"
        

    for idx in tqdm(range(len(input_data) // args.batch_size + 1)):
        batch = input_data[idx*args.batch_size:(idx+1)*args.batch_size]
        if len(batch) == 0:
            break
        

        for item in batch:
            item["input_str"] = prompt.format(inner_prompt.format_map(item))

        processed_batch = [ item["input_str"] for item in batch]
    
        if idx == 0:
            print_example = True
        else:
            print_example = False
        preds, _ = call_model(
            processed_batch, model=model, max_new_tokens=args.max_new_tokens, print_example=print_example)

        for j, item in enumerate(batch):
            pred = preds[j]
            item["output"] = pred
            final_results.append(item)

    task_name = os.path.split(args.model_name)[-1] if 'tfmr' not in args.model_name else '-'.join(args.model_name.split('/')[-3:-1] )
    task_name = task_name + '_api'
    if not os.path.exists("output_baselines"):
        os.makedirs("output_baselines")
    with open(f'output_baselines/{task_name}.json','w') as fw:
        json.dump(final_results,fw,ensure_ascii=False,indent=2)

    res,wrong_data,cor_data = score(final_results)
    if not os.path.exists("result_baselines"):
        os.makedirs("result_baselines")

    with open(f'result_baselines/{task_name}.json','w') as fw:
        json.dump(res,fw,ensure_ascii=False,indent=2)
    
    with open(f'output_baselines/wrong_{task_name}.json','w') as fw:
        json.dump(wrong_data,fw,ensure_ascii=False,indent=2)

    with open(f'output_baselines/true_{task_name}.json','w') as fw:
        json.dump(cor_data,fw,ensure_ascii=False,indent=2)

if __name__ == "__main__":
    main()
