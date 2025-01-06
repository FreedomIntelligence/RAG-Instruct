# RAG-Instruct: Boosting LLMs with Diverse Retrieval-Augmented Instructions
<div align="center">
<h3>
  RAG-Instruct
</h3>
</div>

<p align="center">
📃 <a href="assets/paper.pdf" target="_blank">Paper</a> ｜🤗 <a href="https://huggingface.co/FreedomIntelligence/RAG-Instruct-Llama3-3B" target="_blank">RAG-Instruct-Llama3-3B</a> ｜🤗 <a href="https://huggingface.co/FreedomIntelligence/RAG-Instruct-Llama3-8B" target="_blank">RAG-Instruct-Llama3-8B</a> ｜  📚 <a href="https://huggingface.co/datasets/FreedomIntelligence/RAG-Instruct" target="_blank">RAG-Instruct Dataset</a>
</p>


## ⚡ Introduction
Hello! Welcome to the repository for [RAG-Instruct](https://arxiv.org/abs/2501.00353)!

<div align=center>
<img src="assets/RAG-Instruct.png"  width = "90%" alt="RAG-Instruct" align=center/>
</div>


**RAG-Instruct** is a method for generating diverse and high-quality RAG instruction data. It synthesizes instruction datasets based on any source corpus, leveraging the following approaches:

- **Five RAG paradigms**, which represent diverse query-document relationships to enhance model generalization across tasks.
- **Instruction simulation**, which enriches instruction diversity and quality by utilizing the strengths of existing instruction datasets.

Using this approach, we constructed a 40K instruction dataset from Wikipedia, covering a wide range of RAG scenarios and tasks. 
Our RAG-Instruct significantly enhances the RAG ability of LLMs, demonstrating remarkable improvements in RAG performance across various tasks.

| Model                          | WQA (acc) | PQA (acc) | TQA (acc) | OBQA (EM) | Pub (EM) | ARC (EM) | 2WIKI (acc) | HotP (acc) | MSQ (acc) | CFQA (EM) | PubMed (EM) |
|--------------------------------|-----------|-----------|-----------|-----------|----------|----------|-------------|------------|-----------|-----------|-------------|
| Llama3.1-8B + RAG                   | 56.7      | 56.8      | 71.5      | 57.6      | 57.6     | 61.4     | 60.7        | 45.5       | 23.5      | 53.1      | 63.0   |
| Llama3.1-8B + RAG-Instruct     | 61.9     | 62.8      | 73.9      | 77.2      | 56.8     | 70.3     | 66.8       | 45.5       | 19.0      | 53.7      | 73.6        |
| Llama3.1-8B + RAG-Instruct     | 69.7      | 68.4      | 80.0      | 82.4      | 77.2     | 79.6     | 76.8        | 59.6       | 33.7      | 57.3      | 77.0        |



We open-sourced our models, data, and code here.

<!-- ## 💭 Environment
You can create a conda environment by running the command below.
```
pip install -r requirements.txt
``` -->

## 💻 Model
- **Model Access**

|          Model Name                  | Base LLMs       | Link                                                                         |
| -------------------------- | ------------ | ---------------------------------------------------------------------------- |
| **RAG-Instruct-Llama3-3B** | LLaMA-3.2-3B | [HF Link](https://huggingface.co/FreedomIntelligence/RAG-Instruct-Llama3-3B) |
| **RAG-Instruct-Llama3-8B** | LLaMA-3.1-8B | [HF Link](https://huggingface.co/FreedomIntelligence/RAG-Instruct-Llama3-8B) |


- **Deploy**

RAG-Instruct models can be used just like `Llama-3.1-8B-Instruct`. You can deploy it with tools like [vllm](https://github.com/vllm-project/vllm) or [Sglang](https://github.com/sgl-project/sglang),  or perform direct inference:
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load the model and tokenizer
model = AutoModelForCausalLM.from_pretrained("FreedomIntelligence/RAG-Instruct-Llama3-8B",torch_dtype="auto",device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("FreedomIntelligence/RAG-Instruct-Llama3-8B")

# Example input
input_text = """### Paragraph:
[1] structure is at risk from new development...
[2] as Customs and Excise stores...
[3] Powis Street is partly underway...
...

### Instruction:
Which organization is currently using a building in Woolwich that holds historical importance?
"""

# Tokenize and prepare input
messages = [{"role": "user", "content": input_text}]
inputs = tokenizer(tokenizer.apply_chat_template(messages, tokenize=False,add_generation_prompt=True), return_tensors="pt").to(model.device)

# Generate output
outputs = model.generate(**inputs, max_new_tokens=2048)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```


## 📚 Data
We’ve open-sourced a 40K instruction dataset for RAG. Download it here:

| Data                  | Description                                                                                   | Link                                                                                           |
| -------------------------- | ----------------------------------------------------------------- | --------------------------------------------------------------------------------------------- |
| RAG-Instruct (Wikipedia) | Diverse RAG instruction data based on Wikipedia | [Link](https://huggingface.co/datasets/FreedomIntelligence/RAG-Instruct)  |


## 🛠️ Data Construction

We provide scripts to **synthesize a diverse RAG instruction dataset**.

**1. Download Source Documents.**  
We use preprocessed passage data from DPR and embeddings generated with [Contriever-MSMARCO](https://github.com/facebookresearch/contriever) :

- Download the preprocessed passage data:
  ```bash
  cd retrieval_lm
  wget https://dl.fbaipublicfiles.com/dpr/wikipedia_split/psgs_w100.tsv.gz
  ```
  
- Download the generated embeddings:
  ```bash
  wget https://dl.fbaipublicfiles.com/contriever/embeddings/contriever-msmarco/wikipedia_embeddings.tar
  ```

**2. Prepare Exemplar Datasets.**  

We utilize several high-quality datasets as exemplars, including [ShareGPT](https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered), [Alpaca](https://github.com/tatsu-lab/stanford_alpaca), [WizardLM-70K](https://huggingface.co/datasets/WizardLM/WizardLM_evol_instruct_V70K), [Lmsys-chat-1M](https://huggingface.co/datasets/lmsys/lmsys-chat-1m), and [SlimOrca](https://huggingface.co/datasets/Open-Orca/OpenOrca).

To ensure high-quality data, we filtered and sampled these datasets using GPT-4o to extract **knowledge-intensive data** (Q). Using the exemplar data (Q), we retrieve source documents to construct (D*). Specifically, we match the exemplar instructions or questions with source documents by ranking their relevance. For convenience, we provide a processed dataset containing source documents and exemplar data across five RAG scenarios [here](data_gen/examplar_data/data.json).

**3. Synthesize Data with Prompts.**  
Using the retrieved documents (D*) and exemplar data (Q), we synthesize new data points with tailored prompts to create diverse and high-quality instruction-following datasets.

```bash
cd data_gen
python generate_data.py \
    --data_path examplar_data/data.json \
    --max_workers 16 \
    --save_dir ./output_data/RAG-Instruct.json
```

**4. Run Retriever**  
Before training, we need to perform retrieval on the synthesized RAG-Instruct dataset. For each data entry, we ensure that the retrieval documents includes all source documents (D*) and supplement them with enough unrelated documents (D-) to total 10 documents.
We use preprocessed passage data from DPR and embeddings generated with [Contriever](https://github.com/facebookresearch/contriever). To retrieve noisy documents (D-), use the following command:

```bash
cd retrieval_lm
python passage_retrieval.py \
    --model_name_or_path facebook/contriever-msmarco \
    --passages psgs_w100.tsv \
    --passages_embeddings "wikipedia_embeddings/*" \
    --input_name RAG_INSTRCT_DATA_PATH \
    --output_dir YOUR_OUTPUT_FILE \
    --n_docs 250
```

`RAG_INSTRUCT_DATA_PATH` is the final location of the synthesized `RAG-Instruct.json` file. The input file must be in `json` or `jsonl` format. Each instance should include either a `question` or `instruction` field, which will be used as the query during retrieval. 

Next, we sample documents ranked beyond the top 200 as (D-) and get the final training data. 

## 🚀 Training

**Fine-tuning with RAG-Instruct**

You can fine-tune your large model using the `RAG-Instruct` dataset to significantly boost RAG capabilities. Use the following code:

```bash
accelerate launch --config_file ./configs/sft.yaml \
    --num_processes 8  \
    --num_machines 1 \
    --machine_rank 0 \
    --deepspeed_multinode_launcher standard train_rag_sft.py \
    --experiment_name RAG-Instruct-training \
    --model_path meta-llama/Llama-3.1-8B-Instruct \
    --data_path FreedomIntelligence/RAG-Instruct \
    --max_seq_len 4096 \
    --learning_rate 5e-6 \
    --train_bsz_per_gpu 2 \
    --gradient_accumulation_steps 16 \
    --output_dir ./ckpts \
    --log_dir ./train_logs \
    --n_epochs 3 \
    --gradient_checkpointing
```


## 🧐 Evaluation
1. You first need to install [Sglang](https://github.com/sgl-project/sglang). After installation, deploy the model you want to test using Sglang with the following command:
```bash
log_num=0
model_name="FreedomIntelligence/RAG-Instruct-Llama3-3B" # Path to the model you are deploying
port=21${log_num}35
CUDA_VISIBLE_DEVICES=0  python -m sglang.launch_server --model-path $model_name --port $port --mem-fraction-static 0.8 --dp 1 --tp 1  > sglang${log_num}.log 2>&1 &
```
2. Wait for the model to be deployed. After deployment, you can run the following code for evaluation. 
```bash
model_name="FreedomIntelligence/RAG-Instruct-Llama3-3B" # Path to the model you are deploying
python eval/eval_sglang.py --model_name $model_name --input_file eval/data/eval_data.json --port $port --max_new_tokens 500  
```
Here, we provide the evaluation example using the PopQA dataset in the file `eval/data/eval_data.json`. For other evaluation datasets, please first use the retriever to retrieve (You can refer to the retriever code in the training section), and then use the above script for evaluation.

3. After completing the evaluation, run the following code to stop the Sglang service and release GPU memory.
```bash
bash evaluation/kill_sglang_server.sh
```
The evaluation code above can be used to test most models supported by Sglang.


## 📖 Citation
```
@misc{liu2024raginstructboostingllmsdiverse,
      title={RAG-Instruct: Boosting LLMs with Diverse Retrieval-Augmented Instructions}, 
      author={Wanlong Liu and Junying Chen and Ke Ji and Li Zhou and Wenyu Chen and Benyou Wang},
      year={2024},
      eprint={2501.00353},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2501.00353}, 
}
```
