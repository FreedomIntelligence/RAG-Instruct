# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import argparse
import json
import pickle
import time
import glob
from pathlib import Path

import numpy as np
import torch
import pandas as pd
import tqdm
import src.index
import src.contriever
import src.utils
import src.slurm
import src.data
from src.evaluation import calculate_matches
import src.normalize_text

os.environ["TOKENIZERS_PARALLELISM"] = "true"


class Retriever:
    def __init__(self, args, model=None, tokenizer=None) :
        self.args = args
        self.model = model
        self.tokenizer = tokenizer

    def embed_queries(self, args, queries):
        embeddings, batch_question = [], []
        with torch.no_grad():
            for k, q in enumerate(queries):
                if args.lowercase:
                    q = q.lower()
                if args.normalize_text:
                    q = src.normalize_text.normalize(q)
                batch_question.append(q)

                if len(batch_question) == args.per_gpu_batch_size or k == len(queries) - 1:

                    encoded_batch = self.tokenizer.batch_encode_plus(
                        batch_question,
                        return_tensors="pt",
                        max_length=args.question_maxlength,
                        padding=True,
                        truncation=True,
                    )
                    encoded_batch = {k: v.cuda() for k, v in encoded_batch.items()}
                    output = self.model(**encoded_batch)
                    embeddings.append(output.cpu())

                    batch_question = []

        embeddings = torch.cat(embeddings, dim=0)
        # print(f"Questions embeddings shape: {embeddings.size()}")

        return embeddings.numpy()
    

    def embed_queries_demo(self, queries):
        embeddings, batch_question = [], []
        with torch.no_grad():
            for k, q in enumerate(queries):
                batch_question.append(q)

                if len(batch_question) == 16 or k == len(queries) - 1:

                    encoded_batch = self.tokenizer.batch_encode_plus(
                        batch_question,
                        return_tensors="pt",
                        max_length=200,
                        padding=True,
                        truncation=True,
                    )
                    encoded_batch = {k: v.cuda() for k, v in encoded_batch.items()}
                    output = self.model(**encoded_batch)
                    embeddings.append(output.cpu())

                    batch_question = []

        embeddings = torch.cat(embeddings, dim=0)
        print(f"Questions embeddings shape: {embeddings.size()}")

        return embeddings.numpy()

    def index_encoded_data(self, index, embedding_files, indexing_batch_size):
        allids = []
        allembeddings = np.array([])
        for i, file_path in enumerate(embedding_files):
            print(f"Loading file {file_path}")
            with open(file_path, "rb") as fin:
                ids, embeddings = pickle.load(fin)

            allembeddings = np.vstack((allembeddings, embeddings)) if allembeddings.size else embeddings
            allids.extend(ids)
            while allembeddings.shape[0] > indexing_batch_size:
                allembeddings, allids = self.add_embeddings(index, allembeddings, allids, indexing_batch_size)

        while allembeddings.shape[0] > 0:
            allembeddings, allids = self.add_embeddings(index, allembeddings, allids, indexing_batch_size)

        print("Data indexing completed.")


    def add_embeddings(self, index, embeddings, ids, indexing_batch_size):
        end_idx = min(indexing_batch_size, embeddings.shape[0])
        ids_toadd = ids[:end_idx]
        embeddings_toadd = embeddings[:end_idx]
        ids = ids[end_idx:]
        embeddings = embeddings[end_idx:]
        index.index_data(ids_toadd, embeddings_toadd)
        return embeddings, ids


    def add_passages(self, passages, top_passages_and_scores):
        # add passages to original data
        docs = []
        for i, doc_id in enumerate(top_passages_and_scores[0][0]):
            # if doc_id not in passages:
            #     continue
            try:
                item = passages[doc_id]
            except:
                print("error")
                continue
            item["socre"] = float(top_passages_and_scores[0][1][i])
            docs.append(item)
        return docs
    def add_id(self):
        for i, item in enumerate(self.passages):
            item["id"] = str(i)


    def setup_retriever(self):
        print(f"Loading model from: {self.args.model_name_or_path}")
        self.model, self.tokenizer, _ = src.contriever.load_retriever(self.args.model_name_or_path)
        self.model.eval()
        self.model = self.model.cuda()
        if not self.args.no_fp16:
            self.model = self.model.half()

        self.index = src.index.Indexer(self.args.projection_size, self.args.n_subquantizers, self.args.n_bits)

        # index all passages
        input_paths = glob.glob(self.args.passages_embeddings)
        input_paths = sorted(input_paths)
        embeddings_dir = os.path.dirname(input_paths[0])
        index_path = os.path.join(embeddings_dir, "index.faiss")
        if self.args.save_or_load_index and os.path.exists(index_path):
            self.index.deserialize_from(embeddings_dir)
        else:
            print(f"Indexing passages from files {input_paths}")
            start_time_indexing = time.time()
            self.index_encoded_data(self.index, input_paths, self.args.indexing_batch_size)
            print(f"Indexing time: {time.time()-start_time_indexing:.1f} s.")
            if self.args.save_or_load_index:
                self.index.serialize(embeddings_dir)

        # load gpu
        self.index.convert_to_gpu( 0)

        # load passages
        print("loading passages")
        self.passages = src.data.load_passages(self.args.passages)
        if "id" not in self.passages[0]:
            self.add_id()
        
        self.passage_id_map = {x["id"]: x for x in self.passages}
        print("passages have been loaded")

    def search_document(self, query, top_n=10):
        questions_embedding = self.embed_queries(self.args, [query])

        # get top k results
        start_time_retrieval = time.time()
        top_ids_and_scores = self.index.search_knn(questions_embedding, self.args.n_docs)
        # print(f"Search time: {time.time()-start_time_retrieval:.1f} s.")

        return self.add_passages(self.passage_id_map, top_ids_and_scores)[:top_n]
    
    def search_document_demo(self, query, n_docs=10):
        questions_embedding = self.embed_queries_demo([query])

        # get top k results
        start_time_retrieval = time.time()
        top_ids_and_scores = self.index.search_knn(questions_embedding, n_docs)
        print(f"Search time: {time.time()-start_time_retrieval:.1f} s.")

        return self.add_passages(self.passage_id_map, top_ids_and_scores)[:n_docs]

    def setup_retriever_demo(self, model_name_or_path, passages, passages_embeddings, n_docs=5, save_or_load_index=False):
        print(f"Loading model from: {model_name_or_path}")
        self.model, self.tokenizer, _ = src.contriever.load_retriever(model_name_or_path)
        self.model.eval()
        self.model = self.model.cuda()

        self.index = src.index.Indexer(768, 0, 8)

        # index all passages
        input_paths = glob.glob(passages_embeddings)
        input_paths = sorted(input_paths)
        embeddings_dir = os.path.dirname(input_paths[0])
        index_path = os.path.join(embeddings_dir, "index.faiss")
        if save_or_load_index and os.path.exists(index_path):
            self.index.deserialize_from(embeddings_dir)
        else:
            print(f"Indexing passages from files {input_paths}")
            start_time_indexing = time.time()
            self.index_encoded_data(self.index, input_paths, 1000000)
            print(f"Indexing time: {time.time()-start_time_indexing:.1f} s.")

        

        # load passages
        print("loading passages")
        self.passages = src.data.load_passages(passages)
        if "id" not in self.passages[0]:
            self.add_id()
        self.passage_id_map = {x["id"]: x for x in self.passages}
        print("passages have been loaded")

def add_hasanswer(data, hasanswer):
    # add hasanswer to data
    for i, ex in enumerate(data):
        for k, d in enumerate(ex["ctxs"]):
            d["hasanswer"] = hasanswer[i][k]


def load_data(data_path):
    if data_path.endswith(".json"):
        with open(data_path, "r") as fin:
            data = json.load(fin)
    elif data_path.endswith(".jsonl"):
        data = []
        with open(data_path, "r") as fin:
            for k, example in enumerate(fin):
                example = json.loads(example)
                data.append(example)
    return data


def process_item(item, retriever, n_docs):
    if "question" in item:
        item["Question"] = item["question"]
        del item["question"]
    if "answer" in item:
        item["Answer"] = item["answer"]
        del item["answer"]

    if "Options" in item:
        if isinstance(item["Options"], list):
            # If it is a list, ensure all options are strings
            options = [str(option) for option in item["Options"]]
        elif isinstance(item["Options"], dict):
            # If it is a dictionary, extract the values and ensure all options are strings
            options = [str(option) for option in item["Options"].values()]
        else:
            options = []
        if "q*" in item:
            Question = item["q*"] + " " + " ".join(options)
        else:
            Question = item["Question"] + " " + " ".join(options)
    else:
        if "q*" in item:
            Question = item["q*"]
        else:
            Question = item["Question"]

    query = str(Question) 
    retrieval_docs = retriever.search_document(query, n_docs)  # Retrieve documents
    item["retrieval_ctxs"] = retrieval_docs
    return item


def load_items(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    input_data = []
    for key, values in data.items():
        for item in values:
            item['source'] = key  # Add the original key name as a new field to each item
            input_data.append(item)  # Add to the input data list

    return input_data


def main(args):
    retriever = Retriever(args)
    retriever.setup_retriever()
    # print(retriever.search_document(args.query, args.n_docs))
    file_path = args.input_name  # Data file path

    # Choose the reading method based on the file extension
    if args.combine == False:
        if file_path.endswith('.jsonl'):
            with open(file_path, 'r', encoding='utf-8') as f:
                data = [json.loads(line) for line in f]
        elif file_path.endswith('.json'):
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        else:
            raise ValueError("Unsupported file format. Please use a .json or .jsonl file.")
    else:
        data = load_items(file_path)
    new_data = []
    # # n_threads = 64  # Set the number of threads
    # n_threads = 1  # Set the number of threads
    
    for item in tqdm.tqdm(data, total=len(data)):
        new_data.append(process_item(item, retriever, args.n_docs))
    
    with open(args.input_name.split(".")[0] + "_retrieved.json", 'w', encoding='utf-8') as f:
        json.dump(new_data, f, ensure_ascii=False, indent=2)

    with open(args.input_name.split(".")[0] + "_retrieved.jsonl", 'w', encoding='utf-8') as f:
        for item in new_data:
            json_line = json.dumps(item, ensure_ascii=False)
            f.write(json_line + '\n')





if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help=".json file containing question and answers, similar format to reader data",
    )
    parser.add_argument("--passages", type=str, default=None, help="Path to passages (.tsv file)")
    parser.add_argument("--passages_embeddings", type=str, default=None, help="Glob path to encoded passages")
    parser.add_argument(
        "--output_dir", type=str, default=None, help="Results are written to outputdir with data suffix"
    )
    parser.add_argument("--n_docs", type=int, default=100, help="Number of documents to retrieve per questions")
    parser.add_argument(
        "--validation_workers", type=int, default=32, help="Number of parallel processes to validate results"
    )
    parser.add_argument("--per_gpu_batch_size", type=int, default=64, help="Batch size for question encoding")
    parser.add_argument(
        "--save_or_load_index", action="store_true", help="If enabled, save index and load index if it exists"
    )
    parser.add_argument(
        "--model_name_or_path", type=str, help="path to directory containing model weights and config file"
    )
    parser.add_argument("--no_fp16", action="store_true", help="inference in fp32")
    parser.add_argument("--question_maxlength", type=int, default=512, help="Maximum number of tokens in a question")
    parser.add_argument(
        "--indexing_batch_size", type=int, default=1000000, help="Batch size of the number of passages indexed"
    )
    parser.add_argument("--projection_size", type=int, default=768)
    parser.add_argument(
        "--n_subquantizers",
        type=int,
        default=0,
        help="Number of subquantizer used for vector quantization, if 0 flat index is used",
    )
    parser.add_argument("--n_bits", type=int, default=8, help="Number of bits per subquantizer")
    parser.add_argument("--lang", nargs="+")
    parser.add_argument("--dataset", type=str, default="none")
    parser.add_argument("--lowercase", action="store_true", help="lowercase text before encoding")
    parser.add_argument("--normalize_text", action="store_true", help="normalize text")
    parser.add_argument("--input_name", type=str,  help="input_file_name")
    parser.add_argument("--combine", type=bool,  default=False, help="datasets split or combine")

    args = parser.parse_args()
    src.slurm.init_distributed_mode(args)
    main(args)

    
