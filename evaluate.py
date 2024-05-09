
import json
import numpy as np
import pandas as pd
from yaml import safe_load
import itertools
import tqdm

global result
script = """
{function}
global result
result = sol()
"""

def load_SVAMP(svamp_path):
    dataset = json.load(open(svamp_path))
    return [
        {
            "statement": problem["Body"] + ". " + problem["Question"],
            "solution": problem["Answer"],
        }
        for problem in dataset
    ]
def load_GSM(path):
    dataset = safe_load(open(path))
    return [
        {
            "statement": problem["text"],
            "solution": float(problem["response"].replace(",",""))
        }
        for problem in dataset
    ]

def _eval_file(func, folder, i, k, solution, dataset="svamp"):
    filename = folder + f"/{dataset}-{i:04d}-{k:02d}.py"
    compiles = True

    print("evaluating {}".format(filename))
    if func == "":
        return False, False
    if "    while" in func:
        return False, False
    if "input(" in func:
        return False, False
    if "    for" in func:
        return False, False
    try:
        exec(
            script.format(function=func),
            globals(),
            globals(),
        )  
    except:
        compiles = False
    result = globals().get("result")
    return result == solution, compiles

def eval_algorithm(folder, svamp, df, model, temp, flavor="svamp"):
    pass_at_1 = 0
    pass_at_2 = 0
    pass_at_5 = 0
    pass_at_10 = 0
    corrects_at_first = 0
    model_indx = df.model == model
    temp_indx = df.temperature == temp
    shots_indx = df.nshots==1
    func_indx = model_indx & temp_indx & shots_indx
    func_df = df[func_indx]
    for i in range(len(svamp)):
        problem = svamp[i]
        n_hits = 0
        compiles = False
        sample_indx = func_df["sample"] == i

        for k in range(10):
            gen_indx = func_df.generation == k
     
            response = func_df.loc[sample_indx & gen_indx].response.values[0]
            
            is_correct, file_compiles = _eval_file(response, folder, i, k, problem["solution"], flavor)
            if k == 0 and is_correct:
                corrects_at_first += 1
            if is_correct:
                n_hits += 1
            if not is_correct:
                print(f"Problem {folder}/{flavor}-{i:04d}-{k:02d}.py IS WRONG")
            compiles = compiles or file_compiles  # If none of the examples compile we said none of them compiled

        pass_at_1 += pass_at_k(10, n_hits, 1)
        pass_at_2 += pass_at_k(10, n_hits, 2)
        pass_at_5 += pass_at_k(10, n_hits, 5)
        pass_at_10 += pass_at_k(10, n_hits, 10)

        accuracy = corrects_at_first / (i  + 1)
    return {"pass@1": pass_at_1, "pass@2": pass_at_2, "pass@5": pass_at_5, "pass@10": pass_at_10, "accuracy": accuracy}
            



def pass_at_k(n, c, k):
    """
    :param n: total number of samples
    :param c: number of correct samples
    :param k: k in pass@$k$
    """
    if n - c < k: return 1.0
    return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))
        
def eval_svamp(result_root="experiments/output", svamp_path="SVAMP.json"):
    svamp = load_SVAMP(svamp_path)
    old_models = [
        "facebook/incoder-1B",
        "Salesforce/codegen-350M-mono",
        "Salesforce/codegen-2B-mono",
        "Salesforce/codegen-6B-mono",
        "Salesforce/codegen-16B-mono",
        "facebook/incoder-6B",
        "EleutherAI/gpt-j-6B",
        "GPT-3/davinci-codex",
        "GPT-3/cushman-codex",
    ]
    temperatures = ["0.1", "0.3", "0.5"]
    nshots = [1]
    models = [
        "nvidia/OpenMath-Mistral-7B-v0.1-hf",
        "meta-llama/Meta-Llama-3-8B",
        "codellama/CodeLlama-34b-Python-hf",
        "mistralai/Mistral-7B-v0.1",
        "codellama/CodeLlama-13b-Python-hf",
        "bigcode/starcoder2-7b",
        "codellama/CodeLlama-7b-Python-hf",
        "GPT-3/davinci-codex",
        "bigcode/starcoder2-3b",
        "GPT-3/cushman-codex",
        "THUDM/codegeex2-6b",
        "Salesforce/codegen25-7b-mono",
        "Salesforce/codegen-16B-mono",
        "Salesforce/codegen-6B-mono",
        "Salesforce/codegen-2B-mono",
        "EleutherAI/gpt-j-6B",
        "facebook/incoder-6B",
        "Salesforce/codegen-350M-mono",
        "facebook/incoder-1B",        
    ]
    df = []
    processed_svamp = pd.read_json("SVAMP-processed.json")
    for model, temp, nshots in tqdm.tqdm(itertools.product(models, temperatures, nshots ), total=(len(models)*len(temperatures))):
        if model in old_models:
            result_root = "G:/Mi Unidad/Shared Stamos Pablo/experiments_code_gen/output"
        else:
            result_root = "C:/Users/pablo/outputs/output"
        result_path = "{}/{}-shot/temp={}/{}".format(result_root, nshots, temp, model)
        df.append({
            "temperature": temp,
            "nshots": nshots,
            "model": model,
            **eval_algorithm(result_path, svamp, df=processed_svamp, model=model, temp=float(temp))
        })
        pd.DataFrame(df).to_csv("SVAMP-res.csv")



def eval_gsm8k(result_root="experiments/output/", gsm8k_path="GSM-8K.yaml"):
    dataset = load_GSM(gsm8k_path)
    models = [
        "nvidia/OpenMath-Mistral-7B-v0.1-hf",
        "meta-llama/Meta-Llama-3-8B",
        "codellama/CodeLlama-34b-Python-hf",
        "mistralai/Mistral-7B-v0.1",
        "codellama/CodeLlama-13b-Python-hf",
        "bigcode/starcoder2-7b",
        "codellama/CodeLlama-7b-Python-hf",
        "bigcode/starcoder2-3b",
        "THUDM/codegeex2-6b",
        "Salesforce/codegen25-7b-mono",
        "Salesforce/codegen-16B-mono",
        "Salesforce/codegen-6B-mono",
        "Salesforce/codegen-2B-mono",
        "EleutherAI/gpt-j-6b",
        "facebook/incoder-6B",
        "Salesforce/codegen-350M-mono",
        "facebook/incoder-1B",        
    ]
    temperatures = (0.1 , 0.3, 0.5)
    nshots = [1]
    df = []
    processed_gsm = pd.read_json("GSM-8K-processed.json")

    for model, temp, shots in tqdm.tqdm(itertools.product(models, temperatures, nshots ), total=(len(models)*len(temperatures))):
        
        result_path = "{}/{}-shot/temp={}/{}".format(result_root, shots, temp, model)
        df.append({
            "temperature": temp,
            "nshots": nshots,
            "model": model,
            **eval_algorithm(result_path, dataset, df=processed_gsm, model=model, temp=float(temp), flavor="gsm-8k")

        })
        
        pd.DataFrame(df).to_csv("GSM-res.csv")
if __name__ == "__main__":
    eval_svamp()
    eval_gsm8k()
