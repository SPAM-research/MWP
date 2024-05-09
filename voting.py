
import json
import numpy as np
import pandas as pd
from yaml import safe_load
import itertools
import statistics
import tqdm

global result

script = """
{function}
global result
result = sol()
"""
def loadGSM(path):
    dataset = safe_load(open(path))
    return [
        {
            "statement": problem["text"],
            "solution": float(problem["response"].replace(",",""))
        }
        for problem in dataset
    ]

def loadSVAMP(svamp_path):
    dataset = json.load(open(svamp_path))
    return [
        {
            "statement": problem["Body"] + ". " + problem["Question"],
            "solution": problem["Answer"],
        }
        for problem in dataset
    ]

def _eval_file(func, folder, i, k, solution, dataset="svamp"):
    filename = folder + f"/{dataset}-{i:04d}-{k:02d}.py"
    compiles = True
    if func == "":
        return False, False
    if "    while" in func:
        return False, False
    if "input(" in func:
        return False, False
    if "    for" in func:
        return False, False
    if "    if " in func:
        return False, False
    try:
        exec(
            script.format(function=func),
            globals(),
            globals(),
        )  
    except:
        print("EXCEPTION IN ", filename)
        compiles = False
    result = globals().get("result")
    if result is None:
        result = float('Inf')
    return result, compiles

def eval_algorithm(folder, svamp, df, model, temp, flavor="svamp"):
    results_k = np.empty((len(svamp), 8))
    model_indx = df.model == model
    temp_indx = df.temperature == temp
    shots_indx = df.nshots==1
    func_indx = model_indx & temp_indx & shots_indx
    func_df = df[func_indx]

    for i in range(len(svamp)):
        problem = svamp[i]
        results = []
        sample_indx = func_df["sample"] == i
        for k in range(10):
            gen_indx = func_df.generation == k
            response = func_df.loc[sample_indx & gen_indx].response.values[0]
            results.append(_eval_file(response, folder, i, k, problem["solution"], flavor)[0])
        results = list(map(lambda x: float(x) if (isinstance(x, (int, float) )) else float("-Inf"), results))

        results_k[i, :] = [int(statistics.mode(results[:k]) == problem["solution"]) for k in range(3, 11)]

    
    return {f"accuracy_{k}": f"{v:.3f}" for k, v in list(zip(range(3, 11), list(results_k[:i].mean(axis=0))))}
           
            
       
def main(result_root="experiments/output", svamp_path="SVAMP.json"):
    svamp = loadSVAMP(svamp_path)
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
    result_root = "C:/Users/pablo/outputs/output"
    for model, temp, nshots in tqdm.tqdm(itertools.product(models, temperatures, nshots ),total=len(models)*len(temperatures)):        
        result_path = "{}/{}-shot/temp={}/{}".format(result_root, nshots, temp, model)
        df.append({
            "temperature": temp,
            "nshots": nshots,
            "model": model,
            **eval_algorithm(result_path, svamp, processed_svamp, model, float(temp))
        })
        pd.DataFrame(df).to_csv("SVAMP-res-voteing.csv")

def gsm8k(result_root="experiments/output/", gsm8k_path="GSM-8K.yaml"):
    dataset = loadGSM(gsm8k_path)
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
    nshots = (1,)
    df = []
    processed_gsm = pd.read_json("GSM-8K-processed.json")
    for model, temp, nshots in tqdm.tqdm(itertools.product(models, temperatures, nshots ), total=len(models)*len(temperatures)):
        result_path = "{}/{}-shot/temp={}/{}".format(result_root, nshots, temp, model)
        df.append({
            "temperature": temp,
            "nshots": nshots,
            "model": model,
            **eval_algorithm(result_path, dataset, processed_gsm, model, float(temp), flavor="gsm-8k")
        })
        pd.DataFrame(df).to_csv("GSM-res-voting.csv")
if __name__ == "__main__":
    main()
    gsm8k()