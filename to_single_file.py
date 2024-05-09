
import itertools
import pandas as pd
import tqdm
import multiprocessing
from functools import partial


def load(it, result_path, dataset, temp, nshots, model):
    sample, generation = it
    
    f = open(result_path + f"/{dataset}-{sample:04d}-{generation:02d}.py", "r", encoding="utf-8")
    lines = f.readlines()
    line_contains_return = list(enumerate(map(lambda x: x.startswith("    return"), lines[1:])))
    if not any(map(lambda x: x[1], line_contains_return)):
        func = ""
    else:
        first_return = list(filter(lambda x: x[1], line_contains_return)) [0][0]
        func = "".join(lines[1:first_return+2])
    return {
        "temperature": temp,
        "nshots": nshots,
        "model": model,
        "sample": sample,
        "generation": generation,
        "response": func
    }
def main(dataset="svamp",_len=1000, old_models=[], ignore=[]):

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
        "EleutherAI/gpt-j-6b",
        "facebook/incoder-6B",
        "Salesforce/codegen-350M-mono",
        "facebook/incoder-1B",        
    ]
    df = []
    
    for model, temp, nshots in tqdm.tqdm(itertools.product(models, temperatures, nshots ),  total=(len(models)*len(temperatures))):
        if model in ignore:
            continue
        if model in old_models:
            result_root = "G:/Mi Unidad/Shared Stamos Pablo/experiments_code_gen/output"
        else:
            result_root = "C:/Users/pablo/outputs/output"
        result_path = "{}/{}-shot/temp={}/{}".format(result_root, nshots, temp, model)
        
        f = partial(load,result_path=result_path, model=model, temp=temp, dataset=dataset, nshots=nshots)
        with multiprocessing.Pool(10) as p:
            df += p.map(f,itertools.product(range(_len), range(10)), )
        
    pd.DataFrame(df).to_json(f"{dataset.upper()}-processed.json", orient="records")
if __name__=="__main__":
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
    main("svamp", old_models, ignore={"EleutherAI/gpt-j-6b"})
    main("gsm-8k", _len=1319, ignore={"GPT-3/davinci-codex", "GPT-3/cushman-codex", "EleutherAI/gpt-j-6B"})