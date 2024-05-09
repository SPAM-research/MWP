from transformers import AutoTokenizer, AutoModelForCausalLM
import yaml
import tqdm
import fire

import os
import torch
from torch.utils.data import DataLoader, Dataset
from zipfile import ZipFile

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_TOKEN"] = "XXXXX"
HF_TOKEN = "XXXXX"
NUM_RETURN_SEQUENCES = 10
BATCH_SIZE = 2

solved_problem = """# A book has 3 chapters. The first chapter is 91 pages long the second chapter is 23 pages long and the third chapter is 25 pages long.. How many more pages does the first chapter have than the second chapter?
def sol():
    context = dict()
    context['number of chapters'] = 3
    context['number of pages first chapter'] = 91
    context['number of pages sencond chapter'] = 23
    context['number of pages third chapter'] = 25
    context['pages more first chapter'] = context['number of pages first chapter'] - context['number of pages second chapter']
    return context['pages more first chapter']"""

class CustomDataset(Dataset):
    def __init__(self, dataset: str, model_name: str):
        self.file = yaml.safe_load(open(f"{dataset.upper()}.yaml"))
        self.problems_without_excluded = self.file["examples"]
        self.example_problem = "\n".join(solved_problem) + "\n"
        self.is_facebook_model = "facebook" in model_name
        self.is_codegeex_model = "codegeex" in model_name.lower()
      
    def __len__(self) -> int:
        return len(self.problems_without_excluded)

    def __getitem__(self, index: int) -> str:
        problem = self.problems_without_excluded[index]

        statement = f"# {problem['text']}" + "\ndef sol():"
        if self.is_facebook_model:
            return "<| file ext=.py |>\n" + self.example_problem + statement
        if self.is_codegeex_model:
            return "# language: Python\n" + self.example_problem + statement
        return self.example_problem + statement


def prepare_tokenizer(model: str) -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True, token=HF_TOKEN)

    if "codellama" in model:
        tokenizer.pad_token = tokenizer.bos_token
        tokenizer.padding_side = "left"
    if "bigcode" in model:
        tokenizer.pad_token = tokenizer.bos_token
        tokenizer.padding_side = "left"
    if "facebook" in model:
        tokenizer.pad_token = tokenizer.bos_token
        tokenizer.padding_side = "left"
    if "meta-llama" in model:
        tokenizer.pad_token = tokenizer.bos_token
        tokenizer.padding_side = "left"
    if "codegen" in model:
        tokenizer.add_special_tokens({'pad_token': '<sep>'})
        tokenizer.padding_side = "left"
    if "gpt-j" in model:
        tokenizer.pad_token = tokenizer.bos_token
        tokenizer.padding_side = "left"
    if "OpenMath-Mistral-7B-v0.1-hf" in model:
        tokenizer.pad_token = tokenizer.bos_token
        tokenizer.padding_side = "left"
    if "mistralai" in model:
        tokenizer.pad_token = tokenizer.bos_token
        tokenizer.padding_side = "left"
    return tokenizer

def prepare_model(model: str) -> AutoModelForCausalLM:
    return AutoModelForCausalLM.from_pretrained(model, torch_dtype=torch.float16, token=HF_TOKEN, device_map="auto")

def do_inference(tokenizer: AutoTokenizer, llm: AutoModelForCausalLM, dataloader: DataLoader, temperature: float, output_file: str, prompt: str, dataset_name: str, model: str):
    idx = -1
    path = "output/temp={:.02}/{}".format(temperature, model)


    for prompt in tqdm.tqdm(dataloader):
        tokens = tokenizer(
            prompt, return_tensors="pt", padding=True
        )
        prompt_tokens = tokens.input_ids.cuda()
        attention_mask = tokens.attention_mask.cuda()

        with torch.no_grad():
            generated_ids = llm.generate(
                prompt_tokens,
                max_new_tokens=200,
                top_k=10,
                do_sample=True,
                num_return_sequences=NUM_RETURN_SEQUENCES,
                temperature=temperature,
                attention_mask=attention_mask
            )
        generated_code = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        with ZipFile(output_file, "a") as results_zip:
            for sample in range(BATCH_SIZE):
                idx += 1
                for generation in range(NUM_RETURN_SEQUENCES):
                    script = generated_code[generation + NUM_RETURN_SEQUENCES*sample]
                    script = "#" + script[len(prompt):].split("#")[1] # A file is #statement\ndef sol():\n....#more-stuff
                    with results_zip.open(
                        path + "/{}-{:04d}-{:02d}.py".format(dataset_name.lower(), idx, generation),
                        "w",
                    ) as outfile:
                        outfile.write(script.encode("utf-8"))


def main(model="mistralai/Mixtral-8x7B-v0.1", temperature=0.3, batch_size=BATCH_SIZE, dataset_name="GSM-8K"):
    BATCH_SIZE = batch_size
    output_file_name = f"{model.replace("/", "_")}-{temperature}-output.zip"

    dataset = CustomDataset(dataset=dataset_name, nshots=1, model_name=model)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE)
    tokenizer = prepare_tokenizer(model)
    llm = prepare_model(model)
       


    do_inference(tokenizer, llm, dataloader, temperature, output_file_name, dataset.example_problem, dataset_name, model)
    
if __name__ == "__main__":
    fire.Fire(main)