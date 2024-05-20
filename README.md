# MWP
This code is provided for replication of the results provided in "Automated Math Word Problem solving and quantity identification using Large Language Models for code synthesis"
## DISCLAIMER
**The code is provided as-is. We take no responsability in any damages that it might produce on your system.** 
The scripts have been lightly edited to remove absolute paths from our computers, it is possible that you have to modify routes to make all the scripts work appropiately.
## Usage
### Pre-requisites
 * python 3.10
 * virtual environment with the requirements.txt installed
### Generation of Code Samples
In order to obtain the generated samples you must run the script `inference.py`. This script takes 4 parameters that must be defined in order to produce the desired output:
* `model`: The model to run, a valid huggingface model identificator
* `temperature`: any floating point `0<X<1`
* `batch_size`: any number greater than 1. Used to determine the batch size for inference. Heavy dependant on available memory
* `dataset_name`: etiher GSM-8K or SVAMP

### Evaluation of the generated samples
In order to run the evaluation of the generated samples, you must first to the single_file format, this is done in order to speed up the evaluation of the later scripts by reducing IO.
To do so run the `single_file.py` script.
Once that it is done, you can run `voting.py` or `evaluate.py` in order to obtain the evaluation results of the generated code samples.
