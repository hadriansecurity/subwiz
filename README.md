<pre style="color: lime; background-color: black;">
███████╗██╗   ██╗██████╗     ██╗    ██╗██╗███████╗
██╔════╝██║   ██║██╔══██╗    ██║    ██║██║╚══███╔╝
███████╗██║   ██║██████╔╝    ██║ █╗ ██║██║  ███╔╝ 
╚════██║██║   ██║██╔══██╗    ██║███╗██║██║ ███╔╝  
███████║╚██████╔╝██████╔╝    ╚███╔███╔╝██║███████╗
╚══════╝ ╚═════╝ ╚═════╝      ╚══╝╚══╝ ╚═╝╚══════╝
</pre>

A lightweight GPT model, trained to discover subdomains.

### Installation

```pipx install subwiz```

OR

```pip install subwiz```

### Recommended Use

Use [subfinder](https://github.com/projectdiscovery/subfinder) ❤️ to find subdomains from passive sources:

```subfinder -d example.com -o subdomains.txt```

Seed subwiz with these subdomains:

```subwiz -i subdomains.txt```

### Supported Switches

```commandline
usage: cli.py [-h] -i INPUT_FILE [-o OUTPUT_FILE] [-n NUM_PREDICTIONS] [--no-resolve]
              [--force-download] [--max-recursion MAX_RECURSION] [-t TEMPERATURE]
              [-d {auto,cpu,cuda,mps}] [-m MAX_NEW_TOKENS]
              [--resolution-concurrency RESOLUTION_CONCURRENCY] [--multi-apex] [-q] [-s]

options:
  -h, --help            show this help message and exit
  -i INPUT_FILE, --input-file INPUT_FILE
                        file containing new-line-separated subdomains. (default: None)
  -o OUTPUT_FILE, --output-file OUTPUT_FILE
                        output file to write new-line separated subdomains to. (default: None)
  -n NUM_PREDICTIONS, --num-predictions NUM_PREDICTIONS
                        number of subdomains to predict. (default: 500)
  --no-resolve          do not resolve the output subdomains. (default: False)
  --force-download      download model and tokenizer files, even if cached. (default: False)
  --max-recursion MAX_RECURSION
                        maximum number of times the inference process will recursively re-run
                        after finding new subdomains. (default: 5)
  -t TEMPERATURE, --temperature TEMPERATURE
                        add randomness to the model (recommended ≤ 0.3). (default: 0.0)
  -d {auto,cpu,cuda,mps}, --device {auto,cpu,cuda,mps}
                        hardware to run the transformer model on. (default: auto)
  -m MAX_NEW_TOKENS, --max-new-tokens MAX_NEW_TOKENS
                        maximum length of predicted subdomains in tokens. (default: 10)
  --resolution-concurrency RESOLUTION_CONCURRENCY
                        number of concurrent resolutions. (default: 128)
  --multi-apex          allow multiple apex domains in the input file. runs inference for each
                        apex separately. (default: False)
  -q, --quiet           useful for piping into another tool. (default: False)
  -s, --silent          do not print any output. requires --output-file. (default: False)
```

### In Python

Use subwiz in Python, with the same parameters as the command line interface.

```
import subwiz

known_subdomains = ['test1.example.com', 'test2.example.com']
new_subdomains = subwiz.run(input_domains=known_subdomains)
```

---
### Model

Use the `--no-resolve` flag to inspect model outputs without checking if they resolve.

#### Architecture

Subwiz is a ultra-lightweight transformer model based on [nanoGPT](https://github.com/karpathy/nanoGPT/tree/master) ❤️:

- 17.3M parameters.
- Trained on 26M tokens, lists of subdomains from passive sources.
- Tokenizer trained on same lists of subdomains (8192 tokens).

#### Hugging Face
The model is saved in Hugging Face as [HadrianSecurity/subwiz](https://huggingface.co/HadrianSecurity/subwiz).
It is downloaded when you first run subwiz.

#### Inference

Typically, generative transformer models (e.g. ChatGPT) predict a single output sequence.
Subwiz predicts the N most likely sequences using a beam search algorithm.

![Diagram of the inference algorithm](https://raw.githubusercontent.com/hadriansecurity/subwiz/main/subwiz_inference.png)

*Beam search algorithm to predict the N most likely sequences using a generative transformer model.*
