# **simble** - **Sim**ple **Byte-Pair** **Encoding** Tokenizer
## A simple Python implementation of the Byte-Pair Encoding Algorithm

### Usage
```python
from tokenizer import Tokenizer

t = Tokenizer(nb_merges=X, max_vocab_size=Y, log_level=Z)
t.train(TEXT_CORPUS)

# Print tokens
t.tokenize(TEXT_CORPUS)

# Decode tokens
t.untokenize(t.tokenize(TEXT_CORPUS))   # == TEXT_CORPUS
```
