## Continuous Transformers

Part of Speech tagging using continuous-depth transformer encoders. 
This is a fork of [flax/examples/nlp_seq](https://github.com/google/flax/tree/master/examples/nlp_seq). 

```bash
python3 -m continuous_transformer.main --batch_size=64 --model_dir=/home/afq/tree_bank/ancient_greek --dev=/home/afq/ud-treebanks-v2.0/UD_English/en-ud-dev.conllu --train=/home/afq/ud-treebanks-v2.0/UD_English/en-ud-train.conllu --learning_rate=0.01
```