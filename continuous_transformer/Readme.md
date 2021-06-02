# Continuous-in-Depth Transformers

Part of Speech tagging using continuous-in-depth transformer encoders.

This is a fork of an example from the Flax library: [flax/examples/nlp_seq](https://github.com/google/flax/tree/master/examples/nlp_seq). This repository is unaffiliated with Flax, and the original code is included for replication purposes. The continuous model is a plug-and-play replacement for the original models in the example. The changes made to the original example are as follows:

- Introduce ContinuousTransformer in continuous_transformer.py, which uses baseline_models.py.
- Add refinement to the training loop.
- Add LinearClassifier, an embedding-only model.
- Rename original models.py to baseline_models.py
- Split train.py into main.py and train.py.
- Add our custom model saving code.
- Remove multi-device parallelism.
- Add RNG plumbing.


Train 8 seeds from the root directory of this repo (one level up) with
```bash
python3 -m continuous_transformer.main --batch_size=64 --model_dir=/home/user/english_sequence_experiment --dev=/home/user/ud-treebanks-v2.8/UD_English-GUM/en_gum-ud-dev.conllu --train=/home/user/ud-treebanks-v2.8/UD_English-GUM/en_gum-ud-train.conllu --learning_rate=0.1 --scheme=Euler --num_layers=1 --refine_steps=1000,2000,3000,4000,5000,6000
```
That generates 8 model directories in `/home/user/english_sequence_experiment` with checkpoints and Tensorboard logs. After training, run the following script to perform the compression and graph shortening evaluations on each of those models:
```base
python3 run_transformer_projection.py
```
The outputs will be printed to the screen. It will also generate databases called `convergence.sqlite` in each directory, which can be queried to create the plots.
