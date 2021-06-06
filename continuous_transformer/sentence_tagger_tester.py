from typing import Iterable, Tuple
from tqdm import tqdm
from SimDataDB import SimDataDB2
from continuous_net_jax.convergence import *
from continuous_transformer import input_pipeline
from continuous_transformer.train import *
from flax.training.common_utils import *
from continuous_net_jax.tools import count_parameters


def get_max_length(dataset):
    """Calculate the maximum sentence length to tighten batch size padding."""
    max_len = 0
    for batch in tqdm(iter(dataset)):
        for sentence in batch['inputs']:
            sentence_len = 0
            for c in sentence:
                sentence_len += 1
                if c == 0:
                    break
            if sentence_len > max_len:
                max_len = sentence_len
                print(max_len)
    return max_len


class TransformerTester():

    def __init__(self,
                 batch_size=64,
                 datadir="../ud-treebanks-v2.8/UD_English-GUM/",
                 prefix='en',
                 max_len=256):
        self.batch_size = batch_size
        self.max_len = max_len
        self.train_file = f"{datadir}/{prefix}-ud-train.conllu"
        self.dev_file = f"{datadir}/{prefix}-ud-dev.conllu"
        self.test_file = f"{datadir}/{prefix}-ud-test.conllu"
        self.load_data()

    def load_data(self):
        self.vocabs = input_pipeline.create_vocabs(self.train_file)
        self.inverse_lookup = {
            rank: {v: k for k, v in table.items()
                  } for rank, table in self.vocabs.items()
        }
        attributes_input = [input_pipeline.CoNLLAttributes.FORM]
        attributes_target = [input_pipeline.CoNLLAttributes.XPOS]
        self.train_ds = input_pipeline.sentence_dataset_dict(
            self.train_file,
            self.vocabs,
            attributes_input,
            attributes_target,
            batch_size=self.batch_size,
            bucket_size=self.max_len,
            repeat=1)
        self.dev_ds = input_pipeline.sentence_dataset_dict(
            self.dev_file,
            self.vocabs,
            attributes_input,
            attributes_target,
            batch_size=self.batch_size,
            bucket_size=self.max_len,
            repeat=1)
        self.test_ds = input_pipeline.sentence_dataset_dict(
            self.test_file,
            self.vocabs,
            attributes_input,
            attributes_target,
            batch_size=self.batch_size,
            bucket_size=self.max_len,
            repeat=1)

    def compute_inference_accuracy(self, model, params, eval_ds):
        eval_metrics = []
        eval_iter = iter(eval_ds)
        batch_size = self.batch_size

        @jax.jit
        def eval_step(params, batch):
            """Calculate evaluation metrics on a batch."""
            inputs, targets = batch['inputs'], batch['targets']
            weights = jnp.where(targets > 0, 1.0, 0.0)
            logits = model.apply({'params': params}, inputs=inputs, train=False)
            return compute_metrics(logits, targets, weights)

        for eval_batch in eval_iter:
            eval_batch = jax.tree_map(lambda x: x._numpy(), eval_batch)
            # Handle final odd-sized batch by padding instead of dropping it.
            cur_pred_batch_size = eval_batch['inputs'].shape[0]
            if cur_pred_batch_size != batch_size:
                # pad up to batch size
                eval_batch = jax.tree_map(lambda x: pad_examples(x, batch_size),
                                          eval_batch)
            metrics = eval_step(params, eval_batch)
            eval_metrics.append(metrics)
        eval_metrics = jax.device_get(eval_metrics)
        eval_metrics = stack_forest(eval_metrics)
        eval_metrics_sums = jax.tree_map(jnp.sum, eval_metrics)
        eval_denominator = eval_metrics_sums.pop('denominator')
        eval_summary = jax.tree_map(
            lambda x: x / eval_denominator,  # pylint: disable=cell-var-from-loop
            eval_metrics_sums)
        return eval_summary

    def perform_interpolate_and_infer(self,
                                      ct: ConvergenceTester,
                                      bases: Iterable[str],
                                      n_bases: Iterable[int],
                                      schemes: Iterable[str],
                                      n_steps: Iterable[int]):

        @SimDataDB2(os.path.join(ct.path, "convergence.sqlite"))
        def infer_interpolated_test_error(scheme: str, n_step: int, basis: str,
                                           n_basis: int) -> Tuple[float, int]:
            # Rely on the LRU cache to avoid the second call, and sqlite
            # cache to avoid the first call.
            p_model, p_params, p_state = ct.interpolate(basis, n_basis)
            s_p_model = p_model.clone(n_step=n_step, scheme=scheme)
            metrics = self.compute_inference_accuracy(s_p_model, p_params,
                                                      self.test_ds)
            err = metrics['accuracy']
            return float(err), count_parameters(p_params)

        print("| Basis | n_basis | Scheme | n_step | error | n_params |")
        print("|-------|----------------------------------------------|")
        errors = {}
        for basis in bases:
            for n_basis in n_bases:
                for n_step in n_steps:
                    for scheme in schemes:
                        e, num_params = infer_interpolated_test_error(
                            scheme, n_step, basis, n_basis)
                        print(
                            f"| {basis:20} | {n_basis} | {scheme:5} | {n_step} | {e:1.3f} | {num_params} |"
                        )

    def perform_project_and_infer(self,
                                  ct: ConvergenceTester,
                                  bases: Iterable[str],
                                  n_bases: Iterable[int],
                                  schemes: Iterable[str],
                                  n_steps: Iterable[int]):

        @SimDataDB2(os.path.join(ct.path, "convergence.sqlite"))
        def infer_projected_test_error(scheme: str, n_step: int, basis: str,
                                        n_basis: int) -> Tuple[float, int]:
            # Rely on the LRU cache to avoid the second call, and sqlite
            # cache to avoid the first call.
            p_model, p_params, p_state = ct.project(basis, n_basis)
            s_p_model = p_model.clone(n_step=n_step, scheme=scheme)
            metrics = self.compute_inference_accuracy(s_p_model, p_params,
                                                      self.test_ds)
            err = metrics['accuracy']
            return float(err), count_parameters(p_params)

        print("| Basis | n_basis | Scheme | n_step | error | n_params |")
        print("|-------|----------------------------------------------|")
        errors = {}
        for basis in bases:
            for n_basis in n_bases:
                for n_step in n_steps:
                    for scheme in schemes:
                        e, num_params = infer_projected_test_error(
                            scheme, n_step, basis, n_basis)
                        print(
                            f"| {basis:20} | {n_basis} | {scheme:5} | {n_step} | {e:1.3f} | {num_params} |"
                        )
