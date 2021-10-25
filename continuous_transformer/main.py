from absl import app
from absl import flags


from continuous_net_jax import experiment
from continuous_net_jax.tools import count_parameters


from . import input_pipeline
from . import baseline_models as models
from . import continuous_transformers
from . import linear_baseline
from .train import *

from flax import traverse_util
from flax.core import unfreeze
import flax
from flax.training.common_utils import *

Experiment = experiment.Experiment
FLAGS = flags.FLAGS


flags.DEFINE_string('model_dir', default='', help=('Directory for model data.'))
flags.DEFINE_string('experiment', default='xpos', help=('Experiment name.'))
flags.DEFINE_integer('batch_size',
                     default=64,
                     help=('Batch size for training.'))
flags.DEFINE_integer(
    'eval_frequency',
    default=500,
    help=('Frequency of eval during training, e.g. every 1000 steps.'))
flags.DEFINE_integer('num_train_steps',
                     default=50000,
                     help=('Number of train steps.'))
flags.DEFINE_float('learning_rate', default=0.05, help=('Learning rate.'))
flags.DEFINE_float('weight_decay',
                   default=1e-1,
                   help=('Decay factor for AdamW style weight decay.'))
flags.DEFINE_integer('max_length',
                     default=256,
                     help=('Maximum length of examples.'))
flags.DEFINE_integer('random_seed',
                     default=0,
                     help=('Integer for PRNG random seed.'))
flags.DEFINE_string('train', default='', help=('Path to training data.'))
flags.DEFINE_string('dev', default='', help=('Path to development data.'))

flags.DEFINE_string('scheme', default='Euler', help=('Which integrator scheme to use.'))
flags.DEFINE_string('basis', default='piecewise_constant', help=('Which basis function to use.'))
flags.DEFINE_integer('num_layers',
                     default=1,
                     help=('Length of the encoder transformers.'))

flags.DEFINE_list('refine_steps', default='', help=('Refinement Steps'))



def main(argv):
    if len(argv) > 1:
        raise app.UsageError('Too many command-line arguments.')
    if not FLAGS.dev:
        raise app.UsageError('Please provide path to dev set.')
    if not FLAGS.train:
        raise app.UsageError('Please provide path to training set.')
    for seed in range(0,8):
        do_it(
            batch_size = FLAGS.batch_size,
            learning_rate = FLAGS.learning_rate,
            num_train_steps = FLAGS.num_train_steps,
            eval_freq = FLAGS.eval_frequency,
            random_seed = seed,
            scheme = FLAGS.scheme,
            basis = FLAGS.basis,
            refine_steps = [int(i) for i in FLAGS.refine_steps],
            max_length=FLAGS.max_length,
            num_layers=FLAGS.num_layers,
            experiment=FLAGS.experiment,
            train=FLAGS.train,
            dev=FLAGS.dev,
            model_dir=FLAGS.model_dir
        )


def do_it(
        batch_size,
        learning_rate,
        num_train_steps,
        eval_freq,
        random_seed,
        scheme,
        basis,
        refine_steps,
        max_length,
        num_layers,
        experiment,
        train,
        dev,
        model_dir):
    # Make sure tf does not allocate gpu memory.
    tf.config.experimental.set_visible_devices([], 'GPU')

    if batch_size % jax.device_count() > 0:
        raise ValueError(
            'Batch size must be divisible by the number of devices')
    device_batch_size = batch_size // jax.device_count()

    # create the training and development dataset
    vocabs = input_pipeline.create_vocabs(train)
    # Define the config
    config = models.TransformerConfig(vocab_size=len(vocabs['forms']),
                                      output_vocab_size=len(vocabs['xpos']),
                                      max_len=max_length,
                                      num_layers=num_layers)
    logging.info("%s", config)

    attributes_input = [input_pipeline.CoNLLAttributes.FORM]
    attributes_target = [input_pipeline.CoNLLAttributes.XPOS]
    train_ds = input_pipeline.sentence_dataset_dict(train,
                                                    vocabs,
                                                    attributes_input,
                                                    attributes_target,
                                                    batch_size=batch_size,
                                                    bucket_size=config.max_len)
    train_iter = iter(train_ds)
    eval_ds = input_pipeline.sentence_dataset_dict(dev,
                                                   vocabs,
                                                   attributes_input,
                                                   attributes_target,
                                                   batch_size=batch_size,
                                                   bucket_size=config.max_len,
                                                   repeat=1)


    # Original model from Flax example:
    # model = models.Transformer(config)
    # Linear baseline:
    # model = linear_baseline.LinearClassifer(config)
    # Continuous-in-depth Transformer:
    model = continuous_transformers.ContinuousTransformer(
        config, scheme=scheme, basis=basis,
        n_step=config.num_layers, n_basis=config.num_layers)

    
    rng = random.PRNGKey(random_seed)
    rng, init_rng = random.split(rng)

    # call a jitted initialization function to get the initial parameter tree
    @jax.jit
    def initialize_variables(init_rng):
        init_batch = jnp.ones((config.max_len, 1), jnp.float32)
        init_variables = model.init(init_rng, inputs=init_batch, train=False, rng=None)
        return init_variables

    init_variables = initialize_variables(init_rng)

    optimizer_def = optim.Adam(learning_rate,
                               beta1=0.9,
                               beta2=0.98,
                               eps=1e-9,
                               weight_decay=1e-1)
    optimizer = optimizer_def.create(init_variables['params'])
    # optimizer = jax_utils.replicate(optimizer)
    learning_rate_fn = create_learning_rate_scheduler(
        # 'constant * linear_warmup * decay_every',
        "constant * linear_warmup * rsqrt_decay",
        base_learning_rate=learning_rate,
        decay_factor=0.1)

    # p_train_step = jax.pmap(functools.partial(
    #    train_step, model=model, learning_rate_fn=learning_rate_fn),
    #                        axis_name='batch')
    train_step_p = functools.partial(
        train_step, model=model, learning_rate_fn=learning_rate_fn)
    train_step_p = jax.jit(train_step_p)
    @jax.jit
    def eval_step(params, batch):
        """Calculate evaluation metrics on a batch."""
        inputs, targets = batch['inputs'], batch['targets']
        weights = jnp.where(targets > 0, 1.0, 0.0)
        logits = model.apply({'params': params}, inputs=inputs, train=False, rng=None)
        return compute_metrics(logits, targets, weights)
    # p_eval_step = jax.pmap(eval_step, axis_name='batch')

    # Saving helpers.
    exp = Experiment(model, path=model_dir)
    exp.save_optimizer_hyper_params(optimizer_def, random_seed,
                                    {'learning_rate_decay_epochs': [],
                                     'refine_steps': refine_steps})
    if jax.process_index() == 0:
        train_summary_writer = tensorboard.SummaryWriter(
            os.path.join(exp.path, experiment + '_train'))
        eval_summary_writer = tensorboard.SummaryWriter(
            os.path.join(exp.path, experiment + '_eval'))

    # We init the first set of dropout PRNG keys, but update it afterwards inside
    # the main pmap'd training update for performance.
    # dropout_rngs = random.split(rng, jax.local_device_count())
    dropout_rngs = rng
    metrics_all = []
    tick = time.time()
    best_dev_score = 0
    for step, batch in zip(range(num_train_steps), train_iter):
        # batch = common_utils.shard(jax.tree_map(lambda x: x._numpy(), batch))  # pylint: disable=protected-access
        batch = jax.tree_map(lambda x: x._numpy(), batch)

        # Refine.
        if step in refine_steps:
            print("Refining:")
            flat_opt_state = {'/'.join(k): v for k, v in traverse_util.flatten_dict(unfreeze(optimizer.target)).items()}
            model, params = exp.model.refine(optimizer.target)
            print("New Model: ", model)
            exp.model = model
            optimizer = optimizer_def.create(params)
            #optimizer = jax_utils.replicate(optimizer)
            flat_opt_state = {'/'.join(k): v for k, v in
                              traverse_util.flatten_dict(unfreeze(optimizer.target)).items()}
            # print(jax.tree_map(jnp.shape, flat_opt_state))
            print("Now have ", count_parameters(params))

            # Remake the training and eval functions with the new model.
            #  p_train_step = jax.pmap(functools.partial(
            #      train_step, model=model, learning_rate_fn=learning_rate_fn),
            #                          axis_name='batch')
            train_step_p = functools.partial(
                train_step, model=model, learning_rate_fn=learning_rate_fn)
            train_step_p = jax.jit(train_step_p)
            @jax.jit
            def eval_step(params, batch):
                """Calculate evaluation metrics on a batch."""
                inputs, targets = batch['inputs'], batch['targets']
                weights = jnp.where(targets > 0, 1.0, 0.0)
                logits = model.apply({'params': params}, inputs=inputs, train=False)
                return compute_metrics(logits, targets, weights)
            # p_eval_step = jax.pmap(eval_step, axis_name='batch')

        # Step.
        optimizer, metrics, dropout_rngs = train_step_p(
            optimizer, batch, step, dropout_rng=dropout_rngs)
        metrics_all.append(metrics)

        # Evaluate.
        if (step + 1) % eval_freq == 0:
            metrics_all = jax.device_get(metrics_all)
            metrics_all = stack_forest(metrics_all)
            lr = metrics_all.pop('learning_rate').mean()
            metrics_sums = jax.tree_map(jnp.sum, metrics_all)
            denominator = metrics_sums.pop('denominator')
            summary = jax.tree_map(lambda x: x / denominator, metrics_sums)  # pylint: disable=cell-var-from-loop
            summary['learning_rate'] = lr
            logging.info('train in step: %d, loss: %.4f', step, summary['loss'])
            if jax.process_index() == 0:
                tock = time.time()
                steps_per_sec = eval_freq / (tock - tick)
                tick = tock
                train_summary_writer.scalar('steps per second', steps_per_sec,
                                            step)
                for key, val in summary.items():
                    train_summary_writer.scalar(key, val, step)
                train_summary_writer.flush()

            metrics_all = [
            ]  # reset metric accumulation for next evaluation cycle.

            eval_metrics = []
            eval_iter = iter(eval_ds)

            for eval_batch in eval_iter:
                eval_batch = jax.tree_map(lambda x: x._numpy(), eval_batch)  # pylint: disable=protected-access
                # Handle final odd-sized batch by padding instead of dropping it.
                cur_pred_batch_size = eval_batch['inputs'].shape[0]
                if cur_pred_batch_size != batch_size:
                    # pad up to batch size
                    eval_batch = jax.tree_map(
                        lambda x: pad_examples(x, batch_size), eval_batch)
                # eval_batch = common_utils.shard(eval_batch)

                metrics = eval_step(optimizer.target, eval_batch)
                eval_metrics.append(metrics)
            # eval_metrics = common_utils.get_metrics(eval_metrics)
            eval_metrics = jax.device_get(eval_metrics)
            eval_metrics = stack_forest(eval_metrics)
            eval_metrics_sums = jax.tree_map(jnp.sum, eval_metrics)
            eval_denominator = eval_metrics_sums.pop('denominator')
            eval_summary = jax.tree_map(
                lambda x: x / eval_denominator,  # pylint: disable=cell-var-from-loop
                eval_metrics_sums)

            logging.info('eval in step: %d, loss: %.4f, accuracy: %.4f', step,
                         eval_summary['loss'], eval_summary['accuracy'])

            if best_dev_score < eval_summary['accuracy']:
                best_dev_score = eval_summary['accuracy']
                exp.save_checkpoint(optimizer, {}, step)

            eval_summary['best_dev_score'] = best_dev_score
            logging.info('best development model score %.4f', best_dev_score)
            if jax.process_index() == 0:
                for key, val in eval_summary.items():
                    eval_summary_writer.scalar(key, val, step)
                eval_summary_writer.flush()


if __name__ == '__main__':
    app.run(main)
