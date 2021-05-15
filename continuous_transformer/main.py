from absl import app
from absl import flags

FLAGS = flags.FLAGS

from . import input_pipeline
from . import baseline_models as models
from . import continuous_transformers

from .train import *

flags.DEFINE_string('model_dir', default='', help=('Directory for model data.'))

flags.DEFINE_string('experiment', default='xpos', help=('Experiment name.'))
flags.DEFINE_integer('batch_size',
                     default=64,
                     help=('Batch size for training.'))
flags.DEFINE_integer(
    'eval_frequency',
    default=100,
    help=('Frequency of eval during training, e.g. every 1000 steps.'))
flags.DEFINE_integer('num_train_steps',
                     default=75000,
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


def main(argv):
    if len(argv) > 1:
        raise app.UsageError('Too many command-line arguments.')

    # Make sure tf does not allocate gpu memory.
    tf.config.experimental.set_visible_devices([], 'GPU')

    batch_size = FLAGS.batch_size
    learning_rate = FLAGS.learning_rate
    num_train_steps = FLAGS.num_train_steps
    eval_freq = FLAGS.eval_frequency
    random_seed = FLAGS.random_seed

    if not FLAGS.dev:
        raise app.UsageError('Please provide path to dev set.')
    if not FLAGS.train:
        raise app.UsageError('Please provide path to training set.')
    if batch_size % jax.device_count() > 0:
        raise ValueError(
            'Batch size must be divisible by the number of devices')
    device_batch_size = batch_size // jax.device_count()

    if jax.host_id() == 0:
        train_summary_writer = tensorboard.SummaryWriter(
            os.path.join(FLAGS.model_dir, FLAGS.experiment + '_train'))
        eval_summary_writer = tensorboard.SummaryWriter(
            os.path.join(FLAGS.model_dir, FLAGS.experiment + '_eval'))

    # create the training and development dataset
    vocabs = input_pipeline.create_vocabs(FLAGS.train)
    config = models.TransformerConfig(vocab_size=len(vocabs['forms']),
                                      output_vocab_size=len(vocabs['xpos']),
                                      max_len=FLAGS.max_length)
    logging.info("%s", config)
    print(config)
    attributes_input = [input_pipeline.CoNLLAttributes.FORM]
    attributes_target = [input_pipeline.CoNLLAttributes.XPOS]
    train_ds = input_pipeline.sentence_dataset_dict(FLAGS.train,
                                                    vocabs,
                                                    attributes_input,
                                                    attributes_target,
                                                    batch_size=batch_size,
                                                    bucket_size=config.max_len)
    train_iter = iter(train_ds)

    eval_ds = input_pipeline.sentence_dataset_dict(FLAGS.dev,
                                                   vocabs,
                                                   attributes_input,
                                                   attributes_target,
                                                   batch_size=batch_size,
                                                   bucket_size=config.max_len,
                                                   repeat=1)

    # model = models.Transformer(config)
    model = continuous_transformers.ContinuousTransformer(config)

    rng = random.PRNGKey(random_seed)
    rng, init_rng = random.split(rng)

    # call a jitted initialization function to get the initial parameter tree
    @jax.jit
    def initialize_variables(init_rng):
        init_batch = jnp.ones((config.max_len, 1), jnp.float32)
        init_variables = model.init(init_rng, inputs=init_batch, train=False)
        return init_variables

    init_variables = initialize_variables(init_rng)

    print(list(init_variables.keys()))
    #state = init_variables['ode_state']
    #print(state)
    optimizer_def = optim.Adam(learning_rate,
                               beta1=0.9,
                               beta2=0.98,
                               eps=1e-9,
                               weight_decay=1e-1)
    optimizer = optimizer_def.create(init_variables['params'])
    optimizer = jax_utils.replicate(optimizer)
    learning_rate_fn = create_learning_rate_scheduler(
        'constant * linear_warmup * decay_every',
        base_learning_rate=learning_rate,
        decay_factor=0.1)

    p_train_step = jax.pmap(functools.partial(
        train_step, model=model, learning_rate_fn=learning_rate_fn),
                            axis_name='batch')

    def eval_step(params, batch):
        """Calculate evaluation metrics on a batch."""
        inputs, targets = batch['inputs'], batch['targets']
        weights = jnp.where(targets > 0, 1.0, 0.0)
        logits = model.apply({'params': params}, inputs=inputs, train=False)
        return compute_metrics(logits, targets, weights)

    p_eval_step = jax.pmap(eval_step, axis_name='batch')

    # We init the first set of dropout PRNG keys, but update it afterwards inside
    # the main pmap'd training update for performance.
    dropout_rngs = random.split(rng, jax.local_device_count())
    metrics_all = []
    tick = time.time()
    best_dev_score = 0
    for step, batch in zip(range(num_train_steps), train_iter):
        batch = common_utils.shard(jax.tree_map(lambda x: x._numpy(), batch))  # pylint: disable=protected-access

        optimizer, metrics, dropout_rngs = p_train_step(
            optimizer, batch, dropout_rng=dropout_rngs)
        metrics_all.append(metrics)

        if (step + 1) % eval_freq == 0:
            metrics_all = common_utils.get_metrics(metrics_all)
            lr = metrics_all.pop('learning_rate').mean()
            metrics_sums = jax.tree_map(jnp.sum, metrics_all)
            denominator = metrics_sums.pop('denominator')
            summary = jax.tree_map(lambda x: x / denominator, metrics_sums)  # pylint: disable=cell-var-from-loop
            summary['learning_rate'] = lr
            logging.info('train in step: %d, loss: %.4f', step, summary['loss'])
            if jax.host_id() == 0:
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
                eval_batch = common_utils.shard(eval_batch)

                metrics = p_eval_step(optimizer.target, eval_batch)
                eval_metrics.append(metrics)
            eval_metrics = common_utils.get_metrics(eval_metrics)
            eval_metrics_sums = jax.tree_map(jnp.sum, eval_metrics)
            eval_denominator = eval_metrics_sums.pop('denominator')
            eval_summary = jax.tree_map(
                lambda x: x / eval_denominator,  # pylint: disable=cell-var-from-loop
                eval_metrics_sums)

            logging.info('eval in step: %d, loss: %.4f, accuracy: %.4f', step,
                         eval_summary['loss'], eval_summary['accuracy'])

            if best_dev_score < eval_summary['accuracy']:
                best_dev_score = eval_summary['accuracy']
                # TODO: save model.
            eval_summary['best_dev_score'] = best_dev_score
            logging.info('best development model score %.4f', best_dev_score)
            if jax.host_id() == 0:
                for key, val in eval_summary.items():
                    eval_summary_writer.scalar(key, val, step)
                eval_summary_writer.flush()


if __name__ == '__main__':
    app.run(main)
