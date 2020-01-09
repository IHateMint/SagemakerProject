import ponet_model
import tensorflow as tf
import argparse, os

def model_fn(features, lables, mode):
    gpus = tf.config.experimental.list_physical_devices('GPU')

def model_fn(features, labels, mode):
    inputs = features
    network = ponet_model.ponet_generator()

    logits = network(inputs, mode == tf.estimator.ModeKeys.TRAIN, dim=128, dropout=0.2)
    # TODO

    predictions = {
        'classes': tf.argmax(logits, axis=1),
        'probabilities': tf.nn.softmax(logits, name='softmax_tensor')
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        export_outputs = {
            SIGNATURE_NAME: tf.estimator.export.PredictOutput(predictions)
        }
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions, export_outputs=export_outputs)

    # Calculate loss, which includes softmax cross entropy and L2 regularization.
    cross_entropy = tf.losses.softmax_cross_entropy(
        logits=logits, onehot_labels=tf.one_hot(labels, 10))

    # Create a tensor named cross_entropy for logging purposes.
    tf.identity(cross_entropy, name='cross_entropy')
    tf.summary.scalar('cross_entropy', cross_entropy)

    # Add weight decay to the loss.
    loss = cross_entropy + _WEIGHT_DECAY * tf.add_n(
        [tf.nn.l2_loss(v) for v in tf.trainable_variables()])

    if mode == tf.estimator.ModeKeys.TRAIN:
        global_step = tf.train.get_or_create_global_step()

        # Multiply the learning rate by 0.1 at 100, 150, and 200 epochs.
        boundaries = [int(_BATCHES_PER_EPOCH * epoch) for epoch in [100, 150, 200]]
        values = [_INITIAL_LEARNING_RATE * decay for decay in [1, 0.1, 0.01, 0.001]]
        learning_rate = tf.train.piecewise_constant(
            tf.cast(global_step, tf.int32), boundaries, values)

        # Create a tensor named learning_rate for logging purposes
        tf.identity(learning_rate, name='learning_rate')
        tf.summary.scalar('learning_rate', learning_rate)

        optimizer = tf.train.MomentumOptimizer(
            learning_rate=learning_rate,
            momentum=_MOMENTUM)

        # Batch norm requires update ops to be added as a dependency to the train_op
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = optimizer.minimize(loss, global_step)
    else:
        train_op = None

    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=predictions,
        loss=loss,
        train_op=train_op)


def serving_input_fn():
    """
    Serving input function for CIFAR-10. Specifies the input format the caller of predict() will have to provide.
    For more information: https://www.tensorflow.org/guide/saved_model#build_and_load_a_savedmodel
    """
    inputs = {INPUT_TENSOR_NAME: tf.placeholder(tf.float32, [None, 32, 32, 3])}
    return tf.estimator.export.ServingInputReceiver(inputs, inputs)

def make_batch(data_dir, batch_size=2):
    dataset = tf.data.TFRecordDataset(data_dir).repeat()

    dataset = dataset.map(parser, num_parallel_calls=batch_size)

    min_queue_examples = int(45000 * 0.4)
    # Ensure that the capacity is sufficiently large to provide good random
    # shuffling.
    dataset = dataset.shuffle(buffer_size=min_queue_examples + 3 * batch_size)

    dataset = dataset.batch(batch_size)
    iterator = dataset.make_one_shot_iterator()

    image, label = iterator.get_next()

    return image, label


def parser(serialized_example):
    features = tf.parse_single_example(
        serialized_example,
        features={
            'image': tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([], tf.int64),
        })
    image = tf.decode_raw(features['image'], tf.uint8)
    image.set_shape([DEPTH * HEIGHT * WIDTH])

    image = tf.cast(tf.transpose(tf.reshape(image, [DEPTH, HEIGHT, WIDTH]), [1, 2, 0]), tf.float32)
    label = tf.cast(features['label'], tf.int32)

    return image, label


def train_input_fn(data_dir):
    with tf.device('/cpu:0'):
        train_data = os.path.join(data_dir, 'train.tfrecords')
        image_batch, label_batch = make_batch(train_data, BATCH_SIZE)
        return {INPUT_TENSOR_NAME: image_batch}, label_batch


def eval_input_fn(data_dir):
    with tf.device('/cpu:0'):
        eval_data = os.path.join(data_dir, 'eval.tfrecords')
        image_batch, label_batch = make_batch(eval_data, BATCH_SIZE)

        return {INPUT_TENSOR_NAME: image_batch}, label_batch


def train(model_dir, data_dir, train_steps):
    estimator = tf.estimator.Estimator(model_fn=model_fn, model_dir=model_dir)

    temp_input_fn = functools.partial(train_input_fn, data_dir)

    train_spec = tf.estimator.TrainSpec(temp_input_fn, max_steps=train_steps)

    exporter = tf.estimator.LatestExporter('Servo', serving_input_receiver_fn=serving_input_fn)
    temp_eval_fn = functools.partial(eval_input_fn, data_dir)
    eval_spec = tf.estimator.EvalSpec(temp_eval_fn, steps=1, exporters=exporter)

    tf.estimator.train_and_evaluate(estimator=estimator, train_spec=train_spec, eval_spec=eval_spec)


def main(model_dir, data_dir, train_steps):
    tf.logging.set_verbosity(tf.logging.INFO)
    train(model_dir, data_dir, train_steps)


if __name__ == '__main__':
    args_parser = argparse.ArgumentParser()
    # For more information:
    # https://docs.aws.amazon.com/sagemaker/latest/dg/your-algorithms-training-algo.html
    args_parser.add_argument(
        '--data-dir',
        default='/opt/ml/input/data/training',
        type=str,
        help='The directory where the CIFAR-10 input data is stored. Default: /opt/ml/input/data/training. This '
             'directory corresponds to the SageMaker channel named \'training\', which was specified when creating '
             'our training job on SageMaker')

    # For more information:
    # https://docs.aws.amazon.com/sagemaker/latest/dg/your-algorithms-inference-code.html
    args_parser.add_argument(
        '--model-dir',
        default='/opt/ml/model',
        type=str,
        help='The directory where the model will be stored. Default: /opt/ml/model. This directory should contain all '
             'final model artifacts as Amazon SageMaker copies all data within this directory as a single object in '
             'compressed tar format.')

    args_parser.add_argument(
        '--train-steps',
        type=int,
        default=100,
        help='The number of steps to use for training.')
    args = args_parser.parse_args()
    main(**vars(args))
