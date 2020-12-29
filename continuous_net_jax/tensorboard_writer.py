import tensorflow.summary as tf_summary


class TensorboardWriter:

    def __init__(self, path: str):
        self.summary_writer = tf_summary.create_file_writer(path)

    def Writer(self, name: str):
        step_counter = 0

        def saver(val):
            nonlocal step_counter
            with self.summary_writer.as_default():
                tf_summary.scalar(name, val, step=step_counter)
            step_counter += 1

        return saver

