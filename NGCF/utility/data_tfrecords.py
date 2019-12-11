import tensorflow as tf
import os
flags = tf.flags

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "data_dir", None,
    "The input data dir. Should contain the .tsv files (or other data files) "
    "for the task.")


flags.DEFINE_string(
    "input_file", None,
    "The input files "
    "for the task.")

flags.DEFINE_string(
    "output_file", None,
    "output_file")


def _int64_feature(value):
    if isinstance(value, list):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=value))
    else:
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


if __name__ == '__main__':
    with open(os.path.join(FLAGS.data_dir, FLAGS.input_file), "r") as f:
        writer = tf.python_io.TFRecordWriter(os.path.join(FLAGS.data_dir, FLAGS.output_file))
        for each in f.readlines():
            each = each.split(" ")
            user_id = int(each[0])
            item_ids = map(int, each[1:])
            feature = {'user_id': _int64_feature(user_id),
                       'item_ids': _int64_feature(item_ids)}
            # Create an example protocol buffer
            example = tf.train.Example(features=tf.train.Features(feature=feature))
            writer.write(example.SerializeToString())
        writer.close()
