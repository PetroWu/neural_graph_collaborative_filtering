import tensorflow as tf
import scipy.sparse as sp
import numpy as np


def file_based_input_fn_builder(input_files, is_training, vocab, batch_size):
    """Creates an `input_fn` closure to be passed to TPUEstimator."""

    name_to_features = {
        "user_id": tf.FixedLenFeature([1], tf.int64),
        "item_ids": tf.VarLenFeature(tf.int64),
    }

    def _decode_record(record, name_to_features):
        """Decodes a record to a TensorFlow example."""
        example = tf.parse_single_example(record, name_to_features)

        # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
        # So cast all int64 to int32.
        for name in list(example.keys()):
            t = example[name]
            example[name] = t

        return example["user_id"], example["item_ids"].values

    def input_fn():
        """The actual input function."""
        # For training, we want a lot of parallel reading and shuffling.
        # For eval, we want no shuffling and parallel reading doesn't matter.
        d = tf.data.TFRecordDataset(input_files)
        if is_training:
            d = d.repeat()
            d = d.shuffle(buffer_size=100)

        d = d.map(lambda record: _decode_record(record, name_to_features))

        d = d.map(lambda user_id, item_ids: (
            tf.tile(user_id, [tf.size(item_ids)]),
            item_ids,
            tf.random_uniform(shape=(tf.size(item_ids),), maxval=vocab,dtype=tf.int32)
            ))

        d = d.flat_map(lambda user_ids, item_ids, neg_ids:
                       tf.data.Dataset.from_tensor_slices((user_ids, item_ids, neg_ids))
                       )
        d = d.batch(batch_size=batch_size)

        features = d.map(lambda user_id, item_id, neg_id: {
            "user_id": user_id,
            "item_id": item_id,
            "neg_id": neg_id
        })

        return features

    return input_fn


def negative_sampling(item_ids, vocab):
    #item_ids = tf.squeeze(item_ids, axis=1)
    mask = tf.scatter_nd(item_ids, updates=tf.ones_like(item_ids), shape=tf.constant([vocab]))
    candidates = tf.boolean_mask(tf.range(vocab), mask - 1)
    return tf_random_choice(candidates, 1)[0]


def tf_random_choice(input, sample_num):
    idx = tf.range(tf.shape(input)[0])
    ridxs = tf.random_shuffle(idx)[:sample_num]
    rinput = tf.gather(input, ridxs)
    return rinput


def get_adj_mat(path, n_users, n_items):
    try:
        adj_mat = sp.load_npz(path + '/s_adj_mat.npz')
        norm_adj_mat = sp.load_npz(path + '/s_norm_adj_mat.npz')
        mean_adj_mat = sp.load_npz(path + '/s_mean_adj_mat.npz')

    except Exception:
        adj_mat, norm_adj_mat, mean_adj_mat = create_adj_mat(n_users, n_items)
        sp.save_npz(path + '/s_adj_mat.npz', adj_mat)
        sp.save_npz(path + '/s_norm_adj_mat.npz', norm_adj_mat)
        sp.save_npz(path + '/s_mean_adj_mat.npz', mean_adj_mat)
    return adj_mat, norm_adj_mat, mean_adj_mat


def create_adj_mat(n_users, n_items):
    adj_mat = sp.dok_matrix((n_users + n_items, n_users + n_items), dtype=np.float32)
    adj_mat = adj_mat.tolil()
    R = sp.dok_matrix((n_users, n_items), dtype=np.float32)

    adj_mat[:n_users, n_users:] = R
    adj_mat[n_users:, :n_users] = R.T
    adj_mat = adj_mat.todok()

    def normalized_adj_single(adj):
        rowsum = np.array(adj.sum(1))

        d_inv = np.power(rowsum, -1).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = sp.diags(d_inv)

        norm_adj = d_mat_inv.dot(adj)
        return norm_adj.tocoo()

    def check_adj_if_equal(adj):
        dense_A = np.array(adj.todense())
        degree = np.sum(dense_A, axis=1, keepdims=False)

        temp = np.dot(np.diag(np.power(degree, -1)), dense_A)
        return temp

    norm_adj_mat = normalized_adj_single(adj_mat + sp.eye(adj_mat.shape[0]))
    mean_adj_mat = normalized_adj_single(adj_mat)

    return adj_mat.tocsr(), norm_adj_mat.tocsr(), mean_adj_mat.tocsr()


if __name__ == '__main__':
    inputs_fn = \
        file_based_input_fn_builder("/Users/chris"
                                    "/PycharmProjects"
                                    "/neural_graph_collaborative_filtering"
                                    "/Data/amazon-book/train.tfrecords", True, 91598)
    dataset = inputs_fn()
    dataset = dataset.batch(batch_size=10)
    iterator = dataset.make_one_shot_iterator()
    next_element = iterator.get_next()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(3):
            print(sess.run(next_element))


