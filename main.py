import tensorflow as tf

import numpy as np
import csv
import os
import mmh3
from tqdm import tqdm

import logging

import neuralgym as ng

logger = logging.getLogger()



D = 2**21
data_path = '/media/george/1a4f4334-123f-430d-8a2b-f0c0fa401c75/advert/' 
train_path = os.path.join(data_path, 'train.csv')
resampled_train_path = os.path.join(data_path, 'resampled_train.csv')
submission_path = os.path.join(data_path, 'submission.csv')
test_path = os.path.join(data_path, 'test.csv')
one_constant = tf.constant([1], dtype=tf.float32)
dense_shape = tf.constant([1, D + 2], dtype=tf.int64)
r = 50

def get_stats(path):
    a = [0, 0]
    with open(path, 'r') as f:
        reader = csv.reader(f, delimiter=';')
        for row in reader:
            break
        for i, row in enumerate(reader):
            a[int(row[1])] += 1
            if i % 1000000 == 0:
                print(i, a, a[0] / (a[1] + 1e-10))
    print(a)

def resample(path, new_path):
    a = [0, 0]
    with open(path, 'r') as f, open(resampled_train_path, 'w') as g:
        reader = csv.reader(f, delimiter=';')
        writer = csv.writer(g, delimiter=';')
        for row in reader:
            writer.writerow(row)
            break
        for row in tqdm(reader):
            label = int(row[1])
            a[label] += 1
            if label == 1 or a[0] % r == 0:
                writer.writerow(row)
            
       
def make_submission_file(click_probs):
    with open(submission_path, 'w') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerow(['Id', 'Click'])
        [writer.writerow([i + 1, prob]) for i, prob in enumerate(click_probs)]

def l1_to_sparse(l):
    l = tf.string_to_number(l)
    l = tf.reshape(l, [1])
    return tf.SparseTensor(indices=tf.constant([[0, D]], dtype=tf.int64), values=l, dense_shape=dense_shape)

def l2_to_sparse(l):
    l = tf.string_to_number(l)
    l = tf.reshape(l, [1])
    return tf.SparseTensor(indices=tf.constant([[0, D + 1]], dtype=tf.int64), values=l, dense_shape=dense_shape)

def cg_to_sparse(cg):
    def special_hash(x):
        x = str(x).split(',')
        x = ','.join(sorted(x))
        return mmh3.hash(x)
    h = tf.py_func(lambda x : np.int64([0, special_hash(x) % D]), [cg], tf.int64)
    h = tf.reshape(h, [1, 2])
    return tf.SparseTensor(indices=h, values=one_constant, dense_shape=dense_shape)

def c_to_sparse(c):
    h = tf.py_func(lambda x : np.int64([0, mmh3.hash(x) % D]), [c], tf.int64)
    h = tf.reshape(h, [1, 2])
    return tf.SparseTensor(indices=h, values=one_constant, dense_shape=dense_shape)

def sparse_sum_list(lst):
    res = lst[0]
    for i in range(1, len(lst)):
        res = tf.sparse_add(res, lst[i])
    return res

def parse_row(csv_row):
    record_defaults = [['']] * 19
    _, label, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, cg1, cg2, cg3, l1, l2, c11, c12 = tf.decode_csv(
        csv_row, 
        record_defaults=record_defaults, 
        field_delim=';'
    )
    label = tf.string_to_number(label)
    features = [
        c_to_sparse(x) for x in [c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12]
    ] + [
        cg_to_sparse(x) for x in [cg1, cg2, cg3]
    ] + [ 
        l1_to_sparse(l1), l2_to_sparse(l2)
    ]
    features = sparse_sum_list(features)
    return features, label

def make_dataset(filename, batch_size, epoch, test=False):
    min_after_dequeue = 20000
    capacity = min_after_dequeue + 3 * batch_size

    filename_queue = tf.train.string_input_producer([filename], num_epochs=epoch, shuffle=False)
    reader = tf.TextLineReader(skip_header_lines=1)
    _, csv_row = reader.read(filename_queue)
    #csv_row = tf.Print(csv_row, [csv_row])
    features, label = parse_row(csv_row)
    if test:
        feature_batch, label_batch = tf.train.batch(
            [features, label], 
            batch_size=batch_size, 
            capacity=capacity,
            enqueue_many=False,
            num_threads=8,
            allow_smaller_final_batch=True,
        )
    else:
        feature_batch, label_batch = tf.train.batch(
            [features, label], 
            batch_size=batch_size, 
            capacity=capacity,
            enqueue_many=False,
            num_threads=8,
            allow_smaller_final_batch=True,
        )

    feature_batch = tf.sparse_reshape(feature_batch, [batch_size, D + 2]) 
    label_batch = tf.reshape(label_batch, [batch_size, 1])
    return feature_batch, label_batch

def build_model(x, name='log_reg', reuse=True):
    with tf.variable_scope(name, reuse=reuse):
        w = tf.get_variable('w', shape=[D + 2, 1], dtype=tf.float32, initializer=tf.random_normal_initializer(), trainable=True)
        b = tf.get_variable('b', shape=[1, 1], dtype=tf.float32, initializer=tf.zeros_initializer(), trainable=True)
        y_pred = tf.nn.sigmoid(tf.sparse_tensor_dense_matmul(x, w) + b)
    return y_pred

def build_model_for_train(x, y, reuse=True):
    losses = {}
    y_pred = build_model(x, reuse=reuse)
    losses['log_loss'] = tf.losses.log_loss(labels=y, predictions=y_pred)
    return losses

def build_model_for_infer(x, reuse=True):
    y_pred = build_model(x, reuse=reuse)
    return y_pred / (r * (1 - y_pred + y_pred / r))

def train():
    config = ng.Config('config.yml')
    epoch = None
    batch_size = config.BATCH_SIZE
    x, y = make_dataset(resampled_train_path, batch_size, epoch)
    losses = build_model_for_train(x, y, reuse=False)
    lr = tf.get_variable('lr', shape=[], trainable=False, initializer=tf.constant_initializer(2e-3))
    optimizer = tf.train.AdamOptimizer(lr, beta1=0.5, beta2=0.9)
    #optimizer = tf.train.FtrlOptimizer(lr)
    var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='log_reg')
    print(var_list)
    log_prefix = 'model_logs/' + '_'.join([
        ng.date_uid(), 
        config.LOG_DIR
    ])
    trainer = ng.train.Trainer(
        optimizer=optimizer,
        var_list=var_list,
        max_iters=100000000,
        graph_def=lambda : losses['log_loss'],
        grads_summary=False,
        graph_def_kwargs={
        },
        spe=config.TRAIN_SPE,
        log_dir=log_prefix,
        log_progress=True
    )
    trainer.add_callbacks([
        ng.callbacks.WeightsViewer(),
        ng.callbacks.ModelRestorer(trainer.context['saver'], dump_prefix='model_logs/'+config.MODEL_RESTORE, optimistic=True),
        ng.callbacks.ModelSaver(config.TRAIN_SPE, trainer.context['saver'], log_prefix+'/snap'),
        ng.callbacks.SummaryWriter(config.VAL_PSTEPS, trainer.context['summary_writer'], tf.summary.merge_all()),
    ])
    trainer.train()

def make_submission():
    config = ng.Config('config.yml')
    batch_size = 1#32768
    x, _ = make_dataset(test_path, batch_size, 1, test=True)
    y_pred = build_model_for_infer(x, reuse=False)
    vars_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='log_reg')
    print("vars_list", vars_list)
    assign_ops = []
    for var in vars_list:
        vname = var.name
        from_name = vname
        var_value = tf.contrib.framework.load_variable('model_logs/' + config.MODEL_RESTORE, from_name)
        assign_ops.append(tf.assign(var, var_value))

    with tf.Session() as sess, open(submission_path, 'w') as f:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        writer = csv.writer(f, delimiter=',')
        writer.writerow(['Id', 'Click'])
        print("model loaded")
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        sess.run(assign_ops)
        _y_pred = None
        for i in tqdm(range(10000000000)):
            try:
                _y_pred = sess.run(y_pred)
                for j in range(batch_size):
                    writer.writerow([batch_size * i + j + 1, _y_pred[j, 0]])
            except:
                print(_y_pred)
                for j in range(batch_size):
                    writer.writerow([batch_size * i + j + 1, _y_pred[j, 0]])

                break
        coord.request_stop()
        coord.join(threads)

def test_dataset():
    import time
    from tqdm import tqdm
    batch_size = 64
    N = 1000
    fb, lb = make_dataset(train_path, batch_size)
    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        start_time = time.time()
        for i in tqdm(range(N)):
            sess.run([fb, lb])
        dt = time.time() - start_time
        print("--- %s seconds ---" % (dt))
        print("--- %s seconds/row ---" % (dt / (N * batch_size)))
        coord.request_stop()
        coord.join(threads)

def test1_dataset():
    import time
    # from tqdm import tqdm
    batch_size = 10
    fb, lb = make_dataset(train_path, batch_size, 1, test=False)
    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        start_time = time.time()
        print(sess.run([fb, lb]))
        dt = time.time() - start_time
        print("--- %s seconds ---" % (dt))
        coord.request_stop()
        coord.join(threads)



if __name__ == '__main__':
    #resample(train_path, resampled_train_path)
    train()
    #make_submission()
    #test1_dataset()
