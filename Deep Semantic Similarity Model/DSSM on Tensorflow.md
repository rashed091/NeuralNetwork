# Model DSSM on Tensorflow

Jun 21, 2016

Now with tensorflow installed, we now try to implement our first model on tensorflow. Instead of famous neural networks like LeNet, AlexNet, GoogleNet or ResNet, we choose a very simple but powerful model named named **DSSM** (Deep Structured Semantic Models) for matching web search queries and url based documents. The paper describing the model is published on CIKM'13 and available [here](https://posenhuang.github.io/papers/cikm2013_DSSM_fullversion.pdf).

## DSSM model structure

![Illustration of the DSSM](https://raw.githubusercontent.com/v-liaha/v-liaha.github.io/master/assets/dssm.png)

### Model Input

The input of DSSM model are queries and documents mapped on so-called *n-gram* spaces instead of traditional word spaces. N-grams are defined as a sequence of n letters. A word, for example 'good', is firsted attached with hashtags on both sides to '#good#'. Then it is mapped into a list of 3-grams or trigrams: (*#go, goo, ood, od#*). As the number of characters are limited, the number of possible n-grams are limited as well, which is much smaller compared to number of words available. Typically as shown in Figure 1, the term vector with 500K words can be mapped to n-gram vectors sized only aroudn 30K.

One query and several documents are input to the model at the same time. Only one of the documents, D1, is most related to the query, being the positive document. The other documents are all negative documents, that are not related to the query. A typical query/doc input is shown below as pairs of **N:X**, saying the **N**th n-gram appears **X** times in the query/doc.

```
46238:1 24108:1 24016:1 5618:1 8818:1

```

### Neural Network

There are 3 fully connected (FC) layers in the network, with 300, 300, 128 neurons in each layer. Each input **x** is projected linearly by $$Wx+b$$ and then activated non-linearly with  $$tanh/relu$$ functions to generate input for the next layer.

First Layer: 

​					$$ l_1 = W_1x+b_1$$
Second Layer:

​					 $$ l_2 = f(W_2l_1+b_2) $$
Last Layer: 

​					$$ y = f(W_3l_2+b_3) $$

The output of the FC layers is a 128-length vector and fed to calculate cosine similarities. The cosine-similarity between the query and each document is calculated as:

​			$$ R(Q,D) = cosine(y_Q,y_D) = \frac{y_Q \cdot y_D}{\Vert y_Q \Vert \cdot \Vert y_D \Vert} $$

### Learning the DSSM

For *m* documents, there are *m* cosine similarity values, composing the logit vector. The score for each document is calculated as the posterior probability:

​			$$P(D \vert Q) = \frac{\gamma e^{R(Q,D)}}{\sum_{D;\in\mathbf{D}} \gamma e^{R(Q,D;)}} $$

The loss function is finally defined as:

​		$$ L(\Lambda) = -log\prod_{(Q,D^+)} P(D^+\vert Q)$$

## Tensorflow Implementation

### Import Tensorflow

```
import tensorflow as tf

```

### Input Batching and Sparsifying

To fully utilize the GPU capability, we feed the model with Q and D in batches of size **BS**. The original input vector [TRIGRAM_D] is now a matrix with size shaped [BS, TRIGRAM_D]. TRIGRAM_D is the total number of trigrams appear in all queries and documents.

Another problem is that the input matrix is very sparse. We find that 80% of the queries can be composed of less than 30 trigrams, which makes most of the input matrix values zero.

Tensorflow supports sparse placeholders, which are used to hold the input tensors:

```
query_batch = tf.sparse_placeholder(tf.float32, 
                                    shape=[None,TRIGRAM_D], 
                                    name='QueryBatch')
doc_batch = tf.sparse_placeholder(tf.float32, 
                                    shape=[None, TRIGRAM_D], 
                                    name='DocBatch')

```

### Initialize Weight and Bias

Then the weights and biases are specified for each layer, here only layer 1 is presented. Weights and biases are initialized in the uniform distribution as decribed in the paper.

```
# L1_N = 300
l1_par_range = np.sqrt(6.0 / (TRIGRAM_D + L1_N))
weight1 = tf.Variable(tf.random_uniform([TRIGRAM_D, L1_N], 
                                        -l1_par_range, 
                                        l1_par_range))
bias1 = tf.Variable(tf.random_uniform([L1_N], 
                                       -l1_par_range, 
                                       l1_par_range))

```

### Define FC Layer Operations

Generate the activations (output of the layer) using the sparse-dense matrix multiplication operator and the relu activation function:

```
query_l1 = tf.sparse_tensor_dense_matmul(query_batch, weight1) + bias1
doc_l1 = tf.sparse_tensor_dense_matmul(doc_batch, weight1) + bias1

query_l1_out = tf.nn.relu(query_l1)
doc_l1_out = tf.nn.relu(doc_l1)

```

### Cosine Similarity

After L2 and L3, we get **y**, the output of 3 FC layers, and feed it to cosine-similarity. Here we only caculate yQyQ once and duplicate it for the calculation of cosine-similarity.

```
# NEG is the number of negative documents
query_norm = tf.tile(tf.sqrt(tf.reduce_sum(tf.square(query_y), 1, True)), 
                     [NEG + 1, 1])
doc_norm = tf.sqrt(tf.reduce_sum(tf.square(doc_y), 1, True))

prod = tf.reduce_sum(tf.mul(tf.tile(query_y, [NEG + 1, 1]), doc_y), 1, True)
norm_prod = tf.mul(query_norm, doc_norm)

cos_sim_raw = tf.truediv(prod, norm_prod)
cos_sim = tf.transpose(tf.reshape(tf.transpose(cos_sim_raw), [NEG + 1, BS])) * Gamma

```

### Loss

Finally, the loss is calculated and averaged by the batch size.

```
prob = tf.nn.softmax((cos_sim))
hit_prob = tf.slice(prob, [0, 0], [-1, 1])
loss = -tf.reduce_sum(tf.log(hit_prob)) / BS

```

### Training in One Line!

Training using gradient descent:

```
train_step = tf.train.GradientDescentOptimizer(FLAGS.learning_rate).minimize(loss)

```

### Actually Run DSSM Model

Running Tensorflow with a session:

```
# Allow GPU to allocate memory dynamically 
# instead of taking all memory at once in 
# the beginning
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

with tf.Session(config=config) as sess:
    sess.run(tf.initialize_all_variables())
    for step in range(FLAGS.max_steps):
        sess.run(train_step, feed_dict={query_batch : ...
                                        doc_batch   : ...}})

```

That's it! The code structure is very clear and we don't even need to compose the training part. Tensorflow is capable of constructing the training operation graph automatically. The full code is publicly available in my [github page](https://github.com/v-liaha/tensorflow/blob/r0.9/tensorflow/models/dssm/dssm.py).

## Tensorboard Visualization

We have worked some tricks to change the model a little bit, and one of the advantages of tensorflow is that it provides a visualization called tensorboard to visualize the network you created. I may provide more information on it, but now I'm just providing the visualization of our DSSM network structure:

![Tensorboard Visualization](https://raw.githubusercontent.com/v-liaha/v-liaha.github.io/master/assets/dssm-tensorboard.png)

### **Full Code**

```python
import pickle
import random
import time
import sys
import numpy as np
import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('summaries_dir', '/tmp/dssm-400-120-relu', 'Summaries directory')
flags.DEFINE_float('learning_rate', 0.1, 'Initial learning rate.')
flags.DEFINE_integer('max_steps', 900000, 'Number of steps to run trainer.')
flags.DEFINE_integer('epoch_steps', 18000, "Number of steps in one epoch.")
flags.DEFINE_integer('pack_size', 2000, "Number of batches in one pickle pack.")
flags.DEFINE_bool('gpu', 1, "Enable GPU or not")

start = time.time()

doc_train_data = None
query_train_data = None

# load test data for now
query_test_data = pickle.load(open('../data/query.test.pickle', 'rb')).tocsr()
doc_test_data = pickle.load(open('../data/doc.test.pickle', 'rb')).tocsr()

doc_train_data = pickle.load(open('../data/doc.train.pickle', 'rb')).tocsr()
query_train_data = pickle.load(open('../data/query.train.pickle', 'rb')).tocsr()

end = time.time()
print("Loading data from HDD to memory: %.2fs" % (end - start))

TRIGRAM_D = 49284

NEG = 50
BS = 1024

L1_N = 400
L2_N = 120

query_in_shape = np.array([BS, TRIGRAM_D], np.int64)
doc_in_shape = np.array([BS, TRIGRAM_D], np.int64)


def variable_summaries(var, name):
    """Attach a lot of summaries to a Tensor."""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.scalar_summary('mean/' + name, mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_sum(tf.square(var - mean)))
        tf.scalar_summary('sttdev/' + name, stddev)
        tf.scalar_summary('max/' + name, tf.reduce_max(var))
        tf.scalar_summary('min/' + name, tf.reduce_min(var))
        tf.histogram_summary(name, var)


with tf.name_scope('input'):
    # Shape [BS, TRIGRAM_D].
    query_batch = tf.sparse_placeholder(tf.float32, shape=query_in_shape, name='QueryBatch')
    # Shape [BS, TRIGRAM_D]
    doc_batch = tf.sparse_placeholder(tf.float32, shape=doc_in_shape, name='DocBatch')

with tf.name_scope('L1'):
    l1_par_range = np.sqrt(6.0 / (TRIGRAM_D + L1_N))
    weight1 = tf.Variable(tf.random_uniform([TRIGRAM_D, L1_N], -l1_par_range, l1_par_range))
    bias1 = tf.Variable(tf.random_uniform([L1_N], -l1_par_range, l1_par_range))
    variable_summaries(weight1, 'L1_weights')
    variable_summaries(bias1, 'L1_biases')

    # query_l1 = tf.matmul(tf.to_float(query_batch),weight1)+bias1
    query_l1 = tf.sparse_tensor_dense_matmul(query_batch, weight1) + bias1
    # doc_l1 = tf.matmul(tf.to_float(doc_batch),weight1)+bias1
    doc_l1 = tf.sparse_tensor_dense_matmul(doc_batch, weight1) + bias1

    query_l1_out = tf.nn.relu(query_l1)
    doc_l1_out = tf.nn.relu(doc_l1)

with tf.name_scope('L2'):
    l2_par_range = np.sqrt(6.0 / (L1_N + L2_N))

    weight2 = tf.Variable(tf.random_uniform([L1_N, L2_N], -l2_par_range, l2_par_range))
    bias2 = tf.Variable(tf.random_uniform([L2_N], -l2_par_range, l2_par_range))
    variable_summaries(weight2, 'L2_weights')
    variable_summaries(bias2, 'L2_biases')

    query_l2 = tf.matmul(query_l1_out, weight2) + bias2
    doc_l2 = tf.matmul(doc_l1_out, weight2) + bias2
    query_y = tf.nn.relu(query_l2)
    doc_y = tf.nn.relu(doc_l2)

with tf.name_scope('FD_rotate'):
    # Rotate FD+ to produce 50 FD-
    temp = tf.tile(doc_y, [1, 1])

    for i in range(NEG):
        rand = int((random.random() + i) * BS / NEG)
        doc_y = tf.concat(0,
                          [doc_y,
                           tf.slice(temp, [rand, 0], [BS - rand, -1]),
                           tf.slice(temp, [0, 0], [rand, -1])])

with tf.name_scope('Cosine_Similarity'):
    # Cosine similarity
    query_norm = tf.tile(tf.sqrt(tf.reduce_sum(tf.square(query_y), 1, True)), [NEG + 1, 1])
    doc_norm = tf.sqrt(tf.reduce_sum(tf.square(doc_y), 1, True))

    prod = tf.reduce_sum(tf.mul(tf.tile(query_y, [NEG + 1, 1]), doc_y), 1, True)
    norm_prod = tf.mul(query_norm, doc_norm)

    cos_sim_raw = tf.truediv(prod, norm_prod)
    cos_sim = tf.transpose(tf.reshape(tf.transpose(cos_sim_raw), [NEG + 1, BS])) * 20

with tf.name_scope('Loss'):
    # Train Loss
    prob = tf.nn.softmax((cos_sim))
    hit_prob = tf.slice(prob, [0, 0], [-1, 1])
    loss = -tf.reduce_sum(tf.log(hit_prob)) / BS
    tf.scalar_summary('loss', loss)

with tf.name_scope('Training'):
    # Optimizer
    train_step = tf.train.GradientDescentOptimizer(FLAGS.learning_rate).minimize(loss)

# with tf.name_scope('Accuracy'):
#     correct_prediction = tf.equal(tf.argmax(prob, 1), 0)
#     accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#     tf.scalar_summary('accuracy', accuracy)

merged = tf.merge_all_summaries()

with tf.name_scope('Test'):
    average_loss = tf.placeholder(tf.float32)
    loss_summary = tf.scalar_summary('average_loss', average_loss)


def pull_batch(query_data, doc_data, batch_idx):
    # start = time.time()
    query_in = query_data[batch_idx * BS:(batch_idx + 1) * BS, :]
    doc_in = doc_data[batch_idx * BS:(batch_idx + 1) * BS, :]
    
    if batch_idx == 0:
      print(query_in.getrow(53))
    query_in = query_in.tocoo()
    doc_in = doc_in.tocoo()
    
    

    query_in = tf.SparseTensorValue(
        np.transpose([np.array(query_in.row, dtype=np.int64), np.array(query_in.col, dtype=np.int64)]),
        np.array(query_in.data, dtype=np.float),
        np.array(query_in.shape, dtype=np.int64))
    doc_in = tf.SparseTensorValue(
        np.transpose([np.array(doc_in.row, dtype=np.int64), np.array(doc_in.col, dtype=np.int64)]),
        np.array(doc_in.data, dtype=np.float),
        np.array(doc_in.shape, dtype=np.int64))

    # end = time.time()
    # print("Pull_batch time: %f" % (end - start))

    return query_in, doc_in


def feed_dict(Train, batch_idx):
    """Make a TensorFlow feed_dict: maps data onto Tensor placeholders."""
    if Train:
        query_in, doc_in = pull_batch(query_train_data, doc_train_data, batch_idx)
    else:
        query_in, doc_in = pull_batch(query_test_data, doc_test_data, batch_idx)
    return {query_batch: query_in, doc_batch: doc_in}


config = tf.ConfigProto()  # log_device_placement=True)
config.gpu_options.allow_growth = True
#if not FLAGS.gpu:
#config = tf.ConfigProto(device_count= {'GPU' : 0})

with tf.Session(config=config) as sess:
    sess.run(tf.initialize_all_variables())
    train_writer = tf.train.SummaryWriter(FLAGS.summaries_dir + '/train', sess.graph)
    test_writer = tf.train.SummaryWriter(FLAGS.summaries_dir + '/test', sess.graph)

    # Actual execution
    start = time.time()
    # fp_time = 0
    # fbp_time = 0
    for step in range(FLAGS.max_steps):
        batch_idx = step % FLAGS.epoch_steps
        # if batch_idx % FLAGS.pack_size == 0:
        #    load_train_data(batch_idx / FLAGS.pack_size + 1)

            # # setup toolbar
            # sys.stdout.write("[%s]" % (" " * toolbar_width))
            # #sys.stdout.flush()
            # sys.stdout.write("\b" * (toolbar_width + 1))  # return to start of line, after '['
        if batch_idx == 0:
            temp = sess.run(query_y, feed_dict=feed_dict(True, 0))
            print(np.count_nonzero(temp))
            sys.exit()

        if batch_idx % (FLAGS.pack_size / 64) == 0:
            progress = 100.0 * batch_idx / FLAGS.epoch_steps
            sys.stdout.write("\r%.2f%% Epoch" % progress)
            sys.stdout.flush()

        # t1 = time.time()
        # sess.run(loss, feed_dict = feed_dict(True, batch_idx))
        # t2 = time.time()
        # fp_time += t2 - t1
        # #print(t2-t1)
        # t1 = time.time()
        sess.run(train_step, feed_dict=feed_dict(True, batch_idx % FLAGS.pack_size))
        # t2 = time.time()
        # fbp_time += t2 - t1
        # #print(t2 - t1)
        # if batch_idx % 2000 == 1999:
        #     print ("MiniBatch: Average FP Time %f, Average FP+BP Time %f" %
        #        (fp_time / step, fbp_time / step))


        if batch_idx == FLAGS.epoch_steps - 1:
            end = time.time()
            epoch_loss = 0
            for i in range(FLAGS.pack_size):
                loss_v = sess.run(loss, feed_dict=feed_dict(True, i))
                epoch_loss += loss_v

            epoch_loss /= FLAGS.pack_size
            train_loss = sess.run(loss_summary, feed_dict={average_loss: epoch_loss})
            train_writer.add_summary(train_loss, step + 1)

            # print ("MiniBatch: Average FP Time %f, Average FP+BP Time %f" %
            #        (fp_time / step, fbp_time / step))
            #
            print ("\nEpoch #%-5d | Train Loss: %-4.3f | PureTrainTime: %-3.3fs" %
                    (step / FLAGS.epoch_steps, epoch_loss, end - start))

            epoch_loss = 0
            for i in range(FLAGS.pack_size):
                loss_v = sess.run(loss, feed_dict=feed_dict(False, i))
                epoch_loss += loss_v

            epoch_loss /= FLAGS.pack_size

            test_loss = sess.run(loss_summary, feed_dict={average_loss: epoch_loss})
            test_writer.add_summary(test_loss, step + 1)

            start = time.time()
            print ("Epoch #%-5d | Test  Loss: %-4.3f | Calc_LossTime: %-3.3fs" %
                   (step / FLAGS.epoch_steps, epoch_loss, start - end))
```

