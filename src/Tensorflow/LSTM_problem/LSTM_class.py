import numpy as np
import tensorflow as tf
from tensorflow.python.ops import math_ops
import os


class RNN:
    
    def __init__(self, value = None, input_size=1, output_size=1, lstm_size=128, num_layers=1,
                 num_steps= None, keep_prob=0.8, batch_size=64, init_learning_rate=0.5,
                 learning_rate_decay=0.99, init_epoch=5, max_epoch=100, MODEL_DIR = None, name = 'LSTM_default'):
        self.value = value
        self.input_size = input_size 
        self.output_size = output_size
        self.lstm_size = lstm_size
        self.num_layers = num_layers
        self.num_steps = num_steps
        self.keep_prob = keep_prob    
        self.batch_size = batch_size
        self.init_learning_rate = init_learning_rate
        self.learning_rate_decay = learning_rate_decay
        self.init_epoch = init_epoch
        self.max_epoch = max_epoch
        self.name = name
        self.MODEL_DIR = MODEL_DIR
        
        
    """ LEARNING RATES TO USE """         
    def _compute_learning_rates(self):
        
        learning_rates_to_use = [
        self.init_learning_rate * (
                self.learning_rate_decay ** max(float(i + 1 - self.init_epoch), 0.0))
                for i in range(self.max_epoch)]
       
        print("Middle learning rate:", learning_rates_to_use[len(learning_rates_to_use) // 2])
        return learning_rates_to_use


    """ BUILD GRAPH """
    def build_lstm_graph_with_config(self):
        
        """ Reset and define new graph """
        tf.reset_default_graph()
        self.lstm_graph = tf.Graph()
        
        
        """ Initialize new LSTM graph as default """
        with self.lstm_graph.as_default():
            
            """ Define trapezoidal integral approximation function """
            def trapezoidal_integral_approx(t, y):
                return math_ops.reduce_sum(math_ops.multiply(t[1:] - t[:-1],
                                  (y[:-1] + y[1:]) / 2.), name='integral')
                
            """ Define sequence length """
            def length(sequence):
                  used = tf.sign(tf.reduce_max(tf.abs(sequence), 2))
                  length = tf.reduce_sum(used, 1)
                  length = tf.cast(length, tf.int32)
                  return length
            
            """ Create one LSTM cell with or without dropout """
            def _create_one_cell():
                lstm_cell = tf.contrib.rnn.BasicRNNCell(self.lstm_size, state_is_tuple=True)
                if self.keep_prob < 1.0:
                    lstm_cell = tf.contrib.rnn.DropoutWrapper(lstm_cell, output_keep_prob = self.keep_prob)
                return lstm_cell
            
            """ Define placeholder for learning rate, inputs, targets, time 
                targets[binary] - places where events occured 
                time - vector of time with defined step """
                
            learning_rate = tf.placeholder(tf.float32, None, name="learning_rate")
    
            inputs = tf.placeholder(tf.float32, [None,None, self.input_size], name="inputs")
            targets = tf.placeholder(tf.float32, [None, self.num_steps, self.output_size], name="targets")
            targets = tf.constant(self.value ,dtype = tf.float32, name="targets")
            time = tf.placeholder(tf.float32, shape = None, name = 'time')
            
            """ Define number of examples """
            num_examples = tf.shape(inputs)[0]
            
            """ Define cell """
            cell = tf.contrib.rnn.MultiRNNCell(
                [_create_one_cell() for _ in range(self.num_layers)],
                state_is_tuple=True
            ) if self.num_layers > 1 else _create_one_cell()
            
            seq_len = length(inputs)
            val, _ = tf.nn.dynamic_rnn(cell, inputs, dtype=tf.float32, sequence_length=seq_len, scope="rnn")
    
            # val.get_shape() = (batch_size, num_steps, lstm_size) 
    
            """ Define output layer """
            with tf.name_scope("output_layer"):
                weight = tf.Variable(tf.random_normal([self.lstm_size, self.output_size]), name="weights")
                self.weight = weight
                weight_repeated = tf.tile(tf.expand_dims(weight, 0), [num_examples, 1, 1])
                bias = tf.Variable(tf.constant(0.1, shape=[self.output_size]), name="biases")
                prediction = tf.squeeze(tf.math.exp(tf.add(tf.matmul(val, weight_repeated),bias)), name="prediction")
                outputs_train = tf.multiply(prediction, targets, name='outputs_train')
                nonzero_indices = tf.where(tf.not_equal(outputs_train, tf.zeros_like(outputs_train)))
                outputs_train_nonzero =  tf.gather_nd(outputs_train,nonzero_indices, name='outputs_train_nonzero')
                
                tf.summary.histogram("prediction", prediction)
                tf.summary.histogram("weights", weight)
                tf.summary.histogram("biases", bias)
    
            """ Define train scope """
            with tf.name_scope("train"):
                integral = trapezoidal_integral_approx(time, prediction)
                
                outputs_sum_neg = tf.math.negative(tf.reduce_sum(tf.math.log(outputs_train_nonzero)))

                loss = tf.reduce_sum(tf.add(outputs_sum_neg,integral), name="loss_mse")
                optimizer = tf.train.AdamOptimizer(learning_rate)
                minimize = optimizer.minimize(loss, name="loss_mse_adam_minimize")
                
                tf.summary.scalar("loss_mse", loss)
    
            """ Operators to use after restoring the model """
            for op in [prediction, loss]:
                tf.add_to_collection('ops_to_restore', op)
    
        return self.lstm_graph


    """ TRAIN GRAPH """    
    def train_lstm_graph(self, train_X, train_Targets, interval):
        """
        name (str)
        lstm_graph (tf.Graph)
        """
        def batches(x,y, batchsize):
            for i in range(0, x.shape[0], batchsize):
                yield x[i:i+batchsize], y[i:i+batchsize]
        
        self.graph_name = "%s_lr%.2f_lr_decay%.3f_lstm%d_step%d_input%d_batch%d_epoch%d_kp%.3f_layer%d" % (
            self.name,
            self.init_learning_rate, self.learning_rate_decay,
            self.lstm_size, self.num_steps,
            self.input_size, self.batch_size, self.max_epoch, self.keep_prob, self.num_layers)
    
        print("Graph Name:", self.graph_name)
    
        learning_rates_to_use = RNN._compute_learning_rates(self)
        
        with tf.Session(graph = self.lstm_graph) as sess:
            
            merged_summary = tf.summary.merge_all()
            writer = tf.summary.FileWriter('_logs/' + self.graph_name, sess.graph)
    
            graph = tf.get_default_graph()
            tf.global_variables_initializer().run()
    
            inputs = graph.get_tensor_by_name('inputs:0')
            time = graph.get_tensor_by_name('time:0')
            learning_rate = graph.get_tensor_by_name('learning_rate:0')
    

            loss = graph.get_tensor_by_name('train/loss_mse:0')
            minimize = graph.get_operation_by_name('train/loss_mse_adam_minimize')            

            for epoch_step in range(self.max_epoch):
                current_lr = learning_rates_to_use[epoch_step]
                
                if train_X.shape[0]>1:
                    for batch_X, batch_y in list(batches(train_X, train_Targets, self.batch_size)):
                        train_data_feed = {
                            inputs: batch_X,
                            time: interval,
                            learning_rate: current_lr}
                        train_loss, _ = sess.run([loss, minimize], train_data_feed)
                else:
                    train_data_feed = {
                            inputs: train_X,
                            time: interval,
                            learning_rate: current_lr}
                    train_loss, _ = sess.run([loss, minimize], train_data_feed)
                
                _summary = sess.run(merged_summary, train_data_feed)
                print("Epoch %d [%f]:" % (epoch_step, current_lr), train_loss)
    
                writer.add_summary(_summary, global_step=epoch_step)
    
            graph_saver_dir = os.path.join(self.MODEL_DIR, self.graph_name)
            
            if not os.path.exists(graph_saver_dir):
                os.mkdir(graph_saver_dir)
    
            saver = tf.train.Saver()
            saver.save(sess, os.path.join(
                graph_saver_dir, "LSTM_model_%s.ckpt" % self.name), global_step=epoch_step)
            
  
          
    """ PREDICT """         
    def prediction_by_trained_graph(self, max_epoch, test_X):
        test_prediction = None
    
        with tf.Session() as sess:

            graph_meta_path = os.path.join(
                self.MODEL_DIR, self.graph_name,
                'LSTM_model_{0}.ckpt-{1}.meta'.format(self.name, max_epoch-1))
            
            checkpoint_path = os.path.join(self.MODEL_DIR, self.graph_name)
    
            saver = tf.train.import_meta_graph(graph_meta_path)
            saver.restore(sess, tf.train.latest_checkpoint(checkpoint_path))
    
            graph = tf.get_default_graph()
    
            test_feed_dict = {graph.get_tensor_by_name('inputs:0'): test_X}
    
            prediction = graph.get_tensor_by_name('output_layer/prediction:0')
            test_prediction = sess.run([prediction], test_feed_dict)    
        
        return test_prediction
    


    
    
    def to_dict(self):
            dct = self.__class__.__dict__
            return {k: v for k, v in dct.iteritems() if not k.startswith('__') and not callable(v)}
    
    def __str__(self):
        return str(self.to_dict())
    
    def __repr__(self):
        return str(self.to_dict())
    
        
    

