import tensorflow as tf


class Model():
    def __init__(self, h_nodes=200, layers_cnt=5, learning_rate=0.1):
        self.h_nodes = h_nodes
        self.layers_cnt = layers_cnt
        self.learning_rate = learning_rate


class Seq2Seq(Model):
    outputs = None

    
    def __init__(self, voca_cnt, h_nodes=128, layers_cnt=3, learning_rate=0.01, drop_per=0.5):
        super(__class__, self).__init__(h_nodes, layers_cnt, learning_rate)
        
        self.voca_cnt  = voca_cnt
        self.h_nodes = h_nodes
        self.layers_cnt = layers_cnt
        self.learning_rate = learning_rate
        self.drop_per = drop_per
        
        self.en_input = tf.placeholder(tf.float32, [None, None, self.voca_cnt])
        self.de_input = tf.placeholder(tf.float32, [None, None, self.voca_cnt])
        self.targets = tf.placeholder(tf.int64, [None, None])
        
        self.W = tf.Variable(tf.ones([self.h_nodes, self.voca_cnt]), name="weights")
        self.b = tf.Variable(tf.zeros([self.voca_cnt]), name="bias")
        self.g_step = tf.Variable(0, trainable=False, name="global_step")
        self.build_model()
        self.saver = tf.train.Saver(tf.global_variables())
          
    def build_model(self):
        """
         - build_model method: create lstm model and optimizing
          - parameter is None
        """
        with tf.variable_scope('encode'):
            en_cell = tf.nn.rnn_cell.MultiRNNCell([self._lstm() 
                                                   for _ in range(self.layers_cnt)])
            outputs, en_states = tf.nn.dynamic_rnn(en_cell, self.en_input, dtype=tf.float32)
            
        with tf.variable_scope('decode'):
            de_cell = tf.nn.rnn_cell.MultiRNNCell([self._lstm() 
                                                   for _ in range(self.layers_cnt)]) 
            outputs, de_states = tf.nn.dynamic_rnn(de_cell, self.de_input, dtype=tf.float32,
                                                  initial_state = en_states)
            
        logits, self.cost, self.optimizer = self._cost_optimizing(outputs, self.targets)        
        self.outputs = tf.argmax(logits, 2)
        
    def _lstm(self):
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(self.h_nodes)
        lstm_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_cell, output_keep_prob = self.drop_per)
        return lstm_cell
    
    def _cost_optimizing(self, outputs, targets):
        seq_len = tf.shape(outputs)[1]
        outputs = tf.reshape(outputs, [-1, self.h_nodes])
        
        hypothesis = tf.matmul(outputs, self.W) + self.b
        logits = tf.reshape(hypothesis, [-1, seq_len, self.voca_cnt])
        
        cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels = targets))
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(cost, global_step=self.g_step)
        
        tf.summary.scalar('cost', cost)
        
        return logits, cost, optimizer
    
    def train(self, s, en_input, de_input, targets):
        return s.run([self.optimizer, self.cost],
                    feed_dict={self.en_input: en_input, 
                               self.de_input: de_input, 
                               self.targets: targets})
    
    def test(self, s, en_input, de_input, targets):
        prediction = tf.equal(self.outputs, targets)
        accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))
        return s.run([self.targets, self.outputs, accuracy], 
                     feed_dict={self.en_input: en_input, 
                                self.de_input: de_input, 
                                self.targets: targets})
    
    def predict(self, s, en_input, de_input):
        return s.run(self.outputs,
                     feed_dict={self.en_input: en_input,
                                self.de_input: de_input})
    
        
        
