import tensorflow as tf


class BiLstmModel:
    def __init__(self, config, wordEmbedding):
        super(BiLstmModel, self).__init__()
        self.inputX = tf.placeholder(tf.int32, [None, config.sequenceLength], name="inputX")
        self.inputY = tf.placeholder(tf.int32, [None, config.numClasses], name="inputY")
        self.dropoutKeepProb = tf.placeholder(tf.float32, name="dropoutKeepProb")
        self.config = config
        self.wordEmbedding = wordEmbedding
        # 定义l2损失
        self.l2_loss = tf.constant(0.0)  # 定义l2损失
        # 构建模型
        self.build_model()

    def get_predictions(self):
      """
      得到预测结果
      :return:
      """
      predictions = None
      if self.config.numClasses == 1:
          predictions = tf.cast(tf.greater_equal(self.logits, 0.0), tf.int32, name="predictions")
      elif self.config.numClasses > 1:
          predictions = tf.argmax(self.logits, axis=-1, name="predictions")
      return predictions

    def cal_loss(self):
      """
      计算损失，支持二分类和多分类
      :return:
      """
      with tf.name_scope("loss"):
          losses = 0.0
          if self.config.numClasses == 1:
              losses = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logits,labels=tf.reshape(self.inputY, [-1, 1]))
          elif self.config.numClasses > 1:
              self.inputY = tf.cast(self.inputY, dtype=tf.int32)
              losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits,labels=self.inputY)
          loss = tf.reduce_mean(losses)
          return loss

    def build_model(self):

        # 词嵌入层
        with tf.name_scope("embedding"):
            # 利用预训练的词向量初始化词嵌入矩阵
            if self.wordEmbedding is not None:
                embedding_w = tf.Variable(tf.cast(self.wordEmbedding, dtype=tf.float32, name="word2vec"),
                                          name="embedding_w")
            else:
                embedding_w = tf.get_variable("embedding_w", shape=[self.config.vocab_size+2, self.config.biLstmConfig.embedding_size],
                                              initializer=tf.contrib.layers.xavier_initializer())
            # 利用词嵌入矩阵将输入的数据中的词转换成词向量，维度[batch_size, sequence_length, embedding_size]
            embedded_words = tf.nn.embedding_lookup(embedding_w, self.inputX)

        # 定义两层双向LSTM的模型结构
        with tf.name_scope("Bi-LSTM"):
            for idx, hidden_size in enumerate(self.config.biLstmConfig.hidden_dim):
                with tf.name_scope("Bi-LSTM" + str(idx)):
                    # 定义前向LSTM结构
                    lstm_fw_cell = tf.nn.rnn_cell.DropoutWrapper(
                        tf.nn.rnn_cell.LSTMCell(num_units=hidden_size, state_is_tuple=True),
                        output_keep_prob=self.dropoutKeepProb)
                    # 定义反向LSTM结构
                    lstm_bw_cell = tf.nn.rnn_cell.DropoutWrapper(
                        tf.nn.rnn_cell.LSTMCell(num_units=hidden_size, state_is_tuple=True),
                        output_keep_prob=self.dropoutKeepProb)

                    # 采用动态rnn，可以动态的输入序列的长度，若没有输入，则取序列的全长
                    # outputs是一个元祖(output_fw, output_bw)，其中两个元素的维度都是[batch_size, max_time, hidden_size],
                    # fw和bw的hidden_size一样
                    # self.current_state 是最终的状态，二元组(state_fw, state_bw)，state_fw=[batch_size, s]，s是一个元祖(h, c)
                    outputs, current_state = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell,
                                                                             embedded_words, dtype=tf.float32,
                                                                             scope="bi-lstm" + str(idx))

                    # 对outputs中的fw和bw的结果拼接 [batch_size, time_step, hidden_size * 2]
                    embedded_words = tf.concat(outputs, 2)

        # 取出最后时间步的输出作为全连接的输入
        final_output = embedded_words[:, -1, :]

        output_size = self.config.biLstmConfig.hidden_dim[-1] * 2  # 因为是双向LSTM，最终的输出值是fw和bw的拼接，因此要乘以2
        output = tf.reshape(final_output, [-1, output_size])  # reshape成全连接层的输入维度

        # 全连接层的输出
        with tf.name_scope("output"):
            output_w = tf.get_variable(
                "output_w",
                shape=[output_size, self.config.numClasses],
                initializer=tf.contrib.layers.xavier_initializer())

            output_b = tf.Variable(tf.constant(0.1, shape=[self.config.numClasses]), name="output_b")
            self.l2_loss += tf.nn.l2_loss(output_w)
            self.l2_loss += tf.nn.l2_loss(output_b)
            self.logits = tf.nn.xw_plus_b(output, output_w, output_b, name="logits")
            self.predictions = self.get_predictions()

        self.loss = self.cal_loss()