import tensorflow as tf

class TextCnnModel:
    def __init__(self, config, wordEmbedding):
        super(TextCnnModel, self).__init__()
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
            # 卷积的输入是思维[batch_size, width, height, channel]，因此需要增加维度，用tf.expand_dims来增大维度
            #(?, 600, 200, 1)
            embedded_words_expand = tf.expand_dims(embedded_words, -1)

        # 创建卷积和池化层
        pooled_outputs = []
        # 有三种size的filter，3， 4， 5，textCNN是个多通道单层卷积的模型，可以看作三个单层的卷积模型的融合
        for i, filter_size in enumerate(self.config.textCNNConfig.filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # 卷积层，卷积核尺寸为filterSize * embeddingSize，卷积核的个数为numFilters
                # 初始化权重矩阵和偏置
                # [2,200,1,128] [3,200,1,128] [4,200,1,128] [5,200,1,128]
                filter_shape = [filter_size, self.config.textCNNConfig.embeddingSize, 1, self.config.textCNNConfig.num_filters]
                print(filter_shape)
                #tf.truncated_normal产生截断正态分布随机数，取值范围为 [ mean - 2 * stddev, mean + 2 * stddev ]。
                conv_w = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="conv_w")
                conv_b = tf.Variable(tf.constant(0.1, shape=[self.config.textCNNConfig.num_filters], name="conv_b"))
                """
                tf.nn.conv2d (input, filter, strides, padding, use_cudnn_on_gpu=None, data_format=None, name=None)
                input : 输入的要做卷积的图片，要求为一个张量，shape为 [ batch, in_height, in_weight, in_channel ]，其中batch为图片的数量，in_height 为图片高度，in_weight 为图片宽度，in_channel 为图片的通道数，灰度图该值为1，彩色图为3。（也可以用其它值，但是具体含义不是很理解）
                filter： 卷积核，要求也是一个张量，shape为 [ filter_height, filter_weight, in_channel, out_channels ]，其中 filter_height 为卷积核高度，filter_weight 为卷积核宽度，in_channel 是图像通道数 ，和 input 的 in_channel 要保持一致，out_channel 是卷积核数量。
                strides： 卷积时在图像每一维的步长，这是一个一维的向量，[ 1, strides, strides, 1]，第一位和最后一位固定必须是1
                padding： string类型，值为“SAME” 和 “VALID”，表示的是卷积的形式，是否考虑边界。"SAME"是考虑边界，不足的时候用0去填充周围，"VALID"则不考虑
                use_cudnn_on_gpu： bool类型，是否使用cudnn加速，默认为true
                """
                conv = tf.nn.conv2d(
                    embedded_words_expand,
                    conv_w,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                print("conv.shape:",conv.shape)
                # relu函数的非线性映射
                h = tf.nn.relu(tf.nn.bias_add(conv, conv_b), name="relu")
                print("h.shape:",h.shape)
                # 池化层，最大池化，池化是对卷积后的序列取一个最大值
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, self.config.sequenceLength - filter_size + 1, 1, 1],
                    # ksize shape: [batch, height, width, channels]
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                print("pooled.shape:",pooled.shape)
                pooled_outputs.append(pooled)  # 将三种size的filter的输出一起加入到列表中

        # 得到CNN网络的输出长度 128+128+128=384
        num_filters_total = self.config.textCNNConfig.num_filters * len(self.config.textCNNConfig.filter_sizes)
        print("num_filters_total:",num_filters_total)
        # 池化后的维度不变，按照最后的维度channel来concat Tensor("concat:0", shape=(?, 1, 101, 384), dtype=float32)
        h_pool = tf.concat(pooled_outputs, 3)
        print("h_pool.shape:",h_pool)
        # 摊平成二维的数据输入到全连接层
        h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])
        print("h_pool_flat:",h_pool_flat.shape)
        # dropout
        with tf.name_scope("dropout"):
            h_drop = tf.nn.dropout(h_pool_flat, self.dropoutKeepProb)

        # 全连接层的输出
        with tf.name_scope("output"):
            output_w = tf.get_variable(
                "output_w",
                shape=[num_filters_total, self.config.numClasses],
                initializer=tf.contrib.layers.xavier_initializer())
            output_b = tf.Variable(tf.constant(0.1, shape=[self.config.numClasses]), name="output_b")
            self.l2_loss += tf.nn.l2_loss(output_w)
            self.l2_loss += tf.nn.l2_loss(output_b)
            self.logits = tf.nn.xw_plus_b(h_drop, output_w, output_b, name="logits")
            self.predictions = self.get_predictions()

        # 计算交叉熵损失
        self.loss = self.cal_loss() + self.config.textCNNConfig.l2_reg_lambda * self.l2_loss
 