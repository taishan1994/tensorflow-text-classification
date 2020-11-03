import sys
import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) #当前程序上上一级目录，这里为mycompany
sys.path.append(BASE_DIR) #添加环境变量
from config.globalConfig import PATH


class TrainConfig(object): #训练时的参数
  epoches = 100 #训练的epoches
  evaluateEvery = 100 #可选，每隔多少步进行验证
  save_per_step = 100 #可选，每多少轮写入到tensorboard中scalar
  print_per_step = 20 #每多少轮输出训练集上的性能
  checkpointEvery = 10 #可选，每隔多少步进行存储模型
  learningRate = 0.001 #学习率

class TansformerConfig(object): #模型参数
  embeddingSize = 200 #词嵌入的维度
  filters = 64  #内层一维卷积核的数量，外层卷积核的数量应该等于embeddingSize，因为要确保每个layer后的输出维度和输入维度是一致的。
  numHeads = 8  #多头注意力的头数，需要注意，词嵌入的维度必须被头数整除
  numBlocks = 1  #设置transformer block的数量
  epsilon = 1e-8  #LayerNorm层中的最小除数
  keepProb = 0.9  #多头注意力中的Dropout
  
  dropoutKeepProb = 0.5 #全连接层的dropout
  l2RegLambda = 0.0 #L2正则化的系数

class BiLstmConfig(object):
  embeddingSize = 200 #词嵌入的维度
  hidden_dim = [128]  # 隐藏层神经元
  keepProb = 0.9  #Dropout

class TextCNNConfig(object):
  """CNN配置参数"""
  embeddingSize = 200  # 词向量维度
  num_filters = 128  # 卷积核数目
  filter_sizes = [2, 3, 4] 
  keepProb = 0.9  # dropout保留比例
  l2_reg_lambda = 0.0

class BiLstmAttnConfig:
  embeddingSize = 200 #词嵌入的维度
  hidden_dim = [128]  # 隐藏层神经元
  keepProb = 0.9  #Dropout

class RcnnConfig:
  embeddingSize = 200 #词嵌入的维度
  hidden_dim = [256]  # 隐藏层神经元
  output_size = 128 #输出维度
  keepProb = 0.9  #Dropout 

class FudanConfig(object):
  sequenceLength = 600  #句子的最大长度，若大于该长度则进行截断，小于该长度则进行padding；
  batchSize = 128 #batchszie大小
  trainPath = "data/Fudan/train/" #训练数据路径，文件夹下保存的是txt
  testPath = "data/Fudan/answer/" #测试数据路径，文件夹下保存的是txt
  stopWordSource = "process/Fudan/stopwords.txt" #停止词路径
  wor2vec_path = 'process/Fudan/word2vec/model/Word2vec.w2v' #词向量保存位置
  numClasses = 20  #二分类设置为1，多分类设置为类别的数目
  rate = 0.8  #可选，划分训练集、验证集或者测试集的比例
  vocab_size = 6000 #词汇总数+2，PAD、UNK

  trainConfig = TrainConfig()  
  modelConfig = TansformerConfig()
  biLstmConfig = BiLstmConfig()
  textCNNConfig = TextCNNConfig()
  biLstmAttnConfig = BiLstmAttnConfig()
  rcnnConfig = RcnnConfig()
