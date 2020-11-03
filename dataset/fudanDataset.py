import sys
import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) #当前程序上上一级目录，这里为transformer
sys.path.append(BASE_DIR) #添加环境变量
import pickle
import os
import glob
from config.fudanConfig import FudanConfig
from config.globalConfig import PATH
import jieba
from collections import Counter, OrderedDict
import copy
from sklearn.model_selection import train_test_split
from gensim.models import Word2Vec
import numpy as np


class FudanDataset(object):
  def __init__(self,config):
    self.config = config #一些配置
    self.trainPath = config.trainPath #训练集数据路径
    self.testPath = config.testPath #测试集数据路径
    self.stopWordSource = config.stopWordSource #停止词数据源
    self.embeddingSize = config.modelConfig.embeddingSize #嵌入词向量的维度
    self.batchSize = config.batchSize #批处理大小
    self.rate = config.rate #可选，划分训练集验证集或测试集可用

    self.stopWordDict = {} #停止词字典
    self.trainData = [] #训练集
    self.trainLabels = [] #训练集标签
    
    self.evalData = [] #验证集
    self.evalLabels = [] #验证集标签
    
    self.testData = [] #测试集
    self.testLabels = []#测试集标签
 
    self.wordEmbedding = [] #预训练的词向量
    
    self.labelList = [] #需要分类的标签
  
  def _txtpath_to_txt(self): #将训练集和测试集下的txt路径保存
    train_txt_path = os.path.join(PATH, "process/Fudan/train.txt") 
    test_txt_path = os.path.join(PATH, "process/Fudan//test.txt")
    train_list = os.listdir(os.path.join(PATH, self.trainPath)) #获得该目录下的所有文件夹，返回一个列表
    fp1 = open(train_txt_path,"w",encoding="utf-8")
    fp2 = open(test_txt_path,"w",encoding="utf-8")
    for train_dir in train_list: #取得下一级目录下的所有的txt路径（绝对路径）
      for txt in glob.glob(os.path.join(PATH,self.trainPath+train_dir+"/*.txt")):
        fp1.write(txt+"\n")
    fp1.close()
    test_list = os.listdir(os.path.join(PATH,self.testPath)) #获得该目录下的所有文件夹，返回一个列表
    for test_dir in test_list:
      for txt in glob.glob(os.path.join(PATH, self.testPath+test_dir+"/*.txt")):
        fp2.write(txt+"\n")
    fp2.close()
  
  #将txt中的文本和标签存储到txt中
  def _contentlabel_to_txt(self, txt_path, content_path, label_path):
    files = open(txt_path,"r",encoding="utf-8")
    content_file = open(content_path,"w",encoding="utf-8")
    label_file = open(label_path,"w",encoding="utf-8")
    for txt in files.readlines(): #读取每一行的txt  
      txt = txt.strip() #去除掉\n
      content_list=[] 
      label_str = txt.split("/")[-1].split("-")[-1] #先用/进行切割，获取列表中的最后一个，再利用-进行切割，获取最后一个
      label_list = []
      #以下for循环用于获取标签，遍历每个字符，如果遇到了数字，就终止
      for s in label_str:
        if s.isalpha():
          label_list.append(s)
        elif s.isalnum():
          break
        else:
          print("出错了")
      label = "".join(label_list) #将字符列表转换为字符串，得到标签
      #print(label)
      #以下用于获取所有文本
      fp1 = open(txt,"r",encoding="gb18030",errors='ignore') #以gb18030的格式打开文件，errors='ignore'用于忽略掉超过该字符编码范围的字符
      for line in fp1.readlines(): #读取每一行
        #jieba分词，精确模式
        line = jieba.lcut(line.strip(), cut_all=False)
        #将每一行分词的结果保存在一个list中
        content_list.extend(line)
      fp1.close()

      content_str = " ".join(content_list) #转成字符串
      #print(content_str)
      content_file.write(content_str+"\n") #将文本保存到tx中
      label_file.write(label+"\n")
    content_file.close()
    label_file.close()  
    files.close()

  #去除掉停用词
  def _get_clean_data(self, filePath):
    #先初始化停用词字典
    self._get_stopwords()
    sentence_list = []
    with open(filePath,'r',encoding='utf-8') as fp:
      lines = fp.readlines()
      for line in lines:
        tmp = []
        words = line.strip().split(" ")
        for word in words:
          word = word.strip()
          if word not in self.stopWordDict and word != '':
            tmp.append(word)
          else:
            continue
        sentence_list.append(tmp)
    return sentence_list

  #读取停用词字典
  def _get_stopwords(self):
    with open(os.path.join(PATH, self.stopWordSource), "r") as f:
      stopWords = f.read()
      stopWordList = set(stopWords.splitlines())
      # 将停用词用列表的形式生成，之后查找停用词时会比较快
      self.stopWordDict = dict(zip(stopWordList, list(range(len(stopWordList)))))
  
  #创建词汇表
  def _get_vocaburay(self):
    train_content = os.path.join(PATH, "process/Fudan/word2vec/data/train_content.txt")
    sentence_list = self._get_clean_data(train_content)
    #这里可以计算文本的平均长度，设置配置中的sequenceLength
    #max_sequence = sum([len(s) for s in sentence_list]) / len(sentence_list)
    vocab_before = []
    for sentence in sentence_list:
      for word in sentence:
        vocab_before.append(word)
    count_vocab = Counter(vocab_before) #统计每个词出现的次数
    #print(len(count_vocab))
    count_vocab = sorted(count_vocab.items(),key=lambda x:x[1], reverse=True) #将出现频率按从高到低排序
    vocab_after = copy.deepcopy(count_vocab[:self.config.vocab_size])
    return dict(vocab_after) #返回前6000个词，将元组构成的列表转换为字典

  def _wordToIdx(self): #构建词汇和id的映射
    vocab = list(self._get_vocaburay().keys()) #取得字典中的键，也就是词语，转换成列表
    #print(vocab)
    tmp = ['PAD','UNK']
    vocab = tmp + vocab
    word2idx = {word:i for i,word in enumerate(vocab)}
    idx2word = {i:word for i,word in enumerate(vocab)}
    return word2idx,idx2word
  
  def _labelToIdx(self): #构建词汇列表和到id的映射
    label_path = os.path.join(PATH, "process/Fudan/train_label.txt")
    with open(os.path.join(PATH, label_path), "r") as f:
      labels = f.read()
      labelsList = sorted(set(labels.splitlines())) #为了避免每次标签id变换，这里排个序
      label2idx = {label:i for i,label in enumerate(labelsList)}
      idx2label = {i:label for i,label in enumerate(labelsList)}
      self.labelList = [label2idx[label] for label in labelsList]
    return label2idx,idx2label
  
  def _getData(self,contentPath,labelPath,mode=None):
    #这里有两种操作，如果文本中的词没有在词汇表中出现，则可以舍去或者用UNK代替，我们这里使用UNK
    vocab = self._get_vocaburay()
    word2idx,idx2word = self._wordToIdx()
    label2idx,idx2label = self._labelToIdx()
    data = []
    content_list = self._get_clean_data(contentPath)
    for content in content_list:
      #print(content)
      tmp = []
      if len(content) >= self.config.sequenceLength: #大于最大长度进行截断
        content = content[:self.config.sequenceLength]
      else: #小于最大长度用PAD的id进行填充层
        content = ['PAD']*(self.config.sequenceLength-len(content)) + content
      for word in content: #将词语用id进行映射
        if word in word2idx:
          tmp.append(word2idx[word])
        else:
          tmp.append(word2idx['UNK'])
      data.append(tmp)
    with open(labelPath,'r',encoding='utf-8') as fp:
      labels = fp.read()
      label = [[label2idx[label]] for label in labels.splitlines()]
    return data,label

  def _getTrainValData(self,dataPath,labelPath):
    trainData,trainLabel = self._getData(dataPath,labelPath)
    #方便起见，我们这里就直接使用sklearn中的函数了
    self.trainData,self.valData,self.trainLabels,self.valLabels = train_test_split(trainData,trainLabel,test_size=self.rate,random_state=1)

  def _getTestData(self,dataPath,labelPath):
    self.testData,self.testLabels = self._getData(dataPath,labelPath)

  #获取词汇表中的词向量
  def _getWordEmbedding(self): 
    word2idx,idx2word = self._wordToIdx()
    vocab = sorted(word2idx.items(), key=lambda x:x[1]) #将词按照id进行排序
    #print(vocab)
    w2vModel = Word2Vec.load(os.path.join(PATH,self.config.wor2vec_path))
    self.wordEmbedding.append(np.array([0]*self.embeddingSize,dtype=np.float32)) #PAD对应的词向量
    self.wordEmbedding.append(np.array([0]*self.embeddingSize,dtype=np.float32)) #UNK对应的词向量
    for i in range(2,len(vocab)):
       self.wordEmbedding.append(w2vModel[vocab[i][0]])

#fudanConfig = FudanConfig()
#fudanDataset = FudanDataset(fudanConfig)
#1、将所有的txt路径写入的相关的文件中
#fudanDataset._txtpath_to_txt()
#2、将文本内容、标签写入到txt中
"""
train_txt_path = os.path.join(PATH, "process/Fudan/train.txt") 
test_txt_path = os.path.join(PATH, "process/Fudan/test.txt")
train_content_path = os.path.join(PATH, "process/Fudan/word2vec/data/train_content.txt")
train_label_path = os.path.join(PATH, "process/Fudan/train_label.txt")
test_content_path = os.path.join(PATH, "process/Fudan/word2vec/data/test_content.txt")
test_label_path = os.path.join(PATH, "process/Fudan/test_label.txt")
fudanDataset._contentlabel_to_txt(train_txt_path,train_content_path,train_label_path)
fudanDataset._contentlabel_to_txt(test_txt_path,test_content_path,test_label_path) 
"""
#3、训练词向量，运行utils下的word2vec.py
#4、过滤掉停用词，构建总总词汇表
#vocab = fudanDataset._get_vocaburay()
#5、构建词语到id的映射
#word2idx,idx2word = fudanDataset._wordToIdx()
#6、构建标签到id的映射
#label2idx,idx2label = fudanDataset._labelToIdx()
#7、构建训练集、训练标签、验证集、验证标签、测试集、测试标签
"""
train_content_path = os.path.join(PATH, "process/Fudan/word2vec/data/train_content.txt")
train_label_path = os.path.join(PATH, "process/Fudan/train_label.txt")
fudanDataset._getTrainValData(train_content_path,train_label_path)
trainData,valData,trainLabel,valLabel = fudanDataset.trainData,fudanDataset.valData,fudanDataset.trainLabels,fudanDataset.valLabels
test_content_path = os.path.join(PATH, "process/Fudan/word2vec/data/test_content.txt")
test_label_path = os.path.join(PATH, "process/Fudan/test_label.txt")
fudanDataset._getData(test_content_path,test_label_path)
testData,testLabel = fudanDataset.testData,fudanDataset.testLabels
"""
#8、获取词嵌入
#fudanDataset._getWordEmbedding()
#wordEmbedding = fudanDataset.wordEmbedding
