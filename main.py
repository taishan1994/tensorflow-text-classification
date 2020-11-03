import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) #当前程序上一级目录，这里为transformer
from dataset.fudanDataset import FudanDataset
from models.transformer import Transformer
from models.bilstm import BiLstmModel
from models.textcnn import TextCnnModel
from models.bilstmattn import BiLstmAttnModel
from models.rcnn import RcnnModel
from utils.utils import *
from utils.metrics import *
from config.fudanConfig import FudanConfig
from config.globalConfig import PATH
import numpy as numpy
import argparse
import tensorflow as tf
import time
import datetime
from tkinter import _flatten
from sklearn import metrics
import jieba

def train():
  #save_dir = 'checkpoint/bilstm/'
  #if not os.path.exists(save_dir):
  #  os.makedirs(save_dir)
  #save_path = os.path.join(save_dir, 'best_validation')  # 最佳验证结果保存路径

  # 定义保存输出的列表
  history_train_loss = []
  history_train_acc = []
  history_train_prec = []
  history_train_recall = []
  history_train_f_beta = []
  history_val_loss = []
  history_val_acc = []
  history_val_prec = []
  history_val_recall = []
  history_val_f_beta = []

  globalStep = tf.Variable(0, name="globalStep", trainable=False)
  # 配置 Saver
  saver = tf.train.Saver()

  #定义session
  """
  session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
  session_conf.gpu_options.allow_growth=True
  session_conf.gpu_options.per_process_gpu_memory_fraction = 0.9  # 配置gpu占用率  
  sess = tf.Session(config=session_conf)
  """
  sess = tf.Session()
  print("定义优化器。。。\n")
  # 定义优化函数，传入学习速率参数
  optimizer = tf.train.AdamOptimizer(config.trainConfig.learningRate)
  # 计算梯度,得到梯度和变量
  gradsAndVars = optimizer.compute_gradients(model.loss)
  # 将梯度应用到变量下，生成训练器
  trainOp = optimizer.apply_gradients(gradsAndVars, global_step=globalStep)
  sess.run(tf.global_variables_initializer())

  def trainStep(batchX, batchY):
    """
    训练函数
    """   
    feed_dict = {
      model.inputX: batchX,
      model.inputY: batchY,
      model.dropoutKeepProb: config.modelConfig.dropoutKeepProb,
    }
    _, step, loss, predictions = sess.run([trainOp, globalStep, model.loss, model.predictions], feed_dict)

    if config.numClasses == 1:
        acc, recall, prec, f_beta = get_binary_metrics(pred_y=predictions, true_y=batchY)
    elif config.numClasses > 1:
        acc, recall, prec, f_beta = get_multi_metrics(pred_y=predictions, true_y=batchY, labels=labelList)
        
    return loss, acc, prec, recall, f_beta

  def valStep(batchX, batchY):
    """
    验证函数
    """
    feed_dict = {
      model.inputX: batchX,
      model.inputY: batchY,
      model.dropoutKeepProb: 1.0,
    }
    step, loss, predictions = sess.run([globalStep, model.loss, model.predictions], feed_dict)
    
    if config.numClasses == 1:
        acc, recall, prec, f_beta = get_binary_metrics(pred_y=predictions, true_y=batchY)  
    elif config.numClasses > 1:
        acc, recall, prec, f_beta = get_multi_metrics(pred_y=predictions, true_y=batchY, labels=labelList) 

    return loss, acc, prec, recall, f_beta
  print("开始训练。。。\n")
  best_f_beta_val = 0.0  # 最佳验证集准确率
  last_improved = 0  # 记录上一次提升批次
  require_improvement = 1000  # 如果超过1000轮未提升，提前结束训练
  flag = False
  for epoch in range(config.trainConfig.epoches):
    print('Epoch:', epoch + 1)
    batch_train = batch_iter(train_data, train_label, config.batchSize)
    for x_batch, y_batch in batch_train:
      loss, acc, prec, recall, f_beta = trainStep(x_batch, y_batch)
      history_train_loss.append(loss)
      history_train_acc.append(acc)
      history_train_prec.append(prec)
      history_train_recall.append(recall)
      history_train_f_beta.append(f_beta)
      currentStep = tf.train.global_step(sess, globalStep) 
      # 多少次迭代打印一次训练结果：
      if currentStep % config.trainConfig.print_per_step == 0:
        print("train: step: {}, loss: {:.6f}, acc: {:.4f}, recall: {:.4f}, precision: {:.4f}, f_beta: {:.4f}".format(
                      currentStep, loss, acc, recall, prec, f_beta))

      if currentStep % config.trainConfig.evaluateEvery == 0:
        print("开始验证。。。\n")
        
        losses = []
        accs = []
        f_betas = []
        precisions = []
        recalls = []
        batch_val = batch_iter(val_data, val_label, config.batchSize)
        for x_batch, y_batch in batch_val:
          loss, acc, precision, recall, f_beta = valStep(x_batch, y_batch)
          losses.append(loss)
          accs.append(acc)
          f_betas.append(f_beta)
          precisions.append(precision)
          recalls.append(recall)
            
        if mean(f_betas) > best_f_beta_val:
          # 保存最好结果
          best_f_beta_val = mean(f_betas)
          last_improved = currentStep
          saver.save(sess=sess, save_path=train_save_path)
          improved_str = '*'
        else:
          improved_str = ''
        time_str = datetime.datetime.now().isoformat()
        print("{}, step: {}, loss: {:.6f}, acc: {:.4f},precision: {:.4f}, recall: {:.4f}, f_beta: {:.4f} {}".format(
          time_str, currentStep, mean(losses), mean(accs), mean(precisions), mean(recalls), mean(f_betas), improved_str))
        history_val_loss.append(mean(losses))
        history_val_acc.append(mean(accs))
        history_val_prec.append(mean(precisions))
        history_val_recall.append(mean(recalls))
        history_val_f_beta.append(mean(f_betas))
      if currentStep - last_improved > require_improvement:
        # 验证集正确率长期不提升，提前结束训练
        print("没有优化很长一段时间了，自动停止")
        flag = True
        break  # 跳出循环
    if flag:  # 同上
        break
  sess.close()
  history_dict ={
    "train_loss":history_train_loss,
    "train_acc":history_train_acc,
    "train_prec":history_train_prec,
    "train_recall":history_train_recall,
    "train_f_beta":history_train_f_beta,
    "val_loss":history_val_loss,
    "val_acc":history_val_acc,
    "val_prec":history_val_prec,
    "val_recall":history_val_recall,
    "val_f_beta":history_val_f_beta,
  }
  return history_dict

def test(test_data,test_label):
  print("开始进行测试。。。")
  #save_path = os.path.join(PATH,'checkpoint/bilstm')  
  saver = tf.train.import_meta_graph(os.path.join(test_save_path,"best_validation.meta"))
  with tf.Session() as sess:
    #sess.run(tf.global_variables_initializer())
    saver.restore(sess, os.path.join(predict_save_path,"best_validation"))  # 读取保存的模型

    data_len = len(test_data)
    test_batchsize = 128
    batch_test = batch_iter(test_data, test_label, 128, is_train=False)
    pred_label = []
    for x_batch,y_batch in batch_test:
      feed_dict = {
        model.inputX: x_batch,
        model.inputY: y_batch,
        model.dropoutKeepProb: 1.0,
      }
      predictions = sess.run([model.predictions], feed_dict)
      pred_label.append(predictions[0].tolist())
    pred_label = list(_flatten(pred_label))
    test_label = [np.argmax(item) for item in test_label]
    # 评估
    print("计算Precision, Recall and F1-Score...")
    print(metrics.classification_report(test_label, pred_label, target_names=true_labelList))


def process_sentence(data):
  fudanDataset._get_stopwords()
  sentence_list = []
  for content in data:
    words_list = jieba.lcut(content, cut_all=False)
    tmp1 = []
    for word in words_list:
      word = word.strip()
      if word not in fudanDataset.stopWordDict and word != '':
        tmp1.append(word)
      else:
        continue
    sentence_list.append(tmp1)
  vocab = fudanDataset._get_vocaburay()
  word2idx,idx2word = fudanDataset._wordToIdx()
  label2idx,idx2label = fudanDataset._labelToIdx()
  res_data = []
  #print(content)
  for content in sentence_list:
    tmp2 = []
    if len(content) >= config.sequenceLength: #大于最大长度进行截断
      content = content[:config.sequenceLength]
    else: #小于最大长度用PAD的id进行填充层
      content = ['PAD']*(config.sequenceLength-len(content)) + content
    for word in content: #将词语用id进行映射
      if word in word2idx:
        tmp2.append(word2idx[word])
      else:
        tmp2.append(word2idx['UNK'])
    res_data.append(tmp2)
  return res_data

def get_predict_content(content_path,label_path):
  use_data = 5
  txt_list = []
  label_list = []
  predict_data = []
  predict_label = []
  content_file = open(content_path,"r",encoding="utf-8")
  label_file = open(label_path,"r",encoding="utf-8")
  for txt in content_file.readlines(): #读取每一行的txt  
    txt = txt.strip() #去除掉\n
    txt_list.append(txt)
  for label in label_file.readlines():
    label = label.strip()
    label_list.append(label)
  data = []
  for txt,label in zip(txt_list,label_list):
    data.append((txt,label))
  import random
  predict_data = random.sample(data,use_data)
  p_data = []
  p_label = []
  for txt,label in predict_data:
    with open(txt,"r",encoding="gb18030",errors='ignore') as fp1:
      tmp = []
      for line in fp1.readlines(): #读取每一行
        tmp.append(line.strip())
      p_data.append("".join(tmp))
    p_label.append(label)
  content_file.close()
  label_file.close()
  return p_data,p_label


def predict(data,label,p_data):
  print("开始预测文本的类别。。。")
  predict_data = data
  predict_true_data = label
  #save_path = os.path.join(PATH,'checkpoint/bilstm')  
  saver = tf.train.import_meta_graph(os.path.join(predict_save_path,"best_validation.meta"))
  with tf.Session() as sess:
    #sess.run(tf.global_variables_initializer())
    saver.restore(sess, os.path.join(predict_save_path,"best_validation"))  # 读取保存的模型

    feed_dict = {
      model.inputX: predict_data,
      model.inputY: predict_true_data,
      model.dropoutKeepProb: 1.0,
    }
    predictions = sess.run([model.predictions], feed_dict)
    pred_label = predictions[0].tolist()
    real_label = [np.argmax(item) for item in predict_true_data]
    for content,pre_label,true_label in zip(p_data,pred_label,real_label):
      print("输入的文本是：{}...".format(content[:100]))
      print("预测的类别是：",idx2label[pre_label])
      print("真实的类别是：",idx2label[true_label])
      print("================================================")

if __name__ == '__main__':
  print("解析参数。。。。")
  parser = argparse.ArgumentParser('传入参数：main.py')
  parser.add_argument('-model','--model', default='transformer')
  parser.add_argument('-lr','--lr', default='0.001')
  parser.add_argument('-batchsize','--batchsize', default='128')
  parser.add_argument('-saver_dir','--saver_dir', default='checkpoint/transformer')
  parser.add_argument('-save_png','--save_png', default='images/transformer')
  parser.add_argument('-train','--train',default=False, action='store_true')
  parser.add_argument('-test','--test',default=False, action='store_true')
  parser.add_argument('-predict','--predict',default=False, action='store_true')
  args = parser.parse_args()
  if args.model not in ['transformer','bilstm','bilstmattn','textcnn','rcnn']:
    raise "请确认模型的名称，当前可使用：'transformer','bilstm','bilstmattn','textcnn','rcnn'"
    sys.exit(0)
  lr = float(args.lr)
  batchsize = int(args.batchsize)

  print(args.train,args.test,args.predict)
  print(type(args.train),type(args.test),type(args.predict))

  print("模型保存的位置。。。")
  saver_dir = args.saver_dir
  print(saver_dir)
  if not os.path.exists(saver_dir):
    os.makedirs(saver_dir)
  if args.train:
    train_save_path = os.path.join(saver_dir, 'best_validation')
  if args.test:
    test_save_path = os.path.join(PATH,args.saver_dir)
  if args.predict:
    predict_save_path = os.path.join(PATH,args.saver_dir)

  print("结果可视化保存位置。。。")
  save_png = args.save_png
  if not os.path.exists(save_png):
    os.makedirs(save_png)
  print(save_png)

  config = FudanConfig()
  config.batchSize = batchsize
  config.trainConfig.learningRate = lr

  fudanDataset = FudanDataset(config)
  word2idx,idx2word = fudanDataset._wordToIdx()
  label2idx,idx2label = fudanDataset._labelToIdx()
  print("加载数据。。。")
  train_content_path = os.path.join(PATH, "process/Fudan/word2vec/data/train_content.txt")
  train_label_path = os.path.join(PATH, "process/Fudan/train_label.txt")
  test_content_path = os.path.join(PATH, "process/Fudan/word2vec/data/test_content.txt")
  test_label_path = os.path.join(PATH, "process/Fudan/test_label.txt")
  fudanDataset._getTrainValData(train_content_path,train_label_path)
  fudanDataset._getTestData(test_content_path,test_label_path)
  fudanDataset._getWordEmbedding()
  train_data,val_data,train_label,val_label = fudanDataset.trainData,fudanDataset.valData,fudanDataset.trainLabels,fudanDataset.valLabels
  test_data,test_label = fudanDataset.testData,fudanDataset.testLabels
  train_label = one_hot(train_label)
  val_label = one_hot(val_label)
  test_label = one_hot(test_label)

  wordEmbedding = fudanDataset.wordEmbedding
  labelList = fudanDataset.labelList
  true_labelList = [idx2label[label] for label in labelList]
  
  wordEmbedding = np.array(wordEmbedding)
  print(wordEmbedding.shape)
  print("定义模型。。。")
  if args.model == "transformer":
    model = Transformer(config, wordEmbedding)
  if args.model == "bilstm":
    model = BiLstmModel(config,wordEmbedding)
  if args.model == "textcnn":
    model = TextCnnModel(config,wordEmbedding)
  if args.model == "bilstmattn":
    model = BiLstmAttnModel(config,wordEmbedding)
  if args.model == "rcnn":
    model = RcnnModel(config,wordEmbedding)
  print("使用模型：", args.model)

  if args.train:
    print(args.train)
    print(type(args.train))
    #训练
    history_dict = train()
    draw(history_dict,save_png,args.model)

  if args.test:
    #测试
    test(test_data,test_label)

  if args.predict:
    print("进行预测。。。")
    p_data,p_label = get_predict_content(os.path.join(PATH, "process/Fudan/test.txt"),test_label_path)

    process_data = process_sentence(p_data)
    onehot_label = np.zeros((len(p_label),config.numClasses))
    for i,value in enumerate(p_label):
      onehot_label[i][label2idx[value]] = 1
    process_label = onehot_label

    predict(process_data,process_label,p_data)










