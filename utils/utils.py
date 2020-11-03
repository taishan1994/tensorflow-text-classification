from sklearn.preprocessing import OneHotEncoder
from config.globalConfig import PATH
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator #导入此类，设置坐标轴间隔
import numpy as np
import os

def batch_iter(x, y, batch_size=None, is_train=True):
  """生成批次数据"""
  x = np.array(x)
  y = np.array(y)
  data_len = len(x)
  num_batch = int((data_len - 1) / batch_size) + 1
  if is_train:
    indices = np.random.permutation(np.arange(data_len))
    x_shuffle = x[indices,:]
    y_shuffle = y[indices,:]
  else:
    x_shuffle = x
    y_shuffle = y

  for i in range(num_batch):
      start_id = i * batch_size
      end_id = min((i + 1) * batch_size, data_len)
      yield x_shuffle[start_id:end_id,:], y_shuffle[start_id:end_id,:]

def one_hot(y):
  """将标签转换为onehot编码"""
  y = np.array(y)
  enc = OneHotEncoder()
  res = enc.fit_transform(y).todense().tolist()
  return res

def get_time_dif(start_time):
  """获取已使用时间"""
  end_time = time.time()
  time_dif = end_time - start_time
  return timedelta(seconds=int(round(time_dif)))

#可视化结果
def draw(history_dict,save_png,model):
  train_loss = history_dict['train_loss'] 
  train_acc = history_dict['train_acc'] 
  train_prec = history_dict['train_prec'] 
  train_recall = history_dict['train_recall'] 
  train_f_beta = history_dict['train_f_beta'] 
  val_loss = history_dict['val_loss'] 
  val_acc = history_dict['val_acc'] 
  val_prec = history_dict['val_prec'] 
  val_recall = history_dict['val_recall'] 
  val_f_beta = history_dict['val_f_beta'] 
  train_step = range(len(train_loss))
  val_step = [i*100 for i in range(len(val_loss))]
  plt.figure(figsize=(10, 6))

  train_space = 100
  val_space = 100
  ax1 = plt.subplot(2, 5, 1) # 两行五列，位置是1的子图
  x_major_locator1=MultipleLocator(train_space)
  ax1.xaxis.set_major_locator(x_major_locator1)
  plt.xlabel('step')
  plt.ylabel('train_loss')
  plt.xticks([])
  plt.plot(train_step, train_loss, 'b--')
  plt.tight_layout() #使得不同子图之间区分开

  ax2 = plt.subplot(2, 5, 2) # 两行五列，位置是2的子图
  x_major_locator2=MultipleLocator(train_space)
  ax2.xaxis.set_major_locator(x_major_locator2)
  plt.xlabel('step')
  plt.ylabel('train_acc')
  plt.xticks([])
  plt.plot(train_step, train_acc, 'b--')
  plt.tight_layout()

  ax3 = plt.subplot(2, 5, 3) # 两行五列，位置是3的子图
  x_major_locator3=MultipleLocator(train_space)
  ax3.xaxis.set_major_locator(x_major_locator3)
  plt.xlabel('step')
  plt.ylabel('train_prec')
  plt.xticks([])
  plt.plot(train_step, train_prec, 'b--')
  plt.tight_layout()

  ax4 = plt.subplot(2, 5, 4) # 两行五列，位置是4的子图
  x_major_locator4=MultipleLocator(train_space)
  ax4.xaxis.set_major_locator(x_major_locator4)
  plt.xlabel('step')
  plt.ylabel('train_recall')
  plt.xticks([])
  plt.plot(train_step, train_recall, 'b--')

  ax5 = plt.subplot(2, 5, 5) # 两行五列，位置是5的子图
  x_major_locator5=MultipleLocator(train_space)
  ax5.xaxis.set_major_locator(x_major_locator5)
  plt.xlabel('step')
  plt.ylabel('train_f_beta')
  plt.xticks([])
  plt.plot(train_step, train_f_beta, 'b--')
  plt.tight_layout()

  ax6 = plt.subplot(2, 5, 6) # 两行五列，位置是6的子图
  x_major_locator6=MultipleLocator(val_space)
  ax6.xaxis.set_major_locator(x_major_locator6)
  plt.xlabel('step')
  plt.ylabel('val_loss')
  plt.xticks([])
  plt.plot(val_step, val_loss, 'b--')
  plt.tight_layout()

  ax7 = plt.subplot(2, 5, 7) # 两行五列，位置是7的子图
  x_major_locator7=MultipleLocator(val_space)
  ax7.xaxis.set_major_locator(x_major_locator7)
  plt.xlabel('step')
  plt.ylabel('val_acc')
  plt.xticks([])
  plt.plot(val_step, val_acc, 'b--')
  plt.tight_layout()

  ax8 = plt.subplot(2, 5, 8) # 两行五列，位置是8的子图
  x_major_locator8=MultipleLocator(val_space)
  ax8.xaxis.set_major_locator(x_major_locator8)
  plt.xlabel('step')
  plt.ylabel('val_prec')
  plt.xticks([])
  plt.plot(val_step, val_prec, 'b--')
  plt.tight_layout()

  ax9 = plt.subplot(2, 5, 9) # 两行五列，位置是9的子图
  x_major_locator9=MultipleLocator(val_space)
  ax9.xaxis.set_major_locator(x_major_locator9)
  plt.xlabel('step')
  plt.ylabel('val_recall')
  plt.xticks([])
  plt.plot(val_step, val_recall, 'b--')
  plt.tight_layout()

  ax10 = plt.subplot(2, 5, 10) # 两行五列，位置是10的子图
  x_major_locator10=MultipleLocator(val_space)
  ax10.xaxis.set_major_locator(x_major_locator10)
  plt.xlabel('step')
  plt.ylabel('val_f_beta')
  plt.xticks([])
  plt.plot(val_step, val_f_beta, 'b--')
  plt.tight_layout()
  save_png_path = os.path.join(PATH,save_png)
  png_name = model + ".png"
  plt.savefig(os.path.join(save_png_path,png_name))
  plt.show()
  print("绘图完成了。。。")




  