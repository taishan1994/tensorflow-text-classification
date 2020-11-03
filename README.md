# tensorflow-text-classification
## 基于tensorflow的中文文本分类<br>
数据集：复旦中文语料，包含20类<br>
数据集下载地址：https://www.kesci.com/mw/dataset/5d3a9c86cf76a600360edd04/content<br>
数据集下载好之后将其放置在data文件夹下；<br>
修改globalConfig.py中的全局路径为自己项目的路径；<br>
处理后的数据和已训练好保存的模型，在这里可以下载：<br>
链接：https://pan.baidu.com/s/1ZHzO5e__-WFYAYFIt2Kmsg 
提取码：vvzy <br>

目录结构：<br>
|--checkpint：保存模型目录<br>
|--|--transformer：transformer模型保存位置；<br>
|--config：配置文件；<br>
|--|--fudanConfig.py：包含训练配置、模型配置、数据集配置；<br>
|--|--globaConfig.py：全局配置文件，主要是全局路径、全局参数等；<br>
|--   data：数据保存位置；<br>
|--|--|--Fudan：复旦数据；<br>
|--|--|--train：训练数据；<br>
|--|--|--answer：测试数据；<br>
|--dataset：创建数据集，对数据进行处理的一些操作；<br>
|--images：结果可视化图片保存位置；<br>
|--models：模型保存文件；<br>
|--process：对原始数据进行处理后的数据；<br>
|--tensorboard：tensorboard可视化文件保存位置，暂时未用到；<br>
|--utils：辅助函数保存位置，包括word2vec训练词向量、评价指标计算、结果可视化等；<br>
|--main.py：主运行文件，选择模型、训练、测试和预测；<br>

初始配置：<br>
- 词嵌入维度：200  
- 学习率：0.001  
- epoch：50  
- 词汇表大小：6000+2（加2是PAD和UNK） 
- 文本最大长度：600
- 每多少个step进行验证：100
- 每多少个step进行存储模型：100

环境：<br>
- python=>=3.6
- tensorflow==1.15.0

当前支持的模型：<br>
- bilstm
- bilstm+attention
- textcnn
- rcnn
- transformer


## 说明
数据的输入格式：<br>
（1）分词后去除掉停止词，再对词语进行词频统计，取频数最高的前6000个词语作为词汇表；<br>
（2）像词汇表中加入PAD和UNK,实际上的词汇表的词语总数为6000+2=6002；<br>
（3）当句子长度大于指定的最大长度，进行裁剪，小于最大长度，在句子前面用PAD进行填充；<br>
（4）如果句子中的词语在词汇表中没有出现则用UNK进行代替；<br>
（5）输入到网络中的句子实际上是进行分词后的词语映射的id，比如：<br>
（6）输入的标签是要经过onehot编码的；<br>
<br>
"""<br>
    "我喜欢上海",<br>
    "我喜欢打羽毛球",<br>
"""<br>
词汇表：['我','喜欢','打','上海','羽毛球']，对应映射：[2,3,4,5,6]，0对应PAD，1对应UNK
<br>
得到：<br>
[<br>
    [0,2,3,5],<br>
    [2,3,4,6],<br>
]<br>

```python
python main.py --model transformer --saver_dir checkpoint/transformer --save_png images/transformer  --train  --test  --predict 
```

参数说明：
 -   --model：选择模型，可选[transformer、bilstm、bilstmattn、textcnn、rcnn]
 -   --saver_dir：模型保存位置，一般是checkpoint+模型名称
 -   --save_png：结果可视化保存位置，一般是images+模型名称
 -   --train：是否进行训练，默认为False
 -   --test：是否进行测试，默认为False
 -   --predict：是否进行预测，默认为False

## 结果
```python
以transformer为例：
部分训练结果：
2020-11-01T10:43:16.955322, step: 1300, loss: 5.089711, acc: 0.8546,precision: 0.3990, recall: 0.4061, f_beta: 0.3977 *
Epoch: 83
train: step: 1320, loss: 0.023474, acc: 0.9922, recall: 0.8444, precision: 0.8474, f_beta: 0.8457
Epoch: 84
train: step: 1340, loss: 0.000000, acc: 1.0000, recall: 0.7500, precision: 0.7500, f_beta: 0.7500
Epoch: 85
train: step: 1360, loss: 0.000000, acc: 1.0000, recall: 0.5500, precision: 0.5500, f_beta: 0.5500
Epoch: 86
Epoch: 87
train: step: 1380, loss: 0.000000, acc: 1.0000, recall: 0.7500, precision: 0.7500, f_beta: 0.7500
Epoch: 88
train: step: 1400, loss: 0.000000, acc: 1.0000, recall: 0.7000, precision: 0.7000, f_beta: 0.7000
开始验证。。。

2020-11-01T10:44:07.347359, step: 1400, loss: 5.111372, acc: 0.8506,precision: 0.4032, recall: 0.4083, f_beta: 0.3982 *
Epoch: 89
train: step: 1420, loss: 0.000000, acc: 1.0000, recall: 0.5500, precision: 0.5500, f_beta: 0.5500
Epoch: 90
train: step: 1440, loss: 0.000000, acc: 1.0000, recall: 0.5500, precision: 0.5500, f_beta: 0.5500
Epoch: 91
Epoch: 92
train: step: 1460, loss: 0.000000, acc: 1.0000, recall: 0.7000, precision: 0.7000, f_beta: 0.7000
Epoch: 93
train: step: 1480, loss: 0.000000, acc: 1.0000, recall: 0.7500, precision: 0.7500, f_beta: 0.7500
Epoch: 94
train: step: 1500, loss: 0.000000, acc: 1.0000, recall: 0.6000, precision: 0.6000, f_beta: 0.6000
开始验证。。。

2020-11-01T10:44:57.645305, step: 1500, loss: 5.206666, acc: 0.8521,precision: 0.4003, recall: 0.4040, f_beta: 0.3957 
Epoch: 95
train: step: 1520, loss: 0.000000, acc: 1.0000, recall: 0.6000, precision: 0.6000, f_beta: 0.6000
Epoch: 96
Epoch: 97
train: step: 1540, loss: 0.000000, acc: 1.0000, recall: 0.7500, precision: 0.7500, f_beta: 0.7500
Epoch: 98
train: step: 1560, loss: 0.000000, acc: 1.0000, recall: 0.7000, precision: 0.7000, f_beta: 0.7000
Epoch: 99
train: step: 1580, loss: 0.000000, acc: 1.0000, recall: 0.8000, precision: 0.8000, f_beta: 0.8000
Epoch: 100
train: step: 1600, loss: 0.000000, acc: 1.0000, recall: 0.5000, precision: 0.5000, f_beta: 0.5000
开始验证。。。

2020-11-01T10:45:47.867190, step: 1600, loss: 5.080955, acc: 0.8566,precision: 0.4087, recall: 0.4131, f_beta: 0.4036 *
<Figure size 1000x600 with 10 Axes>
绘图完成了。。。
开始进行测试。。。
计算Precision, Recall and F1-Score...
               precision    recall  f1-score   support

  Agriculture       0.89      0.90      0.89      1022
          Art       0.80      0.95      0.86       742
Communication       0.19      0.26      0.22        27
     Computer       0.95      0.94      0.94      1358
      Economy       0.86      0.91      0.89      1601
    Education       1.00      0.11      0.21        61
  Electronics       0.35      0.39      0.37        28
       Energy       1.00      0.03      0.06        33
  Enviornment       0.88      0.96      0.92      1218
      History       0.79      0.48      0.60       468
          Law       1.00      0.12      0.21        52
   Literature       0.00      0.00      0.00        34
      Medical       0.50      0.13      0.21        53
     Military       0.33      0.01      0.03        76
         Mine       1.00      0.03      0.06        34
   Philosophy       1.00      0.04      0.09        45
     Politics       0.73      0.91      0.81      1026
        Space       0.84      0.86      0.85       642
       Sports       0.93      0.91      0.92      1254
    Transport       0.33      0.03      0.06        59

     accuracy                           0.86      9833
    macro avg       0.72      0.45      0.46      9833
 weighted avg       0.85      0.86      0.84      9833
```
结果可视化图片如下：<br>
![image-1](https://github.com/taishan1994/tensorflow-text-classification/blob/master/images/transformer/transformer.png)

```python
进行预测。。。
开始预测文本的类别。。。
输入的文本是：自动化学报ACTA AUTOMATICA SINICA1997年　第23卷　第4期　Vol.23　No.4　1997一种在线建模方法的研究1)赵希男　粱三龙　潘德惠摘　要　针对一类系统提出了一种通用性...
预测的类别是： Computer
真实的类别是： Computer
================================================
输入的文本是：航空动力学报JOURNAL OF AEROSPACE POWER1999年　第14卷　第1期　VOL.14　No.1　1999变几何涡扇发动机几何调节对性能的影响朱之丽　李　东摘要：本文以高推重比涡扇...
预测的类别是： Space
真实的类别是： Space
================================================
输入的文本是：【 文献号 】1-4242【原文出处】图书馆论坛【原刊地名】广州【原刊期号】199503【原刊页号】13-15【分 类 号】G9【分 类 名】图书馆学、信息科学、资料工作【 作  者 】周坚宇【复印期...
预测的类别是： Sports
真实的类别是： Sports
================================================
输入的文本是：产业与环境INDUSTRY AND ENVIRONMENT1998年 第20卷 第4期 Vol.20 No.4 1998科技期刊采矿——事实与数字引　言本期《产业与环境》中的向前看文章并没有十分详细地...
预测的类别是： Enviornment
真实的类别是： Enviornment
================================================
输入的文本是：环境技术ENVIRONMENTAL TECHNOLOGY1999年 第3期 No.3 1999正弦振动试验中物理计算闫立摘要：本文通过阐述正弦振动试验技术涉及的物理概念、力学原理，编写了较适用的C语言...
预测的类别是： Space
真实的类别是： Enviornment
================================================
```


下面是一些实现的对比：<br>
`transformer`：
|    评价指标     |precision |  recall  |  f1-score | support|  
|    ----        |   ----   |    ----  |    ----   |    ----|
|     accuracy   |          |          |    0.86   |   9833 |
|    macro avg   |    0.72  |    0.45  |    0.46   |   9833 |
| weighted avg   |    0.85  |    0.86  |    0.84   |   9833 |

`bistm`：
|    评价指标     |precision |  recall  |  f1-score | support|  
|    ----        |   ----   |    ----  |    ----   |    ----|
|  accuracy      |          |          |    0.77   |   9833 |
|    macro avg   |    0.47  |    0.40  |    0.41   |   9833 |
| weighted avg   |    0.76  |    0.77  |    0.76   |   9833 |

`bilstmattn`：
|    评价指标     |precision |  recall  |  f1-score | support|  
|    ----        |   ----   |    ----  |    ----   |    ----|
|     accuracy   |          |          |    0.92   |   9833 |
|    macro avg   |    0.70  |    0.64  |    0.65   |   9833 |
| weighted avg   |    0.93  |    0.92  |    0.92   |   9833 |

`textrcnn`：
|    评价指标     |precision |  recall  |  f1-score | support|  
|    ----        |   ----   |    ----  |    ----   |    ----|   
|     accuracy   |          |          |    0.89   |   9833 |
|    macro avg   |    0.71  |    0.46  |    0.48   |   9833 |
| weighted avg   |    0.88  |    0.89  |    0.87   |   9833 |

`rcnn`：<br>
很奇怪，rcnn网络并没有得到有效的训练
|    评价指标     |precision |  recall  |  f1-score | support|  
|    ----        |   ----   |    ----  |    ----   |    ----| 
|     accuracy   |          |          |    0.16   |   9833 |
|    macro avg   |    0.01  |    0.05  |    0.02   |   9833 |
| weighted avg   |    0.04  |    0.16  |    0.05   |   9833 |

十分感谢以下仓库，给了自己很多参考：<br>
https://github.com/jiangxinyang227/NLP-Project/tree/master/text_classifier <br>
https://github.com/gaussic/text-classification-cnn-rnn
