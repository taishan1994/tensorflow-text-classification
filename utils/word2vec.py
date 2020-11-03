from gensim.models import Word2Vec
from gensim.models.word2vec import PathLineSentences
import multiprocessing
import os
import sys
import logging

# 日志信息输出
program = os.path.basename(sys.argv[0])
logger = logging.getLogger(program)
logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
logging.root.setLevel(level=logging.INFO)
logger.info("running %s" % ' '.join(sys.argv))

# check and process input arguments
# if len(sys.argv) < 4:
#     print(globals()['__doc__'] % locals())
#     sys.exit(1)
# input_dir, outp1, outp2 = sys.argv[1:4]

# 训练模型 
# 输入语料目录:PathLineSentences(input_dir)
# embedding size:200 共现窗口大小:10 去除出现次数10以下的词,多线程运行,迭代10次
model = Word2Vec(PathLineSentences('/content/drive/My Drive/transformer/process/Fudan/word2vec/data/'),
                     size=200, window=10, min_count=10,
                     workers=multiprocessing.cpu_count(), iter=10)
model.save('/content/drive/My Drive/transformer/process/Fudan/word2vec/model/Word2vec.w2v')