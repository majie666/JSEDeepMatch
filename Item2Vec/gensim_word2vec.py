# -*-coding:utf8-*-
from gensim.models import word2vec
import gensim

# sentences = word2vec.Text8Corpus("../data/train_data.txt")
sentences = word2vec.LineSentence("../data/train_data.txt")  # 加载语料,LineSentence用于处理分行分词语料
model=word2vec.Word2Vec(sentences,size=128,window=5,sample=0.001,negative=5,hs=0,sg=1,iter=100,workers=5)
model.wv.save_word2vec_format('../data/item_vec.txt',binary = False)

# model = gensim.models.Word2Vec.load('../data/word2vec_model')
# model.train(more_sentences)
# print model.wv["1"]
print(model.most_similar("27"))