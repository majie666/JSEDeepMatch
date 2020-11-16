# -*-coding:utf8-*-
import os
import numpy as np
import operator
def load_item_vec(input_file):
    """
    :param input_file: item_vec.txt
    :return: dict key:itemid value:[num1,num2...]
    """
    if not os.path.exists(input_file):
        return {}
    num = 0
    item_vec = {}
    fp = open(input_file)
    for line in fp:
        if num == 0:
            num += 1
            continue
        item = line.strip().split()
        if len(item) < 129:
            continue
        itemid = item[0]
        if itemid == "</s>":
            continue
        item_vec[itemid] = np.array([float(e) for e in item[1:]])
    fp.close()
    return item_vec

def cal_item_sim(item_vec,itemid,output_file):
    """
    :param item_vec:
    :param itemid: fixed itemid
    :param output_file:
    :return:
    """
    if itemid not in item_vec:
        return
    score = {}
    topk = 10
    fix_item_vec = item_vec[itemid]
    for tmp_itemid in item_vec:
        if tmp_itemid == itemid:
            continue
        tmp_item_vec = item_vec[tmp_itemid]
        # 分母为两个向量模的乘积
        fenmu = np.linalg.norm(fix_item_vec)*np.linalg.norm(tmp_item_vec)
        # 若分母为0，则分子也为0；否则为cosin距离
        if fenmu==0:
            score[tmp_itemid] = 0
        else:
            score[tmp_itemid] = round(np.dot(fix_item_vec,tmp_item_vec)/fenmu,3)
    fw = open(output_file,"w+")
    out_str = itemid+"\t"
    tmp_list = []
    item_info = get_item_info('../data/movies.csv')
    print('推荐与该电影相似的电影：', item_info[itemid])
    print('推荐列表top10：')
    for zuhe in sorted(score.items(),key=operator.itemgetter(1),reverse=True)[:topk]:
        tmp_list.append(zuhe[0]+"_"+str(zuhe[1]))
        print(item_info[zuhe[0]],',相似度：%f'%(zuhe[1]))
    out_str += ";".join(tmp_list)
    fw.write(out_str + "\n")
    fw.close()

def get_item_info(item_file):
    """
    get item info[title, genres]
    Args:
        item_file:input iteminfo file
    return:
        a dict, key itemid, value: [title, genres]
    """
    if not os.path.exists(item_file):
        return {}
    num = 0
    fp = open(item_file, encoding='UTF-8')
    item_info ={}
    for line in fp:
        if num == 0:
            num += 1
            continue
        item = line.strip().split(',')
        if len(item) < 3:
            continue
        if len(item) == 3:
            [itemid, title, genres] = item
        elif len(item) > 3:
            itemid = item[0]
            genres = item[-1]
            title = ','.join(item[1:-1])
        if itemid not in item_info:
            item_info[itemid] = [title, genres]
    fp.close()
    return item_info

def run_main(input_file,output_file):
    item_vec = load_item_vec(input_file)
    cal_item_sim(item_vec,"27",output_file)

if __name__=="__main__":
    # item_vec = load_item_vec("../data/item_vec.txt")
    # print len(item_vec)
    # print item_vec["318"]
    run_main("../data/item_vec.txt","../data/sim_result.txt")