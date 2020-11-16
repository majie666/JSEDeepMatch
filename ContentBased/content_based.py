#-*-coding:utf8-*-
from __future__ import division
import os
import operator
import  sys
sys.path.append("../")
import util.read as read

def get_up(item_cate, input_file):
    """
    Args:
        item_cate:key itemid, value: dict , key category value ratio
        input_file:user rating file
    Return:
        a dict: key userid, value [(category, ratio), (category1, ratio1)] 用户对种类的偏好
    """
    if not os.path.exists(input_file):
        return {}
    record = {}
    up = {}
    linenum = 0
    score_thr = 4.0
    topk = 2 # 选出用户最喜欢的两个类别
    fp = open(input_file)
    for line in fp:
        if linenum == 0:
            linenum += 1
            continue
        item = line.strip().split(',')
        if len(item) < 4:
            continue
        userid, itemid, rating, timestamp = item[0], item[1], float(item[2]), int(item[3])
        if rating < score_thr:
            continue
        if itemid not in item_cate:
            continue
        time_score = get_time_score(timestamp)
        if userid not in record:
            record[userid] = {}
        for fix_cate in item_cate[itemid]:
            if fix_cate not in record[userid]:
                record[userid][fix_cate] = 0
            record[userid][fix_cate] += rating * time_score * item_cate[itemid][fix_cate]
    fp.close()
    for userid in record:
        if userid not in up:
            up[userid] = []
        total_score = 0
        for zuhe in sorted(record[userid].iteritems(), key = operator.itemgetter(1), reverse=True)[:topk]:
            up[userid].append((zuhe[0], zuhe[1]))
            total_score += zuhe[1]
        for index in range(len(up[userid])):
            up[userid][index] = (up[userid][index][0], round(up[userid][index][1]/total_score, 3))
    return up

def get_time_score(timestamp):
    """
    Args:
        timestamp:input timestamp
    Return:
        time score
    """
    fix_time_stamp = 1476086345 # 2016-10-10 15:59:05 数据集中最新时间
    total_sec = 24*60*60
    delta = (fix_time_stamp - timestamp)/total_sec/30 # 时间衰减
    return round(1/(1+delta), 3)

def recom(cate_item_sort, up, userid, topk= 5):
    """
    Args:
        cate_item_sort:reverse sort
        up:user profile
        userid:fix userid to recom
        topk:recom num
    Return:
         a dict, key userid value [itemid1, itemid2]
    """
    if userid not in up:
        return {}
    recom_result = {}
    if userid not in recom_result:
        recom_result[userid] = []
    for zuhe in up[userid]:
        cate = zuhe[0]
        ratio = zuhe[1]
        num = int(topk*ratio) + 1
        if cate not in cate_item_sort:
            continue
        recom_list = cate_item_sort[cate][:num]
        recom_result[userid] += recom_list

    item_info = read.get_item_info("../data/movies.csv")
    print("result---")
    for itemid in recom_result[userid]:
        print(item_info[itemid])

    return  recom_result

def run_main():
    ave_score = read.get_ave_score("../data/ratings.csv")
    item_cate, cate_item_sort =read.get_item_cate(ave_score, "../data/movies.csv")
    up = get_up(item_cate, "../data/ratings.csv")
    print len(up)
    print up["11"]
    print recom(cate_item_sort, up, "11")

if __name__ == "__main__":
    run_main()