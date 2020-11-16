# -*- coding:utf-8 -*-

import os
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
    fp = open(item_file)
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

def get_ave_score(input_file):
    """
    Args:
        input_file:user rating file
    Return:
        a dict, key:itemid value: ave_score
    """
    if not os.path.exists(input_file):
        return {}
    linenum = 0
    record = {}
    ave_score = {}
    fp = open(input_file)
    for line in fp:
        if linenum == 0:
            linenum += 1
            continue
        item = line.strip().split(",")
        if len(item) < 4:
            continue
        userid, itemid, rating = item[0], item[1], float(item[2])
        if itemid not in record:
            record[itemid] = [0, 0]
        record[itemid][0] += rating
        record[itemid][1] += 1
    fp.close()
    for itemid in record:
        ave_score[itemid] = round(record[itemid][0]/record[itemid][1], 3)
    return ave_score

def get_train_data(input_file):
    """
    :param input_file: ratings.csv
    :return: list:[(userid,itemid,label),...] rating>=4?label=1:label=0 喜欢为1，否则为0
    """
    if not os.path.exists(input_file):
        return []
    score_dict = get_ave_score(input_file)
    linenum = 0
    pos_dict = {}
    neg_dict = {}
    train_data = []
    score_thr = 4.0
    fp = open(input_file)
    for line in fp:
        if linenum == 0:
            linenum += 1
            continue
        item = line.strip().split(",")
        if len(item) < 4:
            continue
        userid, itemid, rating = item[0], item[1], float(item[2])
        if userid not in pos_dict:
            pos_dict[userid] = []
        if userid not in neg_dict:
            neg_dict[userid] = []
        if rating >= score_thr:
            pos_dict[userid].append((itemid,1))
        else:
            score = score_dict.get(itemid,0) #因为需要负采样，所以先获取评价低的真实评分
            neg_dict[userid].append((itemid,score))
    fp.close()
    for userid in pos_dict:
        data_num = min(len(pos_dict[userid]),len(neg_dict.get(userid,[])))
        if data_num>0:
            train_data += [(userid,zuhe[0],zuhe[1]) for zuhe in pos_dict[userid]][:data_num]
        else:
            continue
        sorted_neg_list = sorted(neg_dict[userid],key=lambda element:element[1],reverse=True)\
        [:data_num]
        train_data += [(userid,zuhe[0],0) for zuhe in sorted_neg_list]

        # if userid=="24":
        #     print len(pos_dict[userid])
        #     print len(neg_dict[userid])
        #     print len(sorted_neg_list)
        #     print sorted_neg_list

    return train_data

if __name__=="__main__":
    # item_dict = get_item_info("../data/movies.csv")
    # print len(item_dict)
    # print item_dict["1"]
    # print item_dict["11"]

    # score_dict = get_ave_score("../data/ratings.csv")
    # print len(score_dict)
    # print score_dict["31"]

    train_data = get_train_data("../data/ratings.csv")
    print(len(train_data))
    print(train_data[:50])