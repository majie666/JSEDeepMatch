# -*-coding:utf8-*-
import os
import sys
def produce_train_data(input_file,out_file):
    """
    :param input_file: user behavior file
    :param out_file: output file
    :return:
    """
    if not os.path.exists(input_file):
        return
    record = {}
    score_thr = 4.0
    num = 0
    fp = open(input_file)
    for line in fp:
        if num == 0:
            num += 1
            continue
        item = line.strip().split(',')
        if len(item) < 4:
            continue
        userid,itemid,rating = item[0],item[1],float(item[2])
        if rating<score_thr:
            continue
        if userid not in record:
            record[userid] = []
        record[userid].append(itemid)
    fp.close()
    fw = open(out_file,'w+')
    for userid in record:
        fw.write(" ".join(record[userid])+"\n")
    fw.close()

if __name__=="__main__":
    # if len(sys.argv)<3:
    #     print "error"
    #     sys.exit()
    # else:
    #     inputfile = sys.argv[1]
    #     outputfile = sys.argv[2]
    #     produce_train_data(inputfile,outputfile)
    produce_train_data("../data/ratings.csv","../data/train_data.txt")