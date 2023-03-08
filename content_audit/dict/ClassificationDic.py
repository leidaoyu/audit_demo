from content_audit.dict.GetKeywordProcessor import keyword_processor


def Audit(sentence):
    """基于词典的文本内容审核
    :param sentence:需要审核的字符串句子
    """
    hits_list = keyword_processor.extract_keywords(sentence, span_info=True)
    if hits_list:
        result = '不合规'
        hits = {}
        for i in range(len(hits_list)):
            if hits_list[i][0] not in hits:
                hits.update({hits_list[i][0]: [sentence[hits_list[i][1]:hits_list[i][2]]]})
            else:
                hits[hits_list[i][0]].append(sentence[hits_list[i][1]:hits_list[i][2]])
    else:
        result = '合规'
        hits = {}

    result_dict = {
        'text': sentence,
        'result': result,
        'hits': hits
    }

    return result_dict


if __name__ == '__main__':
    # sentence = '习近平今天习进平玩的women很一叶情开心好爽，今晚要去一页情,djasoiji ad掉的哦按哦客服。安康，发得快卡的'
    # print(Audit(sentence, './sen_word_dictionary'))
    pass
