from flashtext import KeywordProcessor
import os


# 根据字典目录获得词典
def get_dic(dic_path, keyword_dict):
    for file in os.listdir(dic_path):
        # print(dic)
        dic_name = file.replace('.txt', '')
        # print(dic_name)
        keyword_list = [line.strip() for line in open(os.path.join(dic_path, file), 'r', encoding='utf-8')]
        keyword_dict.update({dic_name: keyword_list})

    return keyword_dict


# 获得flashtext的keyword_dict
dic_path = r'./data/sensitive_dict'
keyword_dict = {}
get_dic(dic_path, keyword_dict)
keyword_processor = KeywordProcessor()
keyword_processor.add_keywords_from_dict(keyword_dict)
