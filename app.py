import gradio as gr

from functions.predict import model_2, model_4, tokenizer, name_list2, name_list4, predict
from content_audit.dict.ClassificationDic import Audit


def audit(text):
    flag = True  # 指示是否是低质灌水
    result = {
        "text": text,
        "result": "",
        "label": []
    }

    dict_result = Audit(text)

    if dict_result["result"] == "不合规":
        print('dict hit')
        result["result"] = "不合规"
        result["label"].extend(dict_result["hits"].keys())
        result.update({"hits": dict_result["hits"]})
    else:
        cls2 = name_list2[predict(text, model_2, tokenizer)]

        if cls2 == '合规' and flag:
            result = {
                "text": text,
                "result": "合规"
            }
        else:
            print('dl hit')
            cls4 = predict(text, model_4, tokenizer)
            result["result"] = "不合规"
            result["label"] = name_list4[cls4]
    return result


iface = gr.Interface(fn=audit, inputs="text", outputs="text")
iface.launch()
