import gradio as gr
from predict import model_2, model_4, tokenizer, name_list2, name_list4,predict


def audit(text):
    cls2 = predict(text,model_2,tokenizer)
    if cls2 == 0:
        return name_list2[cls2]

    cls4 = predict(text,model_4,tokenizer)
    return name_list4[cls4]


iface = gr.Interface(fn=audit, inputs="text", outputs="text")
iface.launch()
