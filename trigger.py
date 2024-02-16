import prompt.prompt as pt
from etJSON import *
import connectLLM.wenxin as wx


def main(input_text):
    chat=wx.CHAT()
    try:
        response = chat.chat(pt.trigger(input_text))

    except:
        print("trigger 模块报错，可能连接出错")
        return

    try:
        json = get_1_JSON(response)
        return json['key']
    except:
        print("trigger提取json出错")
        return



