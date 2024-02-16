from iAgent.prompt.functionDB import *

########################################回答模板配置######################################################################
#函数
ans_fun = {'function': '',
       'parameters': '',
       'thought': 'Your thoughts on the task',
       "process":"enum[start,end,ongoing]"}

ans_plan={'plan':[],'thought':''}

ans_tri={'key':'enum[0,1]','thought':''}

########################################################################################################################



#规划
def plan(target):
    text = {"user": target}
    prompt = f"You are an independent and autonomous artificial intelligence model, and you can plan independently based on it." \
             f"focus on user." \
             f"\n{text}\n" \
             f".You should only respond in JSON format as described below{ans_plan}.step by step." \
             f"The output is just pure JSON format,with no other descripions"

    return prompt

#函数
def func(target,state):


    text = {"user": target, 'state':state,"function": function}
    prompt = f"You are an independent and autonomous artificial intelligence model, and you can complete tasks by selecting functions." \
             f"focus on user." \
             f"\n{text}\n" \
             f".You should only respond in JSON format as described below{ans_fun}" \
             f"The output is just pure JSON format,with no other descripions"

    return prompt

#前置触发

def trigger(target):
    text = {"user": target,}
    prompt = f"You, as an independent and autonomous artificial intelligence model, " \
             f"possess the prerogative to decide whether to respond or remain silent. " \
             f"Your focus should remain solely on the user, in this case, the term '{target}' which represents the user. " \
             f"Your response, if any, must adhere strictly to the JSON format outlined below:{ans_tri}.Approach this task step by step." \
             f"The output is just pure JSON format,with no other descripions"
    return prompt
