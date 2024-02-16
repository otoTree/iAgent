import json

def call(text):
    a = json.loads(text)
    parameters = ""

    try:
        a['function']
    except:
        print("找不到函数")
        return

    try:
        if a['parameters'] == {}:
            parameters = "0"
        else:
            for i in a["parameters"]:
                parameters.join(f"{i}:{a['parameters'][i]}")
    except:
        print(f"函数{a['function']}的参数提取失败")
        return

    code = """
import tool.{} as call

call.main({})
    """.format(a['function'], parameters)
    try:
        exec(code)
    except:
        print(f"{a['function']}函数调用失败")
        return