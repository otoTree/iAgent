import erniebot

# List supported models
models = erniebot.Model.list()


# ernie-3.5             文心大模型（ernie-3.5）
# ernie-turbo           文心大模型（ernie-turbo）
# ernie-4.0             文心大模型（ernie-4.0）
# ernie-longtext        文心大模型（ernie-longtext）
# ernie-text-embedding  文心百中语义模型
# ernie-vilg-v2         文心一格模型

# Set authentication params
erniebot.api_type = "aistudio"
erniebot.access_token = "<YOUR-TOKEN>"




class CHAT():
    def __init__(self):
        self.message=[]

    def connect(self,messages):
        response = erniebot.ChatCompletion.create(model="ernie-3.5", messages=messages)
        return response.get_result()

    def chat(self,text):
        user={'role':'user','content':text}
        self.message.append(user)
        response = self.connect(self.message)
        assistant={'role':'assistant','content':response}
        self.message.append(assistant)
        return response
