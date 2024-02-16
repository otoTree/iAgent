#能力函数
function = [
    {'name': 'GenerateImage',
     'description': 'Generate images based on prompts',
     'parameters': {'type': 'object',
                    'properties':
                        {'prompt': {'type': 'string', 'description': 'Describe image'},
                         'height': {'type': 'int', 'description': 'image height'},
                         'width': {'type': 'int', 'description': 'image width'}},
                    'required': ['prompt']}},
    {'name': 'WriteArticles',
     'description': 'Write articles based on user ideas',
     'parameters': {'type': 'object',
                    'properties': {
                        'idea': {'type': 'string', 'description': 'some idea'},
                        'MaxWordCount': {'type': 'int', 'description': 'The maximum number of words in an article'}},
                    'required': ['idea']}},
    {'name': 'Chat',
     'description': "When the user's prompt is unclear or nothing is done, this function should be chosen more",
     'parameters': {'type': 'object',
                    'properties': {'input': {'type': 'string', 'description': 'What you say'}},
                    'required': ['input']}},
    {'name': 'takePhoto',
     'description': 'Open the camera to capture images of the real world',
     'parameters': {'type': 'object',
                    'properties': {'key': {'type': 'string', 'description': ''}}
                    }
     },
    {'name': 'None',
     'description': "当用户输入内容无效时调用这个函数",
     'parameters': None}
]

#前置触发器
func_head = [{'name': 'trigger',
              'description': "You are an independent and autonomous artificial intelligence model, and you have the right to choose whether or not to answer. If you want to answer, key=1, otherwise key=0",
              'parameters': {'type': 'object',
                             'properties': {'key': {'type': 'int', 'description': ''}}
                             }}]

