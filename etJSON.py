import json

def get_1_JSON(text):
    return json.loads(text.split("```json")[1].split("```")[0].replace("\'",'\"'))


