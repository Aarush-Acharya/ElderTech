from fastapi import FastAPI
import nltk
nltk.download('punkt')
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
import json
import pickle
import numpy as np
from keras.models import load_model
import random
from pydantic import BaseModel

lemmatizer = WordNetLemmatizer()

class Body(BaseModel):
    userQuery: str

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello World"}

intents = json.loads(open('intents.json').read())

words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model=load_model('chatbotmodel.h5')


@app.post("/getResponse")
async def giveResponse(body: Body):
    ints = predict_class(body.userQuery)
    res = get_responses(ints, intents)
    return {"result":res}


def clean_up_sentence(sentence):                                                    
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words
    
def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0]*len(words)
    for w in sentence_words:
        for i,word in enumerate(words):
            if word == w:
                bag[i] =1
    return np.array(bag)

def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    ERORR_THRESHOLD = 0.25
    result = [[i, r] for i , r in enumerate(res) if r > ERORR_THRESHOLD]
    result.sort(key=lambda x: x[1], reverse=True)
    return_list =[]
    for r in result:
        return_list.append({'intent': classes[r[0]], 'probability':str(r[1])})
    return return_list

def get_responses(intents_list, intents_json):
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result

