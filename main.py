import json
import os
import discum
import sys
from random import randrange
from MyAI.model import NeuralNet
from MyAI.nltk_utils import bag_of_words, tokenize
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
bot = discum.Client(token="THE-TOKEN", log=False)

global myID
myID: int = 0
# My Custom Arrays.
vorstellen = ["Hallo! Ich bin der AI-Responder, von Fido (https://github.com/Fidode07/DiscordResponderAI). Ich bin nicht zum Unterhalten da, ich versuche dafür zu sorgen, dass keine unnötigen Fragen/Spam dafür sorgen, dass Fido abgelenkt wird. Um was geht es denn?",
              "Guten Tag. Vorab: Ich bin eine KI! Ich bin hier, damit Fido nicht von unnötigen Nachrichten, wie Spam abgelenkt wird. Sieh mich wie ein intelligenter Filter, okay? Allerdings bin ich nicht hier, um Unterhaltungen zu führen, sondern deine Frage/Nachricht zu klassifizieren und je nachdem benachrichtige ich ihn oder eben nicht."]

greetResponse = ["Guten Tag, ich bin es, die KI! Wie kann ich dir helfen?",
"Hallo, ich bin es wieder, die KI. Kann ich dir behilflich sein?",
"Moin, ich bin es, die KI. Wie kann ich helfen?"]

if torch.cuda.is_available():
    print("WE USE GRAPHICS CARD, CUZ CUDA IS AVAILABLE!")
else:
    print("Let's try to run this Shit on our CPU ...")

with open("MyAI/Models/german.json", 'r', encoding='utf-8') as json_data:
    intents = json.load(json_data)

FILE = f"MyAI/Models/german.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()
PREFIX = "[AIResponser]"


def send_msg(channel_id, message):
    bot.sendMessage(channel_id, f"{PREFIX} {str(message)}")


class UserHandler:

    def __init__(self) -> None:
        pass

    def user_is_in_db(self, userID):
        user_is: bool = False
        with open("users.txt", "r+") as fp:
            line = fp.readline()
            cnt = 1
            while line:
                if str(userID) in str(line.strip()):
                    user_is = True
                line = fp.readline()
                cnt += 1
        return user_is

    def add_user_to_db(self, userID):
        with open("users.txt", "a") as f:
            f.write(f"\n{userID}")


UHandler = UserHandler()


@bot.gateway.command
def mybot(resp):
    global myID
    if resp.event.ready_supplemental:
        user = bot.gateway.session.user
        print("Logged in as {}#{} ID: {}".format(user['username'], user['discriminator'], user['id']))
        myID = user['id']
    if resp.event.message:
        m = resp.parsed.auto()
        guildID = m['guild_id'] if 'guild_id' in m else None
        if guildID is None:
            if myID != m['author']['id']:
                channelID = m['channel_id']
                username = m['author']['username']
                discriminator = m['author']['discriminator']
                content = m['content']
                authorID = m['author']['id']
                sentence = tokenize(content.lower())
                X = bag_of_words(sentence, all_words)
                X = X.reshape(1, X.shape[0])
                X = torch.from_numpy(X).to(device)

                output = model(X)
                _, predicted = torch.max(output, dim=1)

                tag = tags[predicted.item()]

                probs = torch.softmax(output, dim=1)
                prob = probs[0][predicted.item()]

                print("Genauigkeit: " + str(prob.item()))

                if prob.item() > 0.96:
                    for intent in intents['intents']:
                        if tag == intent["tag"]:
                            if intent['tag'] == "greeting":
                                if UHandler.user_is_in_db(authorID):
                                    send_msg(channelID, greetResponse[randrange(0, len(greetResponse))])
                                else:
                                    UHandler.add_user_to_db(authorID)
                                    send_msg(channelID, vorstellen[randrange(0, len(vorstellen))])
                            elif intent['tag'] == "stopword":
                                return


bot.gateway.run(auto_reconnect=True)
