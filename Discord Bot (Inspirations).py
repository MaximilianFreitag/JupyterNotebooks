#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import discord
import os
import requests
import json




TOKEN = 11111111111111111111111




client = discord.Client()

#Requesting quote data from the API
def get_quote():
  response = requests.get('https://zenquotes.io/api/random')
  json_data = json.loads(response.text)
  quote = json_data[0]['q'] + '  - ' + json_data[0]['a']
  return(quote)


@client.event
async def on_ready():
    print('Logged in as {0.user}'.format(client))

#Handling user input
@client.event
async def on_message(message):
    
    if message.author == client.user:
      return

    if message.content.startswith('hello'):
      await message.channel.send('What do you want Max?')

    if message.content.startswith('bye'):
      await message.channel.send('Have a good one')

    if message.content.startswith('!inspire'):
      quote = get_quote()
      await message.channel.send(quote)  

    client.run()      

client.run(os.getenv('TOKEN'))




