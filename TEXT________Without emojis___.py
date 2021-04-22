#!/usr/bin/env python
# coding: utf-8

# In[23]:


import unicodedata
from unidecode import unidecode

def deEmojify(inputString):
    returnString = ""

    for character in inputString:
        try:
            character.encode("ascii")
            returnString += character
        except UnicodeEncodeError:
            replaced = unidecode(str(character))
            if replaced != '':
                returnString += replaced
            else:
                try:
                     returnString += "[" + unicodedata.name(character) + "]"
                except ValueError:
                     returnString += "[x]"

    return returnString



string = '🙁😠hello😡😞😟😣__emoji😖','🙁😠___free😡😞___world😟😣😖'
print(deEmojify(string))


# In[ ]:




