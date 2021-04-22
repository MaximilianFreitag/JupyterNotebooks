#!/usr/bin/env python
# coding: utf-8

# # Actual Code

# In[1]:


string = '( [ ( ) ] { } ( { [ ( ) ( ) ] ( ) ] ( ) } ) )'

#( [ ( ) ] { } ( { [ ( ) ( ) ] ( ) ] ( ) } ) )

def balanced_unbalanced(string): 
    
    stack = [] 
  
    # Traversing the string 
    for char in string: 
        if char in ["(", "{", "["]: 
  
            # Push the element in the stack 
            stack.append(string) 
        else: 
  
            # IF current character is not opening 
            # bracket, then it must be closing. 
            # So stack cannot be empty at this point. 
            if not stack: 
                return False
            current_char = stack.pop() 
            if current_char == '(': 
                if char != ")": 
                    return False
            if current_char == '{': 
                if char != "}": 
                    return False
            if current_char == '[': 
                if char != "]": 
                    return False
  
    # Check Empty Stack 
    if stack: 
        return False
    return True
  
   
if __name__ == "__main__": 
    
  
    # Toss the string into the created function 
    if balanced_unbalanced(string): 
        print("Output: Balanced") 
    else: 
        print("Output: Not Balanced") 


# ## So our string = '( [ ( ) ] { } ( { [ ( ) ( ) ] ( ) ] ( ) } ) )' is NOT balanced

# In[ ]:




