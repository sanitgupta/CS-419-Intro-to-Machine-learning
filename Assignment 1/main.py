
# coding: utf-8

# In[2]:

import sys
import numpy as np
import pandas as pd
import time
start = time.clock()
#your code here    



if len(sys.argv) != 8 or sys.argv[1] != '--train_data' or sys.argv[3] != '--test_data' or sys.argv[5] != '--min_leaf_size' or (sys.argv[7]!= '--absolute' and sys.argv[7] != '--mean_squared'):
    print("Arguments not provided")
    exit(0)


df = pd.read_csv(sys.argv[2])
test = pd.read_csv(sys.argv[4])
min_leaf_size = int(sys.argv[6])
errtype = sys.argv[7]

#df = df.sample(frac=1).reset_index(drop=True)
train = df[0:int(5*len(df)/6)]


validation = df[int(5*len(df)/6):len(df)]
# In[4]:
f = list(train)[0:-2]
f2 = list(train)[-1]

lan = len(train)
train['id'] = range(lan)
train.head()


# In[5]:


def gini(target_col):
    types,counts = np.unique(target_col,return_counts = True)
    counts = counts / np.sum(counts)
    gini = np.sum(np.multiply(counts,counts)) 
    gini = 1 - gini
    return gini


# In[9]:




# In[8]:


mean_output = np.mean(train[f2])
mean_output 



class Node:
    def __init__(self):
        self.left         = None
        self.right        = None
        self.attri        = None
        self.divider      = None
        self.value        = None
        self.node_type    = None   #if type = 1 then internal node; type = 0 then leaf     
        self.error = 0     

# In[11]:


root = Node()


# In[21]:


def Partition(data,features):
    
    min_var = float("inf")

    for feature in features:
        #print(feature)
        x = data
        #print('x',x.type)
        x = x.sort_values(by=[feature])[['id',feature,f2]].reset_index()
        #print(x.head())
        total_l      = 0
        square_tot_l = 0
        total_r      = np.sum(x[f2])
        square_tot_r = np.sum(np.multiply(x[f2], x[f2]))
        #print(len(data))
        for k in range(len(data[f2])):
            id1     = []
            id2     = []
            if k == 0:
                continue
            #print(x[f2][k], k)
            if k < min_leaf_size:
                
                total_l = total_l + x[f2][k]
                total_r = total_r - x[f2][k]
                
                square_tot_l = square_tot_l + np.square(x[f2][k])
                square_tot_r = square_tot_r - np.square(x[f2][k])
                continue 
            if k == len(data) - 19: #tbd
                break
            total_l      = total_l + x[f2][k]
            total_r      = total_r - x[f2][k]
            square_tot_l = square_tot_l + np.square(x[f2][k])
            square_tot_r = square_tot_r - np.square(x[f2][k])
            if x[feature][k] == x[feature][k-1]:
                continue
            for i in range(k):
                id1.append(i)
            for i in range(k,len(data)):#tbd
                id2.append(i)
            var = square_tot_l - np.square(total_l)/len(id1)+ square_tot_r - np.square(total_r)/len(id2)
            if var < min_var:
                min_var     = var
                #print(feature,var, k)
                min_k       = x[feature][k]
                min_id1     = id1
                min_id2     = id2
                min_feature = feature
    
    if min_var == float("inf"):
        return None, None, None, None
    return min_feature, min_k, min_id1, min_id2

        
        
            
            


# In[22]:


def gen_tree(tree,data,features):
    tree.value = np.mean(data[f2])
    tree.node_type  = 0
    attribute, attribute_value, id1, id2 = Partition(data,features)
    
    if attribute != None:
        #print('attribute')
        tree.node_type = 1
        tree.attri = attribute
        f_l = []
        f_r = []
        for i in features:
            f_l.append(i)
            f_r.append(i)
        
        f_l.remove(attribute)
        f_r.remove(attribute)
        
        tree.divider = attribute_value
        d1 = data[data[attribute] < attribute_value]
        #print(attribute)
        #print(attribute_value)
        #print('d',d1['output'][1])
        
        d2 = data[data[attribute] >= attribute_value]
        #print(d2)
        tree.left  = Node()
        tree.right = Node()
        
        #print(features)
        tree.left  = gen_tree(tree.left,d1,f_l)
        #print(features)
        #print("================================================================")
        tree.right = gen_tree(tree.right,d2,f_r)
    return tree  


# In[109]:


gen_tree(root,train,f)



# In[24]:



# In[25]:


def predict(query):
    x = root
    while(x.node_type != 0 ):
        atr = x.attri
        #print(x.node_type, atr)
        #print(atr)
        div = x.divider
        if query[atr][0] < div:
            x = x.left
        else:
            x = x.right
    return x.value          
            
def predictP(query):
    x = root
    while(x.node_type != 0 ):
        atr = x.attri
        x.error += (query[f2]-x.value)**2
        div = x.divider
        if query[atr] < div:
            x = x.left
        else:
            x = x.right
    return x.value          
            


def nodeLoss(tree, validation):
    z = tree
    for i in range(len(validation)):
        #print(validation.iloc[i]['X1'])
        predictP(validation.iloc[i])

def prune(tree):
    if tree.node_type != 0:
        prune(tree.left)
        prune(tree.right)

        if tree.error < tree.left.error + tree.right.error:
            tree.left = None
            tree.right = None
            tree.node_type = 0

nodeLoss(root, validation)
prune(root)

print("train time =",time.clock() - start)
start = time.clock()
def rmse(test_data):
    sum = 0
    n = len(test_data[f2])
    for i in range(n):
        #print(i,test_data.iloc[[i]].reset_index())
        pred = predict(test_data.iloc[[i]].reset_index())
        sum += np.square(pred - test_data[f2].iloc[i])
    sum = sum/n
    return np.sqrt(sum)

def absolute(test_data):
    sum = 0
    n = len(test_data[f2])
    for i in range(n):
        #print(i,test_data.iloc[[i]].reset_index())
        pred = predict(test_data.iloc[[i]].reset_index())
        sum += abs(pred - test_data[f2].iloc[i])
    sum = sum/n
    return sum

def predict_test(test_data):
    n = len(test_data.axes[0])
    predictions = []
    for i in range(n):
        #print(i,test_data.iloc[[i]].reset_index())
        pred = round(predict(test_data.iloc[[i]].reset_index()))
        predictions.append(pred)
    return predictions
        


# In[110]:


predictions = pd.DataFrame(predict_test(test))
def treeSize(node):
    if node == None:
        return 0
    return treeSize(node.left) + treeSize(node.right) + 1


print(treeSize(root))

# if errtype == '--absolute':
#     print(absolute(tst))
# else:
#     print(rmse(tst))

predictions = pd.DataFrame(predict_test(test))
print("test time =",time.clock() - start)

predictions.to_csv('predtoy.csv')
