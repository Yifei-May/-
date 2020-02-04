import numpy as np
from math import log2
import pandas as pd

def calcEntropy(label,feature=None):
    num=len(label)
    if np.any(feature!=None):
        featureCnt=np.unique(feature)
        rat=[list(feature).count([i])/num for i in featureCnt]
        ent=[]
        for i in featureCnt:
            subData=splitByFeature(np.hstack((feature,label)),0,value=i)
            ent.append(calcEntropy(label=[i[-1] for i in subData]))
        return sum([rat[i]*ent[i] for i in range(len(ent))])
    else:
        labelCnt={}
        for i in label:
            curLabel=i[0]
            if curLabel in labelCnt.keys():
                labelCnt[curLabel]+=1
            else:
                labelCnt[curLabel]=1
        ent=0
        for i in labelCnt.keys():
            ent+=-(labelCnt[i]/num)*log2(labelCnt[i]/num)
        return ent

def splitByFeature(dataset,axis,value):
    subDataset=[]
    for record in dataset:
        if record[axis]==value:
            subDataset.append(record)
    return np.array(subDataset)

def ChooseBestFeatureToSplit(train_array):
    train_array=np.array(train_array)
    numFeatures=np.shape(train_array)[1]-1

    ent_base=calcEntropy(label=train_array[:,-1:])
    ent=np.ones(numFeatures)
    for i in range(numFeatures):
        ent[i]=calcEntropy(label=train_array[:,-1:],feature=train_array[:,i:i+1])
    gain=ent_base-ent
    bestAxis=np.argmax(gain)
    return(bestAxis) 

def getMostLabel(train_array):
    train_array=np.array(train_array)
    label=train_array[:,-1]
    labelCnt={}
    for i in label:
        if i in labelCnt.keys():
            labelCnt[i]+=1
        else:
            labelCnt[i]=1
    return list(labelCnt.keys())[np.argmax(list(labelCnt.values()))]

def createTree(dataset_to_divide,feature_to_divide):
    dataset_to_divide=np.array(dataset_to_divide)

    if len(np.unique(dataset_to_divide[:,-1]))==1: return dataset_to_divide[0,-1] #当最优划分特征某一值下所有样本均为同一类，则返回该类别，停止继续划分
    if np.all(feature_to_divide==None): return getMostLabel(dataset_to_divide)

    bestAxis=ChooseBestFeatureToSplit(dataset_to_divide) #选择最优划分特征
    bestFeature=feature_to_divide[bestAxis] 
    

    myTree={bestFeature:{}} #分类结果以字典形式保存
    bestFeatureType=np.unique(dataset_to_divide[:,bestAxis])
    
    for i in bestFeatureType:
        dataDivided=splitByFeature(dataset_to_divide,bestAxis,value=i)
        myTree[bestFeature][i]=createTree(np.delete(dataDivided,bestAxis,axis=1),np.delete(feature_to_divide,bestAxis))
    return myTree

# 创建带预划分标签的决策树
def createTreeWithLabel(dataset_to_divide,feature_to_divide):
    dataset_to_divide=np.array(dataset_to_divide)

    if len(np.unique(dataset_to_divide[:,-1]))==1: return dataset_to_divide[0,-1] #当最优划分特征某一值下所有样本均为同一类，则返回该类别，停止继续划分
    if np.all(feature_to_divide==None): return getMostLabel(dataset_to_divide)

    bestAxis=ChooseBestFeatureToSplit(dataset_to_divide) #选择最优划分特征
    bestFeature=feature_to_divide[bestAxis] 
    

    myTree={bestFeature:{}} #分类结果以字典形式保存 
    bestFeatureType=np.unique(dataset_to_divide[:,bestAxis])
    
    for i in np.concatenate((['_vpdl'],bestFeatureType)):
        if i=='_vpdl':
            myTree[bestFeature][i]=getMostLabel(dataset_to_divide) #_vpdl:添加划分前的标签以后续剪枝
        else:
            dataDivided=splitByFeature(dataset_to_divide,bestAxis,value=i)
            myTree[bestFeature][i]=createTreeWithLabel(np.delete(dataDivided,bestAxis,axis=1),np.delete(feature_to_divide,bestAxis))
    return myTree

def convertTree(inputLabeledTree):
    newTree=inputLabeledTree.copy()
    nodeName=list(inputLabeledTree.keys())[0]
    featureType=list(inputLabeledTree[nodeName].keys())
    for i in featureType:
        if i=='_vpdl':
            del newTree[nodeName][i]
        elif type(inputLabeledTree[nodeName][i])==dict:
            newTree[nodeName][i]=convertTree(inputLabeledTree[nodeName][i])
    return newTree

def treePostPruning(inputLabeledTree,test_array,feature_name):
    test_array=np.array(test_array)
    feature_name=list(feature_name)

    newTree=inputLabeledTree.copy()
    nodeName=list(inputLabeledTree.keys())[0]

    featPreLabel=newTree[nodeName].pop('_vpdl') #预划分标签
    featureType=list(inputLabeledTree[nodeName].keys())
    axis=feature_name.index(nodeName)

    

    subtreeFlag=0 #初始化: 当前结点的所有分枝均为叶结点
    dataFlag=1 if sum(np.shape(test_array))>0 else 0 #当前结点是否有划分数据
    
    for i in featureType:
        if dataFlag==1 and type(inputLabeledTree[nodeName][i])==dict:
            subtreeFlag=1 #存在一个子结点为子树，递归
            newTree[nodeName][i]=treePostPruning(inputLabeledTree[nodeName][i],splitByFeature(test_array,axis,i),feature_name)
            if type(newTree[nodeName][i])!=dict:
                subtreeFlag=0
        elif dataFlag==0 and type(inputLabeledTree[nodeName][i])==dict:
            subtreeFlag=1
            newTree[nodeName][i]=convertTree(inputLabeledTree[nodeName][i])

    if dataFlag==1 and subtreeFlag==0:
        ratioPreDivision=sum(test_array[:,-1]==featPreLabel)/np.shape(test_array)[0]
        equalCnt=0
        for i in featureType:
            subDataset=splitByFeature(test_array,axis,i)
            if sum(np.shape(subDataset))>0:
                equalCnt=equalCnt+sum(subDataset[:,-1]==newTree[nodeName][i])
        ratioAftDivision=equalCnt/np.shape(test_array)[0]
        if ratioAftDivision<ratioPreDivision:
            newTree=featPreLabel
    return newTree


           
if __name__=='__main__':
    #读取数据
    dataset=pd.read_csv("watermelon.csv")
    dataset=np.array(dataset)[:,1:]
    feature_name=['色泽','根蒂','敲声','纹理','脐部','触感']
    #用所有数据生成ID3决策树
    inputTree=createTree(dataset,feature_name)
    print('所有数据下的ID3决策树:\n',inputTree)

    #划分训练集和测试集
    index=np.subtract([1,2,3,6,7,10,14,15,16,17],1)
    train_array=dataset[index,:]
    index=np.subtract([4,5,8,9,11,12,13],1)
    test_array=dataset[index,:]

    #训练数据生成的ID3决策树
    inputTree=createTree(train_array,feature_name)
    print('训练数据下的ID3决策树:\n',inputTree)

    #生成带有pre标签的决策树以后剪枝
    inputTreeWithLabel=createTreeWithLabel(train_array,feature_name)
    tree_postPruning=treePostPruning(inputTreeWithLabel,test_array,feature_name)
    print('后剪枝下的决策树:\n',tree_postPruning)
   