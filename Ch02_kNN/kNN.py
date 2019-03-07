#coding:utf-8，
# K临近算法
# M小白
from numpy import *
import operator
import matplotlib
import matplotlib.pyplot as plt
from os import listdir                                #列出给定目录的文件名

# 2.1 简单K临近算法
def createDataSet():
    group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels = ['A','A','B','B']
    return group, labels

def classify0(inX, dataSet, labels, k):             #k临近算法 主要函数 inX为需要判断的样本的坐标,dataSet为坐标数据集 n*2                                                    
    dataSetSize = dataSet.shape[0]                     #计算行数 n

    #计算距离
    diffMat = tile(inX, (dataSetSize,1)) - dataSet     #距离差 先把inX扩展，再求差矩阵
    sqDiffMat = diffMat**2                             #所有元素平方 [n,2]的array
    sqDistances = sqDiffMat.sum(axis=1)             #求每行的和[n,1]
    distances = sqDistances**0.5                    #所有元素开根号

    #选择距离最小的k个点
    sortedDistIndicies = distances.argsort()        #距离远近排序,[1,n]的array,第k行代表distances中第k小的索引
    classCount = {}                                    #建立一个字典
    for i in range (k):
        voteIlabel = labels[sortedDistIndicies[i]]    #获取离inX第i近元素的标签 sortedDistIndicies[i] 第i小的索引值
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1 
                                                    #改变字典中元素 voteIlabel是字典中的键key
                                                    #get(key, default=None)：返回指定键的值，如果值不在字典中返回default值
    
    #排序
    sortedClassCount = sorted(classCount.iteritems(),key=operator.itemgetter(1),reverse=True) 
                                                    #按照classCount的值的大小，逆排序(大到小)
    return sortedClassCount[0][0]                     #返回第一名的第一个元素（labels标签）

group, labels = createDataSet()
####################################################
# 2.2  约会匹配
def file2matrix(filename):                          #将文件内容转化为矩阵
    fr = open(filename)                             #打开、读入文件
    arrayOLines = fr.readlines()                    #自动将文件内容分析成一个行的列表
    numberOfLines = len(arrayOLines)                #获取文件行数
    returnMat = zeros((numberOfLines,3))            #创建一个行数和读取数据的行数一样，列数为3的元素全为0的矩阵
    classLabelVector = []                           #创建列表

    #解析文件数据到列表
    index = 0
    for line in arrayOLines:
        line = line.strip()                         #移除字符串头尾指定的字符（默认为空格）
        listFromLine = line.split('\t')             #以\t为分隔符，把读入的数据分成列数为4的矩阵
        returnMat[index,:] = listFromLine[0:3]      #把数据放入新构建的矩阵中，导入0,1,2列，没有3（0开始，3之前，不包含3）
        classLabelVector.append(int(listFromLine[-1])) 
                                                    #负数表示从后往前取，取最后一个元素，并且注明其为int（默认为string）
        index += 1
    return returnMat,classLabelVector                

def showDatingData(i,j):                            #根据数据绘制图
    fig = plt.figure()
    ax = fig.add_subplot(111)
    datingDataMat,datingLabels=file2matrix('datingTestSet2.txt')
    ax.scatter(datingDataMat[:,i],datingDataMat[:,j],15.0*array(datingLabels),15.0*array(datingLabels))
    plt.show()

def autoNorm(dataSet):                                #数据归一
    minVals = dataSet.min(0)                        #0代表选择当前列最小，输入m*n矩阵，返回1*n矩阵
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))                #构建一个和dataSet同样大小的矩阵
    m = dataSet.shape[0]                            #获得行数，也可以normDataSet.shape[0]
    normDataSet = dataSet - tile(minVals,(m,1))        #tile 一个构造矩阵的函数
    normDataSet = normDataSet/tile(ranges,(m,1))
    return normDataSet, ranges, minVals

def datingClassTest():                                #显示算法测试错误率
    hoRatio = 0.10                                    #抽取数据在总数据中的比例
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
                                                    #获取数据
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int (m*hoRatio)                    #测试数据的个数
    errorCount = 0.0
    for i in range(numTestVecs):                    #括号内是数字，无括号，则为array
        classifierResult = classify0(normMat[i,:],normMat[numTestVecs:m,:],datingLabels[numTestVecs:m],3)
                                                    #数据中[numTestVecs:m,:]，不包含测试集
        print "the classifier came back with: %d, the real answer is: %d" % (classifierResult, datingLabels[i])
        if (classifierResult != datingLabels[i]): errorCount += 1.0
    print "the total error rate is: %f" % (errorCount/float(numTestVecs))

def classifyPerson():
    resultList = ['not at all', 'in small doses', 'in large doses']
    percentTats = float(raw_input("percentage of time spent playing video games? "))
    ffMiles = float(raw_input("frequent flier miles earned per year? "))
    iceCream = float(raw_input("liters of ice cream consumed per year? "))
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    inArr = array([ffMiles, percentTats, iceCream])
    classifierResult = classify0((inArr-minVals)/ranges,normMat,datingLabels,3)
                                                    #输入归一化(inArr-minVals)/ranges 很重要，minCals ranges由已由大量数据得出
    print "You will probably like this person: ", resultList[classifierResult - 1]


datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
normMat, ranges, minVals = autoNorm(datingDataMat)

####################################################
# 2.3 手写识别
def img2vector(filename):                            #将文件内容转换为矩阵
    returnVect = zeros((1,1024))                    #是array，不是list，所以是[0,32*i+j]而不是[32*i+j]
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0,32*i+j] = int(lineStr[j])
    return returnVect

def handwritingClassTest():
    hwLabels = []                                    #标签集
    trainingFileList = listdir('digits/trainingDigits') 
                                                    #建立一个trainingFileList的List，list中每个元素是文件中文件名，是字符串
    m = len(trainingFileList)                        #len 获取对象的长度
    trainingMat = zeros((m,1024))                    
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]        #split,将字符串以"."分开，结果为一个list。[0]表示第一个"."以前的内容
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i,:] = img2vector('digits/trainingDigits/%s' % fileNameStr)

    testFileList = listdir('digits/testDigits')
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('digits/testDigits/%s' % fileNameStr)
        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
        print "the classifier came back with: %d, the real answer is: %d" % (classifierResult, classNumStr)
        if(classifierResult != classNumStr): errorCount += 1.0
    print "\nthe total number of errors is: %d" % errorCount
    print "\nthe total error rate is: %f" % (errorCount/float(mTest))
                                            #分母不转化为float也可以，分子分母只要有一个为float，结果为float。int/int结果为int
