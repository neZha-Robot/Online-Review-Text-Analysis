---
title: 电商在线评论分析

---

​	

**文本挖掘：从大量文本数据中抽取出有价值的知识，并且利用这些知识重新组织信息的过程。**

# 一、数据预处理

## 1.1缺失值处理

```python
df = df.dropna() #删除存在缺失值的行
```

## 1.2重复值处理

由于文本评论数据质量高低不一，无用的文本数据很多，所以文本去重就可以删掉许多的没意义的评论。

```python
dfdrop_duplicates(inplace=True)#删除重复记录
```

## 1.3机械压缩

经过文本去重后的评论仍然有很多评论需要处理，比如：“好好好好好好好好好好好”，这种存在连续重复的语句，也是比较常见的无意义文本。这一类语句是需要删除的，但计算机不能自动识别出所有这种类型的语句，若不处理，可能会影响评论情感倾向的判断。因此，需要对语料进行机械压缩去词处理，也就是说要去掉一些连续重复的表达，比如把：“不错不错不错”缩成“不错”。

```python
#定义机械压缩去重函数
def yasuo(st):
    for i in range(1,int(len(st)/2)+1):
        for j in range(len(st)):
            if st[j:j+i] == st[j+i:j+2*i]:
                k = j + i
                while st[k:k+i] == st[k+i:k+2*i] and k<len(st):   
                    k = k + i
                st = st[:j] + st[k:]    
    return st
yasuo(st="啊啊啊啊啊啊啊")

# 对comment列进行操作
df["comment"] = df["comment"].apply(yasuo)
```

## 1.4文本内容清理

文中的表达符号、特殊字符，通常对文本分析的作用不大，删除。删除文本中的指定字符用正则匹配的方式。

```python
pattern = re.compile(u'[^\u4e00-\u9fa5a-zA-Z0-9]+')
# 定义函数，用于删除非中文、英文和数字字符
def clean_text(text):
    return re.sub(pattern, '', text)

# 对comment列进行操作    
df['clean_comment'] = df['comment'].apply(clean_text)
```



# 二、中文分词

## 2.1概念

**中文分词（Chinese Word Segmentation）**：将一个汉字序列切分成一个一个单独的词。

eg：我的家乡是广东省湛江市-->我/的/家乡/是/广东省/湛江市

**停用词（Stop Words）：**
数据处理时，需要过滤掉某些字或词

## 2.2代码实现

```python
import pandas as pd
import jieba
import jieba.posseg as pseg
import os

# 加载停用词表
stopwords = set(pd.read_csv('/home/aistudio/dict/stopwords.txt', header=None, encoding='utf-8', squeeze=True))
# 加载自定义词典
jieba.load_userdict('/home/aistudio/dict/dict.txt')
# 定义需要替换的词和替换成的词
replace_dict = {
    '老师': '教师',
    '医生': '医务工作者',
    # 可以继续添加需要替换的词和替换成的词
}
# 定义选择词性的函数
def select_words(words, pos):
    result = []
    for word, tag in words:
        if tag in pos and word not in stopwords:
            # 替换词
            for k, v in replace_dict.items():
                word = word.replace(k, v)
            result.append(word)
    return ' '.join(result)
# 定义分词函数
def cut_comment(comment, pos):
    # 分词
    words = pseg.cut(comment)
    # 选择词性
    result = select_words(words, pos)
    # 返回结果
    return result
# 指定要读取的文件夹路径
folder_path = "data_clean"
folder_path1 = "data_cut"
# 获取文件夹中的所有xlsx文件
file_list = [f for f in os.listdir(folder_path) if f.endswith('.xlsx')]
# 循环读取每个xlsx文件，并对其中的comment列进行操作
for file_name in file_list:
    file_path = os.path.join(folder_path, file_name)
    # 读取Excel文件
    data = pd.read_excel(file_path)
    # 对clean_comment列应用分词函数
    df['cut_comment'] = df['clean_comment'].apply(lambda x: cut_comment(x, ['n', 'a', 'v', 'vn', 'an']))#保留名词 动词 形容词
    # 将结果保存到一个新的Excel文件中
    new_file_name = 'new_' + file_name
    new_file_path = os.path.join(folder_path1, new_file_name)
    data.to_excel(new_file_path, index=False)
```

# 3.TF-IDF

## 3.1**TF-IDF算法介绍**

​	**TF-IDF（term frequency–inverse document frequency，词频-逆向文件频率）**是一种用于信息检索（information retrieval）与文本挖掘（text mining）的常用**加权技术**。

​	TF-IDF是一种统计方法，用以评估一字词对于一个文件集或一个语料库中的其中一份文件的重要程度。**字词的重要性随着它在文件中出现的次数成正比增加，但同时会随着它在语料库中出现的频率成反比下降。**

​	TF-IDF是一种统计方法，用以评估一字词对于一个文件集或一个语料库中的其中一份文件的重要程度。**字词的重要性随着它在文件中出现的次数成正比增加，但同时会随着它在语料库中出现的频率成反比下降。**

## 3.2代码实现

```python
import pandas as pd
import jieba
import jieba.analyse
import os

# 读取数据
folder_path = "data_cut"
# 获取文件夹中的所有xlsx文件
file_list = [f for f in os.listdir(folder_path) if f.endswith('.xlsx')]
# 循环读取每个xlsx文件，并对其中的comment列进行操作
for file_name in file_list:
    file_path = os.path.join(folder_path, file_name)
    # 读取Excel文件
    data = pd.read_excel(file_path)
    data = data.dropna() #删除存在缺失值的行
    merged_df = data.append(data, ignore_index=True)
    # 输出分词结果
    text = ''.join(data['cut_comment'].tolist())
    # 对拼接后的文本进行关键词提取
    keywords = jieba.analyse.extract_tags(text, topK=20, withWeight=True, allowPOS=('n', 'a', 'v', 'vn', 'an'))
    print(file_name)
    print(keywords)
    
# 将合并后的 DataFrame 保存为新的 Excel 文件
merged_df.to_excel("merged.xlsx", index=False)
```

# 4.情感分析

​	对于评论“物流挺不错的。价格也还可以。”即可拆分为“物流挺不错的”和“价格也还可以”两条评论。利用扩充后的主题词典，通过属性词匹配的方式，找出包含主题词的评论分句，统计各评价维度对应的评论分句数量。假定一条分句中，仅包含评论者对评论实体某一主题的评价，将分句情感得分的结果转换为对评价主体属性的得分。

```python
import re
from snownlp import SnowNLP
import pandas as pd
# 主题词典
topic_dict = {"价格": ["便宜", "贵", "优惠", "划算"],
              "口感": ["好吃", "难吃", "美味"],
              "服务": ["态度", "热情", "周到"],
              "物流": ["物流", "快"]}#示例
# 提取评论中的句子
def extract_sentences(text):
    # 将文本分为句子
    sentences = re.split(r'[。！？]', text)
    # 去除空句子
    sentences = [sent.strip() for sent in sentences if len(sent.strip()) > 0]
    return sentences
# 匹配主题词
def match_topic(sentence, topic_dict):
    # 初始化主题词
    topic = None
    # 遍历主题词典
    for t, attributes in topic_dict.items():
        # 判断主题词是否在句子中
        if t in sentence:
            topic = t
            break
    return topic
# 读取评论数据
df = pd.read_excel("寺库.xlsx")
# 对主题词进行情感分析
for topic in topic_dict:
    topic_sentiments = []
    for comment in df["comment"]:
        # 分句
        sentences = extract_sentences(comment)
        # 遍历句子
        for sentence in sentences:
            # 匹配主题词
            if match_topic(sentence, topic_dict) == topic:
                # 进行情感分析
                s = SnowNLP(sentence)
                sentiment = s.sentiments
                topic_sentiments.append(sentiment)
    # 输出主题词情感分析结果
    if len(topic_sentiments) > 0:
        avg_sentiment = sum(topic_sentiments) / len(topic_sentiments)
        print(f"{topic}的情感倾向为：{avg_sentiment:.2f}")
```



**注：大三的时候第一次做这个项目，大四又做了一次，但因为没有这方面的基础，所以做的很粗糙，写的更加粗擦。之前参考其它博主的文章时，总是抱怨不够详细，直到自己写才知道这难度，不过好在给博客项目开了个头。**
