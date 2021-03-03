<<<<<<< HEAD
# semeval 思考

## TODO

1. `./run_enhanced.sh` 增强`Albert`在`task2`上的表现。 ---- 无显著提高，基本处于80~82准确率。
2. ~~实现sliding window~~ 训练后的eval 环节如何处理，是多个sliding_window 投票？ 直接进行投票测试
3. 错误样本 分布是否是长文本更容易错？ **反而是短文本更容易错**。

----

1. https://www.aclweb.org/anthology/2020.acl-main.603.pdf 这篇会不会有奇效？使用RNN + BERT，但是文中分析的错误分布和当前模型**不一致**，并不是越长的文章错误率越高。
2. BART模型baseline测试

## 试验

albert 低学习率在task-2表现良好。86准确率。

----

~~测试text_a,text_b 的形式在各种模型上的效果，是否一致？ -》基本一样。~~

2020.11.28 

albert task1 enhanced

albert task2 enhanced 

---

2020.11.30

1. 争取实现 sliding window eval手段，使用投票方法
2. ~~分析错误样本长度分布，是否是更长的问题更难回答。~~ 分析反而是长的答对得多
3. 实现减少显存占用的方法， 用时间换空间。

## bugs

~~额外标签在`roberta-large`上dev测试分数78，但是一到semeval没有enhanced就低了很多，不知道为什么。~~ 重新tokenize了样本一下，就好了。。

[question, article] 训练好的模型在[article, question] 下的效果很差，准确率只有原来的一半。

## 尝试

### sliding window

将`sliding window`加入代码中，（但是可能会长文章产生更多的样本，是否会有问题？ eval的时候如何操作？） 对于每一个[question + article] 分隔成多个样本输入到模型进行训练。

~~11.24 准备先跑个bert  试试效果~~。 `doc_stride=100 Albert` 结果准确率82，可能超参还要调试。

### 负样本增强

`roberta-large`效果不错，但`albert`模型上效果好像不太理想，仍然还在尝试。

## 思路

1. 使用<mask>预测的东西，成为负样本，增大原样本个数。

   实现方案：在`train.jsonl`中加入`option_5`作为额外标签，不需要修改。

2. 使用concept net或者concept graph 针对数据生成一个预训练任务，参考[Align, Mask and Select: A Simple Method for Incorporating Commonsense Knowledge into Language Representation Models](https://arxiv.org/pdf/1908.06725.pdf)。 针对抽象词汇建立额外数据集。https://github.com/jessionlin/csqa

3. 使用<cls> + <token> 的值作为预测值。

4. 输入 选项输出embedding， 输入 上下文 + <mask> 的embedding，之后利用这两者对比进行模型训练。(实现中)。

5. 继续针对相关领域的数据进行预训练。


## 数据分析

### 数据构造方式

https://zhuanlan.zhihu.com/p/21343662 语料的构建方法有可能是这样子的。先使用CNN文章片段，之后针对文章生成一个摘要，再挖出**抽象词汇**。类似于摘要，基本没有多跳推理。

### task-1

![image-20201109193456487](http://typoracheasim.test.upcdn.net/20201109193503.png)

87%可以不用截取512长度。

#### top10 选项

[('major', 286), ('national', 252), ('special', 222), ('controversial', 205), ('lost', 130), ('annual', 127), ('latest', 124), ('free', 121), ('rare', 119), ('popular', 108)]

[('major', 62), ('special', 60), ('national', 57), ('controversial', 51), ('lost', 43), ('popular', 32), ('annual', 30), ('free', 27), ('fresh', 27), ('rare', 26)]

-------------

### task-2

![image-20201109193534140](http://typoracheasim.test.upcdn.net/20201109193534.png)

40% 可以不用截取。

#### top10 选项

[('group', 158), ('held', 122), ('body', 119), ('team', 109), ('criticised', 103), ('lost', 93), ('released', 89), ('set', 84), ('launched', 82), ('announced', 77)]

[('group', 40), ('lost', 34), ('body', 34), ('team', 32), ('criticised', 27), ('held', 26), ('side', 24), ('show', 22), ('life', 21), ('launched', 21)]

--------------

### task-3

![image-20201109202145278](http://typoracheasim.test.upcdn.net/20201109202145.png)

task-1,task-2 分布差不多。 task-3 交叉数据集不用截取，较为简单。

#### top10选项

[('major', 96), ('controversial', 62), ('confirmed', 62), ('latest', 56), ('rare', 49), ('special', 46), ('lost', 41), ('famous', 40), ('annual', 35), ('national', 35)]

[('group', 98), ('body', 98), ('team', 68), ('head', 59), ('deal', 54), ('side', 52), ('life', 44), ('centre', 40), ('region', 38), ('state', 35)]

[('future', 90), ('control', 73), ('health', 53), ('power', 51), ('conditions', 46), ('role', 46), ('status', 45), ('value', 43), ('criticism', 42), ('service', 41)]



------

### 实验过程

#### 分词输入模型中

由于roberta等模型可以处理的tokens数量最多为512，所以截取article前400个词和question拼成一个输入。输入为[article + question] 其中question中的@placeholder 替换成为选项中的词语。

还有使用question + article ,截取正好512tokens，或者取article中间部分？



-----

### 实验结果

Roberta-Large使用24层transformer encoder，在将[question, article] 拼接起来作为模型的输入，之后在`[cls]`输出拼接一个线性分类器，输出一个匹配的分数，对于五个`option`分别输入到模型中，最后经过`softmax`变换之后和原来的`label`计算交叉熵损失，进行反向传播作为训练。

所有模型均使用[text_b, text_a]，先问题，后文章的形式。类似于推理？

#### task-1 

类似于culture 等词汇的预测。

| dataset | Roberta-large | Albert-xxlarge | Roberta-large enhanced |
| ------- | ------------- | -------------- | ---------------------- |
| dev     | 75%           | 83%            | 78%                    |

直接利用模型去预测[mask] Train  52.4%  Dev  56.8%

task1在各种长度上的错误率大致相同，为啥？ 并且在更长的片段中准确率还**提高**了！

#### task-2

类似于涵盖内容更多 无脊椎动物的词汇。 问题更倾向于针对article的总结。 article 陈述一段事实，question 将上述的事实部分用一段语言表示，之后挖空了比较抽象的词汇。

| dataset | Roberta-large | Albert-xxlarge | Albert-xxlarge enhanced | roberta-large | Albert                                |
| ------- | ------------- | -------------- | ----------------------- | ------------- | ------------------------------------- |
| dev     | 83%           | 86%            | 82.8%                   | 84.25         | 86.8 `./saved_model_file/albet_task2` |

#### task-3

总和上述两种抽象词汇的混合数据集。探究在task-1训练之后在task-2上的表现，等等。

| x         | albert   train on task1 | roberta-large on task2 | albert   train on task2 | Roberta-large on task 1 |
| --------- | ----------------------- | ---------------------- | ----------------------- | ----------------------- |
| Dev-task1 | 85.4%                   | 84.8%                  | 74.7%                   | 53.6%                   |
| Dev-task2 | 89%                     | 87.9%                  | 76.8%                   | 51.8%                   |
| dev-task3 | 92.6%                   | 92.2%                  | 80.8%                   | 60%                     |



----

###  错误分析

对于roberta模型，可能是 lower the morale 更加通顺， adversely affected 这个词 抽象出来的意思。

{"article": "Media playback is unsupported on your device 7 May 2013 Last updated at 18:47 BST It is a particular problem in more affluent countries, with sleep experts linking it to the use of mobile phones and computers in bedrooms late at night. It is such a serious disruption that lessons have to be pitched at a lower level to accommodate the sleep-starved learners, the study found. The international comparison, carried out by Boston College, found the United States to have the highest number of sleep deprived students, with 73% of 9 and 10 year olds and 80% of 13 and 14 year olds identified by their teachers as being **adversely affected**. The BBC's Jane O'Brien reports.",

 "question": "Sleep deprivation is a significant hidden factor in lowering the @placeholder of school pupils , according to researchers carrying out international education tests .", 

"option_0": "morale", 

"option_1": "iq", 

"option_2": "mortality", 

"option_3": "closure", 

"option_4": "achievement",

 "label": 4, "wrong_label": 0}



模型对于上下文的理解还不够， 还是更倾向于只看了question，就选出了答案。

{"article": "East Street, Hammet Street and St James' Street in Taunton will be closed to cars during the day in the 18-month trial due to start in the autumn. Buses will still use East Street. Roger Habgood from Taunton Deane Borough Council said: \"We're trying to improve the town for all of us.\" But critics said it was \"stupid\" and would increase congestion. The plans, which were posted on BBC Somerset's Facebook page, attracted nearly 300 comments with some welcoming the move but the majority fearing it would cause more congestion. Sheila Jordan wrote: \"This is an absolutely stupid idea, how do the council think this will help the every day pressure of the roads and constant queues at rush hour, people are already fed up to the back teeth that nothing is being done to help this.\" Another tweet from Laura Webber said: \"They're trying to pedestrianise the whole of Taunton town centre except for buses. As if traffic wasn't bad enough already.\" Mr Habgood said: \"We're pretty sure it will be popular because it was when East Street was closed for other reasons and we want to make it a friendlier place to be. \"It does fit with our larger objectives to improve the town and **make it safer for cyclists and pedestrians**.\" Colin Barrell, president of Taunton Chamber of Commerce said he was concerned disabled people would be unable to park close to shops. He also said retailers would be concerned about a loss of footfall. The trial will only begin once the Northern Inner Distribution Route (NIDR) project is completed, the council said.", 

"question": "Three busy town centre streets are to be pedestrianised in a bid to improve @placeholder for shoppers and cyclists .",

 "option_0": "opportunities", 

"option_1": "services", 

"option_2": "quality",

 "option_3": "disruption", 

"option_4": "safety", 

"label": 4, "wrong_label": 1}

对于在词汇底下的语义理解还不够透彻，或者说"I"和"Aled Sion Davies"联系不够紧密。

{"article": "Briton Davies won F42 shot put gold with a Games record at Rio 2016, but was unable to defend his 2012 discus title as it did not feature in Brazil. \"I don't normally say what I'm going for,\" said the Welshman, 25. \"But **this time I'm definitely going for the two golds **in both disciplines and nothing will be better than being in front of a home crowd.\" Davies is set to resume training after taking a post-Rio break. \"As much as I love time off I love what I do and focus has already shifted to London,\" he said. \"Rio was only a stepping stone towards that and London is going to be a huge event, back in the Olympic Stadium.\" Davies won F42 discus and shot put golds at the past two IPC World Athletics Championships - in Lyon in 2013 and in Doha two years later.", 

"question": "Paralympic champion Aled Sion Davies @placeholder two gold medals at the 2017 World Para Athletics Championships in London .",

 "option_0": "suffered",

 "option_1": "promoted",

 "option_2": "remains", 

"option_3": "wants", 

"option_4": "achieved", 

"label": 3, "wrong_label": 4}



这是一个很特别的错误，每一个错误都试验过作为[MASK]查看模型输出，这个在未经过训练的模型以MLM输入是可以输出正确的答案，但是经过训练后变成错误的了。

{"article": "Shaun Duffy, 28, from the Partick area, admitted seriously assaulting Simon Ross and attacking Jason Martin outside the Record Factory in Byres Road. During the attack, on 4 February, Duffy punched Mr Martin unconscious and kicked and stamped on his head When police traced Duffy his clothing was stained with the victims' blood. Jailing him at the High Court in Glasgow, judge Lord Boyd told Duffy: \"You assaulted Mr Ross by punching him rendering him unconscious and then assaulted Mr Martin. \"You then returned and continued your assault on Mr Ross by kicking and stamping on his head, that was particularly dangerous. \"There was banter, but it is not suggested any comments were inappropriate.\" The court heard that Mr Ross suffered a facial fracture, cuts to the back of his head, a broken nose and his right eye was swollen and closed. Mr Martin had bruising above his right eye and a lump on the back of his head.", 

"question": "A man who **@placeholder** being drunk on cocktails for assaulting two people outside a pub in Glasgow has been jailed for five years and three months .",

 "option_0": "denied", "option_1": "lost", "option_2": "blamed", "option_3": "underwent", "option_4": "reported", "

label": 2, "wrong_label": 0, 

"logits": [0.9999809265136719, 4.12926301387361e-29, 1.9056920791626908e-05, 1.2005882399863533e-30, 1.9256362129849158e-11]}



{"article": "Southend United striker Nile Ranger, 25, has been charged with conspiracy to defraud and conspiracy to commit money laundering, police said. The offences are alleged to have been committed in February 2015. Mr Ranger appeared alongside two other defendants at Highbury Corner Magistrates' Court earlier. Aseany Duncan, 19, was charged with possessing the personal bank details of 500 people on his phone for the use of fraud. He was also charged with conspiracy to defraud and conspiracy to commit money laundering alongside Mr Ranger and Reanne Morgan, 18. The case will next be heard at Wood Green Crown Court in the New Year. Southend United have said Mr Ranger will continue to be available for selection at the League One club. Read more on this story and other Essex news Mr Ranger, of Bounds Green, north London, signed a new three-and-a-half year contract with Southend at the start of this month, having started his career at Newcastle United before moving to Swindon Town and then Blackpool. He joined Southend in the summer after a trial, having not played a first-team game since November 2014 while at Blackpool.", 

"question": "A footballer has been charged over an alleged scam involving a vulnerable person 's bank details being @placeholder so their savings could be accessed .", 

"option_0": "punched", 

"option_1": "threw", 

"option_2": "taken",

 "option_3": "manipulated", 

"option_4": "denied", 

"label": 2, "wrong_label": 3, 

"logits": [0.07021217048168182, 1.1431486775925889e-15, 1.0296787877450697e-05, 0.9297775030136108, 4.259185506453461e-13]}



{"article": "Henrhyd Falls in the Brecon Beacons National Park was used as the cover for the Batcave in the film The Dark Knight Rises. Fans previously had to clamber down a steep tree-lined hill and wade across a river to reach the falls. Park chiefs have now built a path to the waterfall for easier access. The 88ft (27m) wall of water was seen in the final film of the Christopher Nolan trilogy as the location where Robin discovers the Batcave after the apparent death of Batman. The new trail and footbridge bring visitors out at the foot of the falls, and they can then walk behind the water and into the entrance of the Batcave. Henrhyd Falls is the tallest waterfall in south Wales and is a site of special scientific interest due to the rare mosses, liverworts and lichen which grow in the damp, heavily wooded gorge, with its thin soils and steep rocky slopes. Batman fan Jackie Davies, 24, from Bridgend, said: \"I love Christian Bale in the Batman films and coming here you can feel like you're a part of that world. \"The waterfall is spectacular and I had to come with my costume for a picture. My boyfriend finds it embarrassing but I love it.\" Judith Harvey, wardens manager for Brecon Beacons National Park Authority, said the falls were a popular walking destination in the Brecon Beacons. \"With work now complete, it means that people for many generations to come can really enjoy what Abercraf and the surrounding area has to offer, safe in the knowledge that they are not causing any damage.\"", 

"question": "The waterfall used by Batman star Christian Bale to conceal the superhero 's secret @placeholder on screen can now be reached via a woodland trail .", 

"option_0": "hideout", "option_1": "powers", "option_2": "impact", "option_3": "cave", "option_4": "structure", "label": 3, "wrong_label": 0, 

"logits": [0.9993686079978943, 1.1315256457505862e-12, 1.055245853833195e-22, 0.0006313661579042673, 3.525976169024836e-13]}



{"article": "Images include prisoners taking part in a snowball fight during World War One and a tunnel which may have been dug as part of an escape attempt. The pictures have been shared by the great nephew of Captain Eli Bowers, who was among the camp's guards. They \"add to our understanding of the history of the camp\", said island historian Ian Ronayne. The site at Les Blanches Banques housed nearly 2,000 men from the German armed forces from 1915-1917. Dr Brian K Feltman, from Georgia Southern University in the United States, said the snowball fight showed prisoners \"breaking the monotony of camp life\". He said it could be an example of prisoners staving off \"barbed wire disease\", a form of depression associated with \"the boredom and regulations\" imposed on prisoners of war. Mr Ronayne, a WW1 blogger for Jersey Heritage, said some of the images showed tunnels which may have been used during one of the attempts to escape the camp. He said if the images related to one of these incidents it would be a \"fantastic find\". Another showed two sets of fencing at the camp - \"barbed wire on the inside... and then an electrified fence\" -  which was rare, according to Dr Heather Jones from the International History at the London School of Economics. \"I have not come across an electric fence being used for a UK home front prisoner of war camp before,\" she added. Capt Bowers served in the Royal Jersey Militia and took the six photos from late 1916 to early 1917. Helier Falle found the pictures in his mother's house. \"Rather than leave them sitting in a drawer for nobody to see, I thought they should be shared,\" he said. Though he had seen similar photos auctioned, he added \"it wouldn't feel right to profit from them\".", 

"question": "Never before seen photographs have been @placeholder showing life in Jersey 's prisoner of war camp .", 

"option_0": "discovered", "option_1": "targeted", "option_2": "released", "option_3": "praised", "option_4": "criticised", "label": 2, "wrong_label": 0, 

"logits": [0.880974531173706, 1.1364979363719133e-20, 0.11902539432048798, 4.938195878637977e-18, 1.4639556631888016e-17]}







-----

> 参考资料 
>
> https://mp.weixin.qq.com/s/DT0rc-RGnVT0jerX51x8nw



### 一些脑洞

placeholder 在对文档做一次attention

增强context question联系。

基于question trian一个模型，预测的差异作为额外的信息。

加强@placeholder 对于结果的权重。 替换前后的区别，去计算loss

孪生网络， 

模型保存，后续模型融合 **stacking essemble** https://zhuanlan.zhihu.com/p/32896968

融合context 上的知识。 找已有思路，试一试。pretrain 或者 Fine-tune作为额外特征。

test data 融入，预训练。 自评

统计20个bad_cases 原因的比例。

~对抽象的词语进行一个知识库建模？





did you know wikipedia

针对任务问题分类。 bad cases 分类， 探究是否有reasoning 问题，单跳，多跳。

-----

针对数据集特点，专门构造self-supervised 进行继续训练。类似于span-bert 针对mprc 。



尝试更多的任务。

1. 对这些预料进行 mask entity+继续预训练+fine-tune
2. 更多的模型架构 孪生网络架构 few-shot learning 找最近领

=======
# SemEval2021Task4
>>>>>>> b2b634f127f9bfcdddca22ba6b9ec11ef9b8032f
