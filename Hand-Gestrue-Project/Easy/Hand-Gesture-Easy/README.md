*文件说明 

* 函数部分 


getImageData.m 通过文件名识别训练集和测试集并读取文件

randomData.m 对训练集随机采样进行处理

showImageData.m 展示某个图像集的全体图像

showImageEdge.m 功能同上

edgeGetCanny.m 同Canny算法提取图像的边缘信息

bioGetOtsu.m 使用高斯滤波和顶帽变换进行图像预处理

obtainFeature 使用Hog算法提取图像的特征

classifyHog.m 对测试集进行分类

showAccuracy.m 展示预测结果

train.m 集成训练中的数据读取、图像预处理、特征提取和分类任务

---

* 文件部分 

test.m 模拟测试

parameterChosen.m 设计程序选择合适的参数

---

函数调用方法