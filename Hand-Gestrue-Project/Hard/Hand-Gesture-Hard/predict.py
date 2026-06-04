# 作业类型: (3) 对全部 6 种手势进行识别
import cv2
from model import RecognitionNet
from torchvision.transforms import v2 # 图像初始化的操作

DATA_CATEGORY = ['A', 'B', 'C', 'Five', 'Point', 'V']
test_transform = v2.Compose([
    v2.Normalize(mean = [0.5, 0.5, 0.5], std = [0.5, 0.5, 0.5])])

import os
import torch


def load_model(model_folder="Model", model_date=None, model_version=None, prefix="Flip_Rotation_Normalize", p=0.8):
    """
    加载模型权重。

    参数:
        model_folder: 模型根目录
        model_date: 日期文件夹，如 "20260516"。为 None 时自动选最新日期
        model_version: 版本号，如 1。为 None 时自动选最新版本
        prefix: 模型文件名前缀
        p: 训练集比例

    返回:
        net: 加载好权重的模型
        info: 模型信息字符串
    """
    net = RecognitionNet()

    # 1. 自动选最新日期
    if model_date is None:
        dates = [d for d in os.listdir(model_folder) if os.path.isdir(os.path.join(model_folder, d))]
        if not dates:
            raise FileNotFoundError(f"在 {model_folder} 中没有找到日期文件夹")
        model_date = sorted(dates)[-1]  # 最新的日期

    date_path = os.path.join(model_folder, model_date)

    # 2. 自动选最新版本
    if model_version is None:
        pth_files = [f for f in os.listdir(date_path) if f.startswith(prefix) and f.endswith('.pth')]
        if not pth_files:
            raise FileNotFoundError(f"在 {date_path} 中没有找到 {prefix} 开头的模型文件")
        # 按版本号排序，取最大
        latest = sorted(pth_files)[-1]
        model_name = latest
    else:
        model_name = f"{prefix}_v{model_version}_p={p}.pth"

    model_path = os.path.join(date_path, model_name)

    # 3. 加载
    state_dict = torch.load(model_path, weights_only=True)
    net.load_state_dict(state_dict)
    net.eval()

    info = f"已加载: {model_date}/{model_name}"
    print(info)
    return net, info

def predict_the_train_data(statu):
    if not statu:
        return False # 是0就不运这个文件
    path = "Data"  # 自行更改 这里暂时设置了Data是为了检测效果
    for category in DATA_CATEGORY:
        path = os.path.join("Data", category)
        # os.makedirs(path, exist_ok = True)
        filelist = os.listdir(path)  # 寻找Test训练集中

        net.eval()  # 推理模式
        count = 0
        for image_name in filelist:
            image_path = os.path.join(path, image_name)
            image = cv2.imread(image_path)
            image = cv2.resize(image, (128, 128))
            image = image.astype("float32")
            image = torch.from_numpy(image)
            image = image / 255.0
            image = image.permute(2, 0, 1)
            # image = test_transform(image)
            input_image = image.unsqueeze(0)
            with torch.no_grad():
                output_image = net(input_image)
                probabilities = torch.softmax(output_image, dim=1)  # 转为概率
                confidence, result = torch.max(probabilities, dim=1)
                # print(f"{image_name}: {DATA_CATEGORY[result.item()]} (置信度: {confidence.item():.2%})")
            result = torch.max(output_image, dim=1)[1].item()
            # print(DATA_CATEGORY[result])
            if DATA_CATEGORY[result] == category:
                count += 1
        # print(f"category: {category}, count = {count}, total image = {len(filelist)}")
        print(f"category: {category} accuracy: {count / len(filelist) * 100:.4f}%")
    return True

def predict_the_test_data():
    path = "Test"
    # os.makedirs(path, exist_ok = True) 最初创建训练集需要新建文件夹
    filelist = os.listdir(path)  # 寻找Test训练集中
    count_tot = 0
    count_a = 0
    net.eval()  # 推理模式
    for image_name in filelist:
        image_path = os.path.join(path, image_name)
        image = cv2.imread(image_path)
        image = cv2.resize(image, (128, 128))
        image = image.astype("float32")
        image = torch.from_numpy(image)
        image = image / 255.0
        image = image.permute(2, 0, 1)
        # image = test_transform(image)
        input_image = image.unsqueeze(0)
        with torch.no_grad():
            output_image = net(input_image)
            probabilities = torch.softmax(output_image, dim=1)  # 转为概率
            confidence, result = torch.max(probabilities, dim=1)
            print(f"{image_name}: {DATA_CATEGORY[result.item()]} (置信度: {confidence.item():.2%})")
        result = torch.max(output_image, dim=1)[1].item()
        if DATA_CATEGORY[result] == 'A':
            count_a += 1
        count_tot += 1
    print(f"total {count_tot}, a {count_a}")


# ====== 使用方式 ======
# 方式1：自动加载最新模型
# net, info = load_model()

# 方式2：指定日期和版本
net, info = load_model(model_date="20260604", model_version=2, prefix="No_Flip_Rotation")

# 方式3：只指定日期，自动选最新版本
# net, info = load_model(model_date="20260516")

# 方式4：不同 prefix（不同增强方案对比）
# net, info = load_model(prefix="Flip_Rotation_Normalize")
# net, info = load_model(prefix="recognition_net")  # 旧的

# predict_the_train_data(0) # 检查模型用训练集的效果 0表示不运行

predict_the_test_data() # 检查模型测试集的训练效果