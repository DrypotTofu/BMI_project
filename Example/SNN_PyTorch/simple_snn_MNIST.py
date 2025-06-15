import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time

# ----------------- 替代梯度函数（ATan） -----------------
class SurrogateSpike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, threshold=1.0):
        ctx.save_for_backward(input)
        ctx.threshold = threshold
        return (input >= threshold).float()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        threshold = ctx.threshold
        scale = 2.0  # 可调节平滑程度
        grad_input = grad_output.clone()
        # 反正切函数的导数：scale / (π * (scale^2 * (input-threshold)^2 + 1))
        surrogate_grad = scale / (3.1415926 * (scale**2 * (input - threshold)**2 + 1))
        return grad_input * surrogate_grad, None

# ----------------- LIF基础神经元 -----------------

class LIFNeuron(nn.Module):
    """
    简化版 LIF 神经元模型，支持替代梯度。
    """
    def __init__(self, beta=0.9, threshold=1.0, reset_value=0.0):
        super().__init__()
        self.beta = beta
        self.threshold = threshold
        self.reset_value = reset_value

    def forward(self, input_current, mem=None):
        # mem为上一时刻的膜电位，若无则初始化
        if mem is None:
            mem = torch.full_like(input_current, self.reset_value)
        mem = self.beta * mem + input_current
        spike = SurrogateSpike.apply(mem, self.threshold)
        mem = torch.where(spike.bool(), mem - self.threshold, mem)
        return spike, mem

# ----------------- SNN三层全连接网络及训练/测试流程 -----------------

class SNN_MNIST(nn.Module):
    def __init__(self, beta=0.9, threshold=1.0, time_steps=25):
        super().__init__()
        self.fc1 = nn.Linear(28*28, 1000)
        self.lif1 = LIFNeuron(beta=beta, threshold=threshold)
        self.fc2 = nn.Linear(1000, 10)
        self.lif2 = LIFNeuron(beta=beta, threshold=threshold)
        self.time_steps = time_steps

    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        mem1 = mem2 = None
        out_spk = torch.zeros(batch_size, 10, device=x.device)
        for t in range(self.time_steps):
            cur1 = self.fc1(x)
            spk1, mem1 = self.lif1(cur1, mem1)
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            out_spk += spk2
        return out_spk / self.time_steps

# 数据加载
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0,), (1,))
])
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
# 提高速度：增大batch_size，开启pin_memory和num_workers
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, drop_last=True, num_workers=2, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False, num_workers=2, pin_memory=True)

def train(model, device, train_loader, optimizer, criterion, epoch, loss_hist):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)  # 加速数据搬运
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        loss_hist.append(loss.item())
        if batch_idx % 100 == 0:
            print(f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}] Loss: {loss.item():.6f}")

def test(model, device, test_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
    test_loss /= len(test_loader)
    acc = 100. * correct / total
    print(f"\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{total} ({acc:.2f}%)\n")
    return acc

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SNN_MNIST().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    loss_hist = []

    os.makedirs("figure", exist_ok=True)
    os.makedirs("eval_result", exist_ok=True)

    # 训练计时
    train_start = time.time()
    for epoch in range(1, 6):
        train(model, device, train_loader, optimizer, criterion, epoch, loss_hist)
    train_end = time.time()
    train_time = train_end - train_start

    # 评估计时
    eval_start = time.time()
    acc = test(model, device, test_loader, criterion)
    eval_end = time.time()
    eval_time = eval_end - eval_start

    # 绘制并保存loss曲线（使用seaborn，图像更大，线更细）
    plt.figure(figsize=(12, 6))
    sns.set(style="darkgrid")
    sns.lineplot(x=range(len(loss_hist)), y=loss_hist, linewidth=1)
    plt.title("Training Loss Curve")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.savefig("figure/simple_snn_MNIST_Loss.png")
    plt.close()

    # 输出评估结果到Markdown文件
    model_name = "simple_snn_MNIST"
    result_md = f"""# simple_snn_MNIST 评估结果

- **模型名称**: {model_name}
- **训练时间**: {train_time:.2f} 秒
- **评估时间**: {eval_time:.2f} 秒
- **准确率**: {acc:.2f}%
- **损失曲线文件**: figure/simple_snn_MNIST_Loss.png

"""
    with open("eval_result/simple_snn_MNIST.txt", "w", encoding="utf-8") as f:
        f.write(result_md)

if __name__ == "__main__":
    main()