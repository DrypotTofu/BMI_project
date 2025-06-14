import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

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
    transforms.Normalize((0.1307,), (0.3081,))
])
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

def train(model, device, train_loader, optimizer, criterion, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print(f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}] Loss: {loss.item():.6f}")

def test(model, device, test_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
    test_loss /= len(test_loader)
    acc = 100. * correct / len(test_loader.dataset)
    print(f"\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({acc:.2f}%)\n")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SNN_MNIST().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(1, 6):
        train(model, device, train_loader, optimizer, criterion, epoch)
        test(model, device, test_loader, criterion)

if __name__ == "__main__":
    main()