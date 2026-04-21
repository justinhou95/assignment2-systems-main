import torch
import torch.nn as nn


class ToyModel(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 10, bias=False)
        self.ln = nn.LayerNorm(10)
        self.fc2 = nn.Linear(10, out_features, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.ln(x)
        x = self.fc2(x)
        return x


def check_dtypes():
    intermediates = {}

    def make_hook(name):
        def hook(module, input, output):
            intermediates[name] = output

        return hook

    model = ToyModel(in_features=4, out_features=3).cuda()  # params stored as FP32

    model.fc1.register_forward_hook(make_hook("fc1_out"))
    model.ln.register_forward_hook(make_hook("ln_out"))

    x = torch.randn(8, 4, device="cuda")
    target = torch.randint(0, 3, (8,), device="cuda")
    loss_fn = nn.CrossEntropyLoss()

    with torch.autocast(device_type="cuda", dtype=torch.float16):
        logits = model(x)
        loss = loss_fn(logits, target)

    loss.backward()

    print("=== Parameter dtypes (autocast does NOT change stored weights) ===")
    for name, param in model.named_parameters():
        print(f"  {name}: {param.dtype}")

    print("\n=== Activation dtypes (inside autocast context) ===")
    print(
        f"  fc1 output  (Linear -> autocasted):        {intermediates['fc1_out'].dtype}"
    )
    print(
        f"  LayerNorm output (precision-sensitive):    {intermediates['ln_out'].dtype}"
    )
    print(f"  logits / fc2 output (Linear -> autocasted): {logits.dtype}")

    print("\n=== Loss dtype ===")
    print(f"  loss: {loss.dtype}")

    print("\n=== Gradient dtypes (.grad matches param dtype) ===")
    for name, param in model.named_parameters():
        print(f"  {name}.grad: {param.grad.dtype}")


if __name__ == "__main__":
    check_dtypes()

    s = torch.tensor(0, dtype=torch.float32)
    for i in range(1000):
        s += torch.tensor(0.01, dtype=torch.float32)
    print(s)
    s = torch.tensor(0, dtype=torch.float16)
    for i in range(1000):
        s += torch.tensor(0.01, dtype=torch.float16)
    print(s)
    s = torch.tensor(0, dtype=torch.float32)
    for i in range(1000):
        s += torch.tensor(0.01, dtype=torch.float16)
    print(s)
    s = torch.tensor(0, dtype=torch.float32)
    for i in range(1000):
        x = torch.tensor(0.01, dtype=torch.float16)
        s += x.type(torch.float32)
    print(s)
