import torch
import torch.nn as nn
import torch.optim as optim

from generate_data import generate_inputs, generate_labels
from split_data import split_data
from model import SimpleMLP

# 1. Generate data
X = generate_inputs()
yA, yB_high, yB_medium, yB_low = generate_labels(X)

# 2. Split data
data = split_data(
    X,
    yA,
    yB_high,
    yB_medium,
    yB_low
)

# 3. Take Task A data
X_train = data["X_train"]
X_val = data["X_val"]

y_train = data["yA_train"]
y_val = data["yA_val"]

# 4. 转换成Tensor
X_train = torch.tensor(X_train, dtype=torch.float32)
X_val = torch.tensor(X_val, dtype=torch.float32)

y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)  # 增加一个维度，把数据和模型输出的结构保持一致
y_val = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)

# 5. Create model
model = SimpleMLP()

# 6. Define loss function
criterion = nn.BCEWithLogitsLoss()

# 7. Define optimizer
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 8. Training loop
num_epochs = 100

for epoch in range(num_epochs):

    # Forward pass
    logits = model(X_train)

    # Calculate loss
    loss = criterion(logits, y_train)

    # Clear old gradients
    optimizer.zero_grad()

    # Backpropagation
    loss.backward()

    # Update model parameters
    optimizer.step()

    # Print training progress
    if (epoch + 1) % 10 == 0:
        print(
            f"Epoch {epoch + 1:3d}/{num_epochs}, "
            f"Loss: {loss.item():.4f}"
        )