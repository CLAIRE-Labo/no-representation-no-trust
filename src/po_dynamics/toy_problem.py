import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

# Examples
# Rank 1, increase alpha makes things worse
# Also increase feature norm makes things worse
# seed=5, lr=1

seed = 1
torch.manual_seed(seed)
N_layers = 1


# Define the model
class PolicyNetwork(nn.Module):
    def __init__(self):
        super(PolicyNetwork, self).__init__()
        self.linear = nn.Linear(N_layers, 2, bias=False)  # Single state input, two actions output

    def forward(self, state):
        logits = self.linear(state)
        return nn.functional.softmax(logits, dim=-1)


# Initialize the model and optimizer
model = PolicyNetwork()
optimizer = optim.SGD(model.parameters(), lr=1.5)

# Example state x and alpha
feature_norm = 1
x = feature_norm * torch.randn((1, N_layers), dtype=torch.float)
y = feature_norm * torch.randn((1, N_layers), dtype=torch.float)

copy_n = N_layers
alpha = 3  # -1 for interference, 3 for boost
y[:, :copy_n] = alpha * x[:, :copy_n]

# Fixed advantage and old policy probabilities for demonstration
A = 1.0  # Advantage
old_pi_a1_x = model(x)[0, 0].item()
old_pi_a1_y = model(y)[0, 0].item()
epsilon = 0.1


def compute_loss(prob_a1, old_prob_a1):
    """assume A > 0 for simplicity"""
    ratio = prob_a1 / old_prob_a1
    clipped_ratio = torch.clamp(ratio, max=1 + epsilon)
    return A * torch.min(ratio, clipped_ratio)


# Training loop with alternating updates
ratio_x_history = [1]
ratio_y_history = [1]

steps = 20
for step in range(steps):
    # Alternate between x and y
    state = x if step % 2 == 0 else y
    old_pi_a1 = old_pi_a1_x if step % 2 == 0 else old_pi_a1_y

    # Forward pass
    probs = model(state)
    prob_a1 = probs[0, 0]

    # Compute loss
    loss = -compute_loss(prob_a1, old_pi_a1)

    # Backward pass and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Log probabilities for plotting
    with torch.no_grad():
        pi_x = model(x)[0, 0].item()
        pi_y = model(y)[0, 0].item()
        ratio_x = pi_x / old_pi_a1_x
        ratio_y = pi_y / old_pi_a1_y
        ratio_x_history.append(ratio_x)
        ratio_y_history.append(ratio_y)

        print(f"Step {step}: pi_x = {pi_x:.2f}, pi_y = {pi_y:.2f}, ratio_x = {ratio_x:.2f}, ratio_y = {ratio_y:.2f}")
# Plotting
# Set the y-axis between 0 and 2
# plt.ylim(0, 2)
plt.plot(ratio_x_history, label=r"$\pi_\theta(a_1| x) / \pi_\text{old}(a_1|x)$")
plt.plot(ratio_y_history, label=r"$\pi_\theta(a_1 | y) / \pi_\text{old}(a_1 | y)$")
plt.xlabel("Minibatch")
plt.ylabel("Prob ratio")
# plot the 1+epsilon line
plt.axhline(1 + epsilon, color="r", linestyle="--", label=r"1+$\epsilon$")
# x-axis as integer ticks
plt.xticks(range(0, steps + 1))
# show grid lines
plt.grid()

plt.legend()
plt.show()
