import numpy as np
import matplotlib.pyplot as plt
from active_learning import run_active_learning

# Settings
n_avg = 3
n_iterations = 10

common_kwargs = dict(
    data_dir="data/cifar-10-batches-py",
    n_initial_per_class=30,   # 30 per class = 300 total
    n_iterations=n_iterations,
    query_size=10,
    batch_size=64,
    epochs=3,
    lr=1e-3,
)

experiments = {
    "Least confident": {"paradigm": "active learning", "strategy": "least confident"},
    "Margin": {"paradigm": "active learning", "strategy": "margin"},
    "Entropy": {"paradigm": "active learning", "strategy": "entropy"},
    "Random": {"paradigm": "random"},
}

results = {}

for name, settings in experiments.items():
    print(f"\nRunning averaged experiment: {name}")

    all_scores = []

    for i in range(n_avg):
        print(f"  Run {i+1}/{n_avg}")

        scores, labeled_sizes, _, _ = run_active_learning(
            **common_kwargs,
            **settings,
            seed=i
        )

        all_scores.append(scores)

    all_scores = np.array(all_scores)

    results[name] = {
        "mean": all_scores.mean(axis=0),
        "std": all_scores.std(axis=0),
        "labeled_sizes": labeled_sizes
    }

plt.figure(figsize=(8, 5))

for name, res in results.items():
    mean = res["mean"]
    std = res["std"]
    x = res["labeled_sizes"]

    plt.plot(x, mean, label=name)
    plt.fill_between(x, mean - std, mean + std, alpha=0.2)

plt.xlabel("Number of labeled samples")
plt.ylabel("Test accuracy")
plt.title("Active Learning (averaged over runs)")
plt.legend()
plt.grid(True)
plt.show()