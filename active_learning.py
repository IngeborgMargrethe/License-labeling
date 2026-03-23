import numpy as np
from torch.utils.data import DataLoader

from data import prepare_data, CIFARDataset
from model import CNN, train_model, evaluate_model, predict_probabilities


def update_labeled_pool(X_labeled, y_labeled, X_pool, y_pool, selected_indices):
    """
    Move selected samples from pool to labeled set.
    """
    selected_indices = np.array(selected_indices)

    X_new = X_pool[selected_indices]
    y_new = y_pool[selected_indices]

    X_labeled = np.concatenate([X_labeled, X_new], axis=0)
    y_labeled = np.concatenate([y_labeled, y_new], axis=0)

    mask = np.ones(len(X_pool), dtype=bool)
    mask[selected_indices] = False

    X_pool = X_pool[mask]
    y_pool = y_pool[mask]

    return X_labeled, y_labeled, X_pool, y_pool


def evaluate_uncertainty(prob, strategy):
    """
    Compute uncertainty scores from predictive probabilities.
    """
    if strategy == "least confident":
        return 1 - np.max(prob, axis=1)

    elif strategy == "margin":
        sorted_prob = np.sort(prob, axis=1)
        p1 = sorted_prob[:, -1]
        p2 = sorted_prob[:, -2]
        return 1 - (p1 - p2)

    elif strategy == "entropy":
        prob = np.clip(prob, 1e-12, 1.0)
        return -np.sum(prob * np.log(prob), axis=1)

    else:
        raise ValueError(f"Unknown strategy: {strategy}")


def make_dataloaders(
    X_labeled, y_labeled,
    X_pool, y_pool,
    X_test, y_test,
    batch_size=64,
):
    train_dataset = CIFARDataset(X_labeled, y_labeled)
    pool_dataset = CIFARDataset(X_pool, y_pool)
    test_dataset = CIFARDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    pool_loader = DataLoader(pool_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, pool_loader, test_loader


def run_active_learning(
    data_dir="data/cifar-10-batches-py",
    n_initial_per_class=10,
    n_iterations=10,
    query_size=10,
    strategy="entropy",
    paradigm="active learning",
    batch_size=64,
    epochs=5,
    lr=1e-3,
    seed=42,
):
    """
    Run active learning on CIFAR-10 with a CNN.

    Returns
    -------
    scores : list
        Test accuracies after each iteration
    labeled_sizes : list
        Labeled set sizes after each iteration
    best_model :
        Best-performing model across iterations
    best_acc : float
        Best test accuracy observed
    """
    data = prepare_data(
        data_dir=data_dir,
        n_initial_per_class=n_initial_per_class,
        seed=seed,
    )

    X_labeled = data.X_initial.copy()
    y_labeled = data.y_initial.copy()
    X_pool = data.X_pool.copy()
    y_pool = data.y_pool.copy()
    X_test = data.X_test
    y_test = data.y_test

    scores = []
    labeled_sizes = []

    rng = np.random.default_rng(seed)

    best_model = None
    best_acc = 0

    for iteration in range(n_iterations):
        print(f"\n=== Iteration {iteration + 1}/{n_iterations} ===")
        print(f"Labeled set size: {len(X_labeled)}")
        print(f"Pool size:        {len(X_pool)}")

        train_loader, pool_loader, test_loader = make_dataloaders(
            X_labeled, y_labeled,
            X_pool, y_pool,
            X_test, y_test,
            batch_size=batch_size,
        )

        model = CNN(num_classes=10)
        model = train_model(model, train_loader, epochs=epochs, lr=lr)

        test_acc = evaluate_model(model, test_loader)
        scores.append(test_acc)
        labeled_sizes.append(len(X_labeled))

        print(f"Test accuracy: {test_acc:.4f}")

        if test_acc > best_acc:
            best_acc = test_acc
            best_model = model

        if len(X_pool) == 0:
            print("Pool is empty. Stopping early.")
            break

        n_select = min(query_size, len(X_pool))

        if paradigm == "active learning":
            probs = predict_probabilities(model, pool_loader).numpy()
            uncertainty = evaluate_uncertainty(probs, strategy)
            selected_indices = np.argsort(-uncertainty)[:n_select]

        elif paradigm == "random":
            selected_indices = rng.choice(len(X_pool), size=n_select, replace=False)

        else:
            raise ValueError(f"Unknown paradigm: {paradigm}")

        X_labeled, y_labeled, X_pool, y_pool = update_labeled_pool(
            X_labeled, y_labeled, X_pool, y_pool, selected_indices
        )

    return scores, labeled_sizes, best_model, best_acc