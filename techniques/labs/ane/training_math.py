"""Tiny standard-library mathematical witnesses; no device/runtime imports."""
import math


def transpose(matrix):
    return [list(column) for column in zip(*matrix)]


def matmul(a, b):
    return [[math.fsum(x * y for x, y in zip(row, column))
             for column in zip(*b)] for row in a]


def linear_loss_gradient(x, w, target):
    y = matmul(x, w)
    residual = [[a - b for a, b in zip(row, reference)] for row, reference in zip(y, target)]
    loss = .5 * math.fsum(v * v for row in residual for v in row) / len(x)
    dy = [[v / len(x) for v in row] for row in residual]
    return loss, matmul(transpose(x), dy), matmul(dy, transpose(w))


def witness():
    x = [[.5, -.25], [-.75, 1.], [.25, .5]]
    w = [[.2, -.4], [.3, .1]]
    target = [[.1, -.2], [.3, .2], [-.1, .4]]
    loss, dw, _ = linear_loss_gradient(x, w, target)
    epsilon = 1e-6
    errors = []
    for i in range(2):
        for j in range(2):
            plus, minus = [r[:] for r in w], [r[:] for r in w]
            plus[i][j] += epsilon
            minus[i][j] -= epsilon
            numeric = (linear_loss_gradient(x, plus, target)[0] - linear_loss_gradient(x, minus, target)[0]) / (2 * epsilon)
            errors.append(abs(numeric - dw[i][j]))
    # First-step Adam's bias correction cancels; epsilon does not cancel scale.
    g, eps, scale = 1e-9, 1e-8, 64.
    original = g / (abs(g) + eps)
    scaled_unchanged_eps = scale * g / (scale * abs(g) + eps)
    scaled_scaled_eps = scale * g / (scale * abs(g) + scale * eps)
    return {"scope": "CPU mathematical witness, not device training",
            "loss": loss, "maximum_gradient_absolute_error": max(errors),
            "adam_first_step_original": original,
            "adam_scaled_gradient_same_epsilon": scaled_unchanged_eps,
            "adam_scaled_gradient_scaled_epsilon": scaled_scaled_eps}


def training_contract(seed=17):
    from common import report
    value = witness()
    if value["maximum_gradient_absolute_error"] >= 1e-9:
        raise ValueError("finite-difference gradient contract failed")
    return report("ane-training-math", value,
                  scope="standard-library CPU gradient/Adam-epsilon mathematical controls; no device execution")


if __name__ == "__main__":
    import json
    print(json.dumps(witness(), indent=2))
