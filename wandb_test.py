import random

import wandb

run = wandb.init(
    entity="quguanni-california-institute-of-technology",
    project="c51_implementation",
    config={
        "learning_rate": 0.02,
        "architecture": "CNN",
        "dataset": "CIFAR-100",
        "epochs": 10,
    },
)

epochs = 10
offset = random.random() / 5
for epoch in range(2, epochs):
    acc = 1 - 2**-epoch - random.random() / epoch - offset
    loss = 2**-epoch + random.random() / epoch + offset

    run.log({"acc": acc, "loss": loss})

run.finish()