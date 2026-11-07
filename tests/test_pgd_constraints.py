import torch
from triaudit.attacks.pgd import pgd, validate_constraints
from triaudit.models.resnet_cifar import ResNet18CIFAR

def test_pgd_constraints():
    model = ResNet18CIFAR().eval()
    x = torch.rand(4, 3, 32, 32)
    y = torch.randint(0, 10, (4,))
    eps = 8/255
    adv = pgd(model, x, y, eps=eps, step_size=2/255, steps=7, random_start=True)
    c = validate_constraints(x, adv, eps)
    assert c["box_ok"] and c["eps_ok"], c
