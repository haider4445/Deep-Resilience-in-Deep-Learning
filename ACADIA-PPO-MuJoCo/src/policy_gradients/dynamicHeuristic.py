import torch
import torch.nn as nn

from .Attack import Attack

class DynamicHeuristic(Attack):
    def __init__(self, model, targeted = -1, eps=0.007):
        super().__init__("DynamicHeuristic", model)
        self.eps = eps
        self._supported_mode = ['default', 'targeted']
        self.targeted = targeted

    def forward(self, images, labels):
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        if self.targeted:
            target_labels = self._get_target_label(images, labels)

        loss = nn.CrossEntropyLoss()

        with torch.enable_grad():

            images.requires_grad = True
            outputs = self.model(images)[0]

            # Calculate loss
            if self.targeted:
                cost = -loss(outputs, target_labels)
            else:
                cost = loss(outputs, labels)

            # Update adversarial images

            grad = torch.autograd.grad(cost, images,
                                       retain_graph=False, create_graph=False)[0]

            adv_images = images + self.eps*grad.sign()
            adv_images = torch.clamp(adv_images, min=0, max=1).detach()

        return adv_images