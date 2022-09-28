import torch
import torch.nn as nn
import numpy as np

from Attack import Attack


class AIRFGM(Attack):
    r"""
    R+FGSM in the paper 'Ensemble Adversarial Training : Attacks and Defences'
    [https://arxiv.org/abs/1705.07204]
    Distance Measure : Linf
    Arguments:
        model (nn.Module): model to attack.
        eps (float): strength of the attack or maximum perturbation. (Default: 16/255)
        alpha (float): step size. (Default: 8/255)
        steps (int): num    ber of steps. (Default: 1)
    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.
    Examples::
        >>> attack = torchattacks.RFGSM(model, eps=16/255, alpha=8/255, steps=1)
        >>> adv_images = attack(images, labels)
    """
    def __init__(self, model, targeted = 0, eps=8/255, alpha=2/255, steps=20, decay = 0.99, decay2 = 0.999):
        super().__init__("AIRFGM", model)
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self._supported_mode = ['default', 'targeted']
        self.targeted = targeted
        self.decay = decay
        self.decay2 = decay2

    def forward(self, images, labels):
        r"""
        Overridden.
        """
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        last_cost = 1e-10
        delta = 1e-08
        best_adverse_state = images.clone().detach()
        accumulated_grad = torch.zeros_like(images).detach().to(self.device)

        accumulated_squared_grad = torch.zeros_like(images).detach().to(self.device)
        
        

        Sum_adap=0
        t = np.arange(1,self.steps+0.1,1)
        y1 = np.sqrt(1 - self.decay2**t) / (1 - self.decay**t)

        for x1 in y1:
            Sum_adap+=x1

        loss = nn.CrossEntropyLoss()

        adv_images = images + self.alpha*torch.randn_like(images).sign()
        adv_images = torch.clamp(adv_images, min=0, max=1).detach()
        with torch.enable_grad():

            for _ in range(self.steps):
                adv_images.requires_grad = True
                outputs = self.model(adv_images)[0]

                # Calculate loss
                if self.targeted:
                    cost = -loss(outputs, labels)
                else:
                    cost = loss(outputs, labels)

                # Update adversarial images
                grad = torch.autograd.grad(cost, adv_images,
                                           retain_graph=False, create_graph=False)[0]

                
                grad = grad / torch.mean(torch.abs(grad), dim=(1), keepdim=True)
                
                accumulated_grad = accumulated_grad*self.decay + grad * (1-self.decay)
                accumulated_squared_grad = accumulated_squared_grad*self.decay2 + (torch.square(grad))*(1-self.decay2)
                grad = accumulated_grad/(delta +torch.sqrt(accumulated_squared_grad))

                square = torch.sum(torch.square(grad))
                grad = grad / torch.sqrt(square)

                Adap_alpha = self.eps * np.sqrt(84*84) * (np.sqrt(1-self.decay2**(_+1))/(1-self.decay**(_+1))) / (Sum_adap)


                adv_images = adv_images.detach() + Adap_alpha*grad
                delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
                adv_images = torch.clamp(images + delta, min=0, max=1).detach()


        return adv_images