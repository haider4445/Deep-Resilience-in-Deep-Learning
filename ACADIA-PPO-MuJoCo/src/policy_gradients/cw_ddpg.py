import torch
import torch.nn as nn
import torch.optim as optim

from .Attack import Attack


class CW(Attack):
    r"""
    CW in the paper 'Towards Evaluating the Robustness of Neural Networks'
    [https://arxiv.org/abs/1608.04644]
    Distance Measure : L2
    Arguments:
        model (nn.Module): model to attack.
        c (float): c in the paper. parameter for box-constraint. (Default: 1e-4)    
            :math:`minimize \Vert\frac{1}{2}(tanh(w)+1)-x\Vert^2_2+c\cdot f(\frac{1}{2}(tanh(w)+1))`
        kappa (float): kappa (also written as 'confidence') in the paper. (Default: 0)
            :math:`f(x')=max(max\{Z(x')_i:i\neq t\} -Z(x')_t, - \kappa)`
        steps (int): number of steps. (Default: 1000)
        lr (float): learning rate of the Adam optimizer. (Default: 0.01)
    .. warning:: With default c, you can't easily get adversarial state. Set higher c like 1.
    Shape:
        - state: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - action: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of action`.
        - output: :math:`(N, C, H, W)`.
    Examples::
        >>> attack = torchattacks.CW(model, c=1e-4, kappa=0, steps=1000, lr=0.01)
        >>> adverse_state = attack(state, action)
    .. note:: Binary search for c is NOT IMPLEMENTED methods in the paper due to time consuming.
    """
    def __init__(self, model, targeted = -1, c=1e-4, kappa=0, steps=1000, lr=0.01):
        super().__init__("CW", model)
        self.c = c
        self.kappa = kappa
        self.steps = steps
        self.lr = lr
        self._supported_mode = ['default', 'targeted']
        self.targeted = targeted

    def forward(self, state, action, target_action=None, target_location=None):
        r"""
        Overridden.
        """
        state = state.clone().detach().to(self.device)
        action = action.clone().detach().to(self.device)

        # w = torch.zeros_like(state).detach() # Requires 2x times
        state = torch.div(torch.add(torch.div(state, 100.0), 1), 2.0)
        w = self.inverse_tanh_space(state).detach()
        w.requires_grad = True

        best_adverse_state = state.clone().detach()
        prev_cost = 1e10
        last_cost = 1e10

        MSELoss = nn.MSELoss(reduction='none')
        MSE_Sum = nn.MSELoss(reduction='sum')
        Flatten = nn.Flatten()

        optimizer = optim.Adam([w], lr=self.lr)
        with torch.enable_grad():
            for step in range(self.steps):
                # print('Step: ', step)
                # Get adversarial state
                adverse_state = self.tanh_space(w)

                # Calculate loss
                current_L2 = MSELoss(Flatten(adverse_state), Flatten(state)).sum(dim=1)
                L2_loss = current_L2.sum()
                # print('L2 Loss: ', L2_loss.item())
                
                adverse_action = self.model(torch.mul(torch.sub(torch.mul(adverse_state, 2.0), 1), 100.0))[0]
                if self.targeted:
                    f_loss = self.f(adverse_action, target_action).sum()
                    # print('F Loss: ', f_loss.item())
                    # print(target_action)
                    # print(adverse_action)
                    # print('F Loss: ', f_loss.item())
                else:
                    f_loss = self.f(adverse_action, action).sum()
                cost = L2_loss + self.c * f_loss
                # print('Cost: ', cost.item())
                optimizer.zero_grad()
                cost.backward()
                optimizer.step()

                # Update adversarial state
                if f_loss.item() < last_cost:
                    best_adverse_state = adverse_state.detach()
                    last_cost = f_loss.item()

                # print('Step: ', step, 'F Loss', MSE_Sum(self.model.network.actor(torch.mul(torch.sub(torch.mul(best_adverse_state, 2.0), 1), 100.0)), target_action).item())

                # Early stop when loss does not converge.
                if step % (self.steps//10) == 0:
                    if cost.item() > prev_cost:
                        return torch.mul(torch.sub(torch.mul(best_adverse_state, 2.0), 1), 100.0)
                    prev_cost = cost.item()
                # print('Loss:', MSELoss(Flatten(torch.mul(torch.sub(torch.mul(best_adverse_state, 2.0), 1), 100.0)), Flatten(torch.mul(torch.sub(torch.mul(state, 2.0), 1), 100.0))).sum(dim=1)[1].item())

        return torch.mul(torch.sub(torch.mul(best_adverse_state, 2.0), 1), 100.0)

    def tanh_space(self, x):
        return 1 / 2 * (torch.tanh(x) + 1)

    def inverse_tanh_space(self, x):
        # torch.atanh is only for torch >= 1.7.0
        return self.atanh(x * 2 - 1)

    def atanh(self, x):
        return 0.5 * torch.log((1 + x) / (1 - x))

    # f1 from the paper adapted to mujoco
    def f(self, adverse_action, action):
        temp_loss = torch.nn.MSELoss(reduction='none')
        if self.targeted:
            return temp_loss(adverse_action, action)
        else:
            return -temp_loss(adverse_action, action) + 1