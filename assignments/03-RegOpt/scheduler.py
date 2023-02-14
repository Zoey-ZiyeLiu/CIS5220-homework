from typing import List
import torch
from torch.optim.lr_scheduler import _LRScheduler


class CustomLRScheduler(_LRScheduler):
    """
    Create a nre Scheduler.

    """

    def __init__(
        self,
        optimizer,
        last_epoch=-1,
        initial_lr=0.01,
        eta_min=0,
        T_max=2,
    ):
        """
        Create a new scheduler.

        Note to students: You can change the arguments to this constructor,
        if you need to add new parameters.

        """
        # ... Your Code Here ...

        # print("HERERE")
        self.optimizer = optimizer
        self.lr = initial_lr
        self.last_epoch = last_epoch
        # print(self.last_epoch,"NERE")
        self.eta_min = eta_min
        self.T_max = T_max
        super(CustomLRScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        """
        Get New learing rate.

        """
        # Note to students: You CANNOT change the arguments or return type of
        # this function (because it is called internally by Torch)

        # ... Your Code Here ...
        # Here's our dumb baseline implementation:
        # print(self.base_lrs,"base_lrs")
        # print("here")
        pi = torch.acos(torch.zeros(1)) * 2
        # print(type(pi))
        # self.lr=initial_lr
        # print(self.lr)
        if self.last_epoch == -1:
            lr = self.lr
            # self.last_epoch+=1
            # print(self.lr,"if")
        else:
            # print(self.last_epoch,"epoch")
            lr = self.eta_min + 0.5 * abs(self.lr - self.eta_min) * (
                1 + torch.cos(pi * self.last_epoch / self.T_max)
            )
            # print(lr,"else")
            # self.last_epoch+=1
        # return [i for i in self.lr]
        return lr
