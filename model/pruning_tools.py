import torch.nn.utils.prune as prune
import torch
import numpy as np
class Regrowth(prune.BasePruningMethod):
    '''Regrow pruned parameters
    regrowth method specified in constructor should be
    - "random" for regrowing random connections
     or
    - "magnitude" for regrowing important connections
    '''

    PRUNING_TYPE = 'global'

    def __init__(self,
                 regrowth_method,
                 amount,
                 seed=5000):
        '''Regrowth class constructor
            args: regrowth_method -> "random" or "magnitude"
                  amount of connections to be regrown (0, 1)'''

        self.regrowth_method = regrowth_method
        self.amount = amount
        self.seed = seed
    def compute_mask(self, t, default_mask):
        mask = default_mask.clone()
        # calculate complementary to given mask e.g. [0, 0, 1, 0] -> [1, 1, 0, 1]
        complement_mask = torch.logical_xor(mask, torch.ones_like(mask)).type(mask.type())
        num_pruned = int(torch.sum(complement_mask))
        num_to_regrow = int(self.amount * num_pruned)
        pruned_weight_indices = torch.nonzero(complement_mask)
        if self.regrowth_method == 'random':
            indices_of_chosen_ones = np.random.choice(num_pruned, size=num_to_regrow, replace=False)
            chosen_ones = pruned_weight_indices[indices_of_chosen_ones]
            mask[chosen_ones] = 1.
        if self.regrowth_method == 'magnitude':
            pruned_weights_mask = complement_mask
            pruned_weights = t * pruned_weights_mask
            pruned_weights_flat = pruned_weights.flatten()
            mask_flat = mask.flatten()
            most_important = torch.topk(torch.abs(pruned_weights_flat), num_to_regrow).indices
            mask_flat[most_important] = 1.
            mask = torch.reshape(mask_flat, tuple(mask.size()))
        return mask

def regrowth_unstructured(module, name, regrowth_method, amount, seed=5000):
    kwargs = {'regrowth_method': regrowth_method,
              'amount': amount,
              'seed': seed}
    Regrowth.apply(module, name, **kwargs)


class RegrowthRigL(prune.BasePruningMethod):


    PRUNING_TYPE = 'global'

    def __init__(self,
                 amount):
        self.amount = amount
    def compute_mask(self, t, default_mask):
        mask = default_mask.clone()
        # calculate complementary to given mask e.g. [0, 0, 1, 0] -> [1, 1, 0, 1]
        complement_mask = torch.logical_xor(mask, torch.ones_like(mask)).type(mask.type())
        num_pruned = int(torch.sum(complement_mask))
        number_of_weights = torch.numel(t)
        num_to_regrow = int(self.amount * number_of_weights)
        #regrow
        pruned_grads_mask = complement_mask
        pruned_grads = t * pruned_grads_mask
        pruned_grads_flat = pruned_grads.flatten()
        mask_flat = mask.flatten()
        most_important = torch.topk(torch.abs(pruned_grads_flat), num_to_regrow).indices
        mask_flat[most_important] = 1.
        mask = torch.reshape(mask_flat, tuple(mask.size()))
        return mask

def regrowth_rigl(module, name, amount):
    kwargs = {'amount': amount}
    RegrowthRigL.apply(module, name, **kwargs)

if __name__ == "__main__":
    regrowth_pruning = Regrowth(regrowth_method='magnitude', amount=0.5)
    regrowth_pruning.compute_mask(torch.tensor([[20., 10., 1.],
                                                 [1., 40., 10.]]), torch.tensor([[0., 0., 1.],
                                                                                 [1., 0., 0.]]))
