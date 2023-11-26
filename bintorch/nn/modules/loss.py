from .module import Module
from .. import functional as F

class CrossEntropyLoss(Module):

    def __init__(self, size_average=True, weight=None, reduction='mean'):
        super(CrossEntropyLoss, self).__init__()
        self.size_average = size_average
        self.weight = weight
        self.reduction = reduction

    def forward(self, input, target):
        return F.cross_entropy(input, target, 
            size_average=self.size_average, 
            weight=self.weight, 
            reduction=self.reduction
        )
