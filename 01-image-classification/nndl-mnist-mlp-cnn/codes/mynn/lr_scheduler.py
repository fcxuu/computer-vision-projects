from abc import abstractmethod
import numpy as np

class scheduler():
    def __init__(self, optimizer) -> None:
        self.optimizer = optimizer
        self.step_count = 0
    
    @abstractmethod
    def step():
        pass


class StepLR(scheduler):
    def __init__(self, optimizer, step_size=30, gamma=0.1) -> None:
        super().__init__(optimizer)
        self.step_size = step_size
        self.gamma = gamma

    def step(self) -> None:
        self.step_count += 1
        if self.step_count >= self.step_size:
            self.optimizer.init_lr *= self.gamma
            self.step_count = 0

class MultiStepLR(scheduler):
    def __init__(self, optimizer, milestones=[30, 60, 90], gamma=0.1) -> None:
        """
        Decays the learning rate by gamma once the number of steps reaches one of the milestones.
        
        Args:
            optimizer: The optimizer to adjust learning rate
            milestones: List of step indices at which to reduce the learning rate
            gamma: Multiplicative factor of learning rate decay
        """
        super().__init__(optimizer)
        self.milestones = sorted(milestones)
        self.gamma = gamma
        self.milestone_index = 0

    def step(self) -> None:
        self.step_count += 1
        
        # Check if we've reached a milestone
        if self.milestone_index < len(self.milestones) and self.step_count == self.milestones[self.milestone_index]:
            self.optimizer.init_lr *= self.gamma
            self.milestone_index += 1

class ExponentialLR(scheduler):
    def __init__(self, optimizer, gamma=0.95) -> None:
        """
        Decays the learning rate by gamma every step.
        
        Args:
            optimizer: The optimizer to adjust learning rate
            gamma: Multiplicative factor of learning rate decay
        """
        super().__init__(optimizer)
        self.gamma = gamma

    def step(self) -> None:
        self.step_count += 1
        # Apply exponential decay to learning rate
        self.optimizer.init_lr *= self.gamma