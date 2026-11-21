import torch

class SkillState:

    def __init__(self, n, device):
        self.mode = torch.zeros(n, 2, dtype=torch.bool, device=device)
        self.left = torch.zeros(n, 2, dtype=torch.long, device=device)
        self.command = torch.zeros(n, 2, 3, device=device)
        self.settled = torch.zeros(n, 2, device=device)
        self.kick_count = torch.zeros(n, 2, dtype=torch.long, device=device)

    def reset(self, ids=None):
        ids = slice(None) if ids is None else ids
        for tensor in (self.mode, self.left, self.command, self.settled, self.kick_count):
            tensor[ids] = 0

    def start(self, triggers):
        starts = triggers & ~self.mode
        self.mode |= starts
        self.left[starts] = 100
        self.kick_count += starts.long()
        return starts

    def tick(self):
        self.left[self.mode] -= 1
        self.mode &= self.left > 0
