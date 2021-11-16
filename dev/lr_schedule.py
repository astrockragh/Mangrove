from torch.optim.lr_scheduler import ExponentialLR, CosineAnnealingLR

class warmup_expcos():
    def __init__(self,optimizer, learning_rate=0, g_up=2, g_down=0.95, warmup=5, period=5, eta_min=1e-5):
        self.warmup=warmup
        self.s1 = ExponentialLR(optimizer, gamma=g_up)
        self.s2 = ExponentialLR(optimizer, gamma=g_down)
        #eta_min cannot be bigger than or equal to initial learning rate
        self.cos = CosineAnnealingLR(optimizer, period, eta_min=eta_min) 
    def step(self, epoch):
        if epoch<self.warmup:    
            self.s1.step()
        else:
            self.cos.step()
            self.s2.step()

class warmup_exp():
    def __init__(self,optimizer, learning_rate=0, g_up=2, g_down=0.95, warmup=5):
        self.warmup=warmup
        self.s1 = ExponentialLR(optimizer, gamma=g_up)
        self.s2 = ExponentialLR(optimizer, gamma=g_down)
    def step(self, epoch):
        if epoch<self.warmup:    
            self.s1.step()
        else:
            self.s2.step()
    