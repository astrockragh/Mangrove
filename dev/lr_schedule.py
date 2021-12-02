from torch.optim.lr_scheduler import ExponentialLR, CosineAnnealingLR, MultiplicativeLR

class warmup_expcos():
    def __init__(self,optimizer, g_up=2, g_down=0.95, warmup=5, period=5, eta_min=1e-5,  learning_rate=1, schedule='null'):
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

class cosanneal():
    def __init__(self,optimizer, g_up=2, g_down=0.95, warmup=5, period=5, eta_min=1e-5,  learning_rate=1, schedule='null'):
        #eta_min cannot be bigger than or equal to initial learning rate
        self.cos = CosineAnnealingLR(optimizer, period, eta_min=eta_min) 
    def step(self, epoch):
        self.cos.step()

class warmup_exp():
    def __init__(self,optimizer, g_up=2, g_down=0.95, warmup=5, period=5, eta_min=1e-5,  learning_rate=1, schedule='null'):

        self.warmup=warmup
        self.up = ExponentialLR(optimizer, gamma=g_up)
        self.down = ExponentialLR(optimizer, gamma=g_down)
    def step(self, epoch):
        if epoch<self.warmup:    
            self.up.step()
        else:
            self.down.step()

class constant_down():
    def __init__(self,optimizer, g_up=2, g_down=0.95, warmup=5, period=5, eta_min=1e-5,  learning_rate=1, schedule='null'):

        self.warmup=warmup
        self.up = ExponentialLR(optimizer, gamma=1)
        self.down = ExponentialLR(optimizer, gamma=g_down)
    def step(self, epoch):
        if epoch<self.warmup:    
            self.up.step()
        else:
            self.down.step()

class constant():
    def __init__(self,optimizer, g_up=2, g_down=0.95, warmup=5, period=5, eta_min=1e-5,  learning_rate=1, schedule='null'):
        lmbda = lambda epoch: 1
        self.s1=MultiplicativeLR(optimizer, lr_lambda=lmbda)
    def step(self, epoch):
        self.s1.step()
