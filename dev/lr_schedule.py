from torch.optim.lr_scheduler import ExponentialLR, CosineAnnealingLR, MultiplicativeLR, OneCycleLR

##################################################
### LR schedule, call as string               ####
### warmup_expcos is homemade but works well  ####
##################################################
class warmup_expcos():
    ''' exp up and down followed by cosine annealing'''
    def __init__(self,optimizer, g_up=4, g_down=0.98, warmup=15, period=5, eta_min=1e-5,  learning_rate=1, schedule='null', total_steps=10000):
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
    '''cosine anneal that needs to fit with the rest'''
    def __init__(self,optimizer, g_up=2, g_down=0.95, warmup=5, period=5, eta_min=1e-5,  learning_rate=1, schedule='null', total_steps=10000):
        #eta_min cannot be bigger than or equal to initial learning rate
        self.cos = CosineAnnealingLR(optimizer, period, eta_min=eta_min) 
    def step(self, epoch):
        self.cos.step()

class warmup_exp():
    '''Double exponential'''

    def __init__(self,optimizer, g_up=2, g_down=0.95, warmup=5, period=5, eta_min=1e-5,  learning_rate=1, schedule='null', total_steps=10000):

        self.warmup=warmup
        self.up = ExponentialLR(optimizer, gamma=g_up)
        self.down = ExponentialLR(optimizer, gamma=g_down)
    def step(self, epoch):
        if epoch<self.warmup:    
            self.up.step()
        else:
            self.down.step()

class constant():
    '''constant lr, good place to start'''
    def __init__(self,optimizer, g_up=2, g_down=0.95, warmup=5, period=5, eta_min=1e-5,  learning_rate=1, schedule='null', total_steps=10000):
        lmbda = lambda epoch: 1
        self.s1=MultiplicativeLR(optimizer, lr_lambda=lmbda)
    def step(self, epoch):
        self.s1.step()

class onecycle():
    '''One cycle LR scheme, not guaranteed to be optimal, but it's pretty stable for lr <1e-2'''
    def __init__(self,optimizer, g_up=2, g_down=0.95, warmup=5, period=5, eta_min=1e-5,  learning_rate=1, schedule='null', total_steps=10000):
        self.s1=OneCycleLR(optimizer, max_lr=learning_rate, total_steps=total_steps, pct_start=0.15, final_div_factor=1e3)
    def step(self, epoch):
        self.s1.step()