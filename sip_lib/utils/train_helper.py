


from omegaconf import OmegaConf
class TrainHelper():
    def __init__(self, num_steps_per_epoch, num_eval, num_save,  num_train_steps=None, num_epochs=None,  **kwargs):
        assert not (num_train_steps is not None and num_epochs is not None)
        assert num_train_steps is not None or num_epochs is not None 
        
        if num_train_steps is not None:
            self.num_train_steps = num_train_steps 
            self.num_epochs = int(num_train_steps / num_steps_per_epoch)
        else:
            self.num_train_steps = num_epochs * num_steps_per_epoch
        
        if num_epochs is not None:
            self.num_epochs = num_epochs
            self.num_train_steps = num_epochs * num_steps_per_epoch
        else:
            self.num_epochs = int(num_train_steps / num_steps_per_epoch)

        self.num_steps_per_epoch = num_steps_per_epoch
        self.num_eval = num_eval
        self.num_save = num_save
        self.eval_freq = self.num_train_steps // num_eval
        self.save_freq = self.num_train_steps // num_save
        self.global_step = 0 
        self.config = OmegaConf.create({
            "num_train_steps": self.num_train_steps,
            "num_epochs": self.num_epochs,
            "num_steps_per_epoch": self.num_steps_per_epoch,
            "num_eval": self.num_eval,
            "num_save": self.num_save,
            'save_freq': self.save_freq,
            'eval_freq': self.eval_freq,
            "global_step" : self.global_step,
        })
        print(self.config)
        
    def update_global_step(self):
        self.global_step += 1
        self.config['global_step'] = self.global_step 
        
    def is_eval_step(self):
        if self.global_step % self.eval_freq ==0 :
            return True 
        elif self.global_step == self.num_train_steps -1 :
            return True 
        else:
            return False 
    
    def is_save_step(self):
        if self.global_step % self.save_freq ==0 :
            return True 
        elif self.global_step == self.num_train_steps -1 :
            return True 
        else:
            return False 
    
    def prepare_resume_training(self, global_step, optimizer, lr_scheduler):
        for _ in range(global_step):
            lr_scheduler.step()
            self.global_step += 1
        return optimizer, lr_scheduler
        