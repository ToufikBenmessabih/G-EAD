'''# Define the warmup function
    def warmup_lr_lambda(epoch):
        if epoch < config.warmup_epochs:
            return float(epoch + 1) / float(config.warmup_epochs)
        return 1.0
    
    # Define the warmup scheduler
    warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_lr_lambda)

    # Define the post-warmup scheduler
    post_warmup_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config.step_size, gamma=config.gamma)'''


    '''# Define the custom learning rate schedule
    def custom_lr_lambda(epoch):
        if epoch < config.fixed_epoch:  # config.fixed_epoch is the epoch where you want to stop increasing LR (e.g., 20)
            return config.fixed_epoch - epoch
        else:
            return 1 # Keep it fixed after reaching the desired epoch

    # Apply the custom scheduler
    custom_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=custom_lr_lambda)'''