import torch
from tqdm import tqdm
from itertools import repeat
import matplotlib.pyplot as plt
from loss import ISMLoss, DSMLoss

def freeze(model):
    """
    (Optional) This is for Alternating Schrodinger Bridge.
    """
    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    return model


def unfreeze(model):
    """
    (Optional) This is for Alternating Schrodinger Bridge.
    """
    for p in model.parameters():
        p.requires_grad = True
    model.train()
    return model


def get_sde_step_fn(model, ema, opt, loss_fn, sde):
    def step_fn(batch):
        # uniformly sample time step
        t = sde.T*torch.rand(batch.shape[0])

        # TODO forward diffusion
        xt = None

        # get loss
        if isinstance(loss_fn, DSMLoss):
            logp_grad = None # TODO
            loss = loss_fn(t, xt.float(), model, logp_grad, diff_sq)
        elif isinstance(loss_fn, ISMLoss):
            loss = loss_fn(t, xt.float(), model)
        else:
            print(loss_fn)
            raise Exception("undefined loss")

        # optimize model
        opt.zero_grad()
        loss.backward()
        opt.step()

        if ema is not None:
            ema.update()

        return loss.item()

    return step_fn


def get_sb_step_fn(model_f, model_b, ema_f, ema_b,
                   opt_f, opt_b, loss_fn, sb, joint=True):
    def step_fn_alter(batch, forward):
        """
        (Optional) Implement the optimization step for alternating
        likelihood training of Schrodinger Bridge
        """
        pass

    def step_fn_joint(batch):
        """
        (Optional) Implement the optimization step for joint likelihood
        training of Schrodinger Bridge
        """
        pass

    if joint:
        return step_fn_joint
    else:
        return step_fn_alter


def repeater(data_loader):
    for loader in repeat(data_loader):
        for data in loader:
            yield data


def train_diffusion(dataloader, step_fn, N_steps, plot=False):
    pbar = tqdm(range(N_steps), bar_format="{desc}{bar}{r_bar}", mininterval=1)
    loader = iter(repeater(dataloader))

    log_freq = 200
    loss_history = torch.zeros(N_steps//log_freq)
    for i, step in enumerate(pbar):
        batch = next(loader)
        loss = step_fn(batch)

        if step % log_freq == 0:
            loss_history[i//log_freq] = loss
            pbar.set_description("Loss: {:.3f}".format(loss))

    if plot:
        plt.plot(range(len(loss_history)), loss_history)
        plt.show()
