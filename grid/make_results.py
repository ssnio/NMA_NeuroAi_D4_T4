###################################################################################################
# # The transformer attention code is from "https://github.com/MathInf/toroidal" by Thomas Viehmann
# # The data is based on "https://arxiv.org/abs/2110.10090" by Edelman et al.
##########################################################################################

import random
import torch
import time
import os
import logging


def setup_logger(logger_name: str, logger_dir: str, logger_id: str):
    """
    Args:
        logger_name (str): Name of the log file
        logger_dir (str): Directory to save the log file into
    """
    # file and directory
    filename = os.path.join(logger_dir, logger_name + "_" + logger_id + ".log")

    logger = logging.getLogger(logger_name)

    # Set up handlers if not done already
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(module)s:%(funcName)s - %(message)s')
        file_handler = logging.FileHandler(filename)
        file_handler.setFormatter(formatter)
        stream_handler = logging.StreamHandler()
        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)
    
    return logger


# # Dataset
class OneDAnd:  # 1-Dimensional AND
    def __init__(self, T: int, s: int):
        self.T = T # context length
        self.s = s # sparsity
        self.p = 0.5**(1.0/3.0)  # probability chosen for balanced data
        self.f_i = None

    def pick_an_f(self):
        self.f_i = sorted(random.sample(range(self.T), 3))
        self.others = list(i for i in range(self.T) if i not in self.f_i)

    def generate(self, m: int, split: str = "train", verbose: bool = False):
        if self.f_i is None:
            self.pick_an_f()
        max_try = 100
        i_try = 0
        while i_try < max_try:
            i_try += 1
            X, y = torch.zeros(m, self.T), torch.zeros(m, 1)
            X[torch.rand(m, self.T) < self.p] = 1
            y[X[:, self.f_i].sum(dim=1) == self.s] = 1
            if y.sum()/m < 0.4 or y.sum()/m > 0.6:
                verbose and print(f"Large imbalance in the training set {y.sum()/m}, retrying...")
                continue
            else:
                verbose and print(f"Data-label balance: {y.sum()/m}")
            if split == "train":  # currently we choose not to do this
                bad_batch = False
                for i in self.f_i:
                    for o in self.others:
                        if (X[:, i] == X[:, o]).all():
                            verbose and print(f"Found at least another compatible hypothesis {i} and {o}")
                            bad_batch = True
                            break
                if bad_batch:
                    continue
                else:
                    break
            else:
                break
        else:
            print("Could not find a compatible hypothesis")
        return X.long(), y.float()


class BinaryEmbedding(torch.nn.Module):
    def __init__(self, T: int, d: int, v: int = 2):
        super().__init__()
        self.T = T  # context length
        self.d = d  # embedding size
        self.v = v  # vocabulary size
        self.token_embedding = torch.nn.Embedding(2, d)
        self.cls = torch.nn.Parameter(torch.randn(1, 1, d))  # "cls / class / global" learnable token
        self.position_embedding = torch.nn.Parameter(torch.randn(1, T + 1, d))  # positional embedding

        torch.nn.init.normal_(self.token_embedding.weight, std=0.02)
        torch.nn.init.normal_(self.position_embedding, std=0.02)
        torch.nn.init.normal_(self.cls, std=0.02)

    def forward(self, x):
        B = x.size(0)  # batch size
        x = self.token_embedding(x)
        x = torch.cat([x, self.cls.expand(B, -1, -1)], dim=1)
        x = x + self.position_embedding
        return x


class MLP(torch.nn.Sequential):
    # N.B.: timm also has a dropout layer between gelu and fc2
    #       but vit and deit seem to use no dropout and we want to be compatible
    #       with (A. Karpathy's mingpt implementation of) GPT2
    def __init__(self, d, n, mlp_drop=0.1):
        # d: input dimension
        # n: hidden dimension
        modules = [
            torch.nn.Linear(d, n),
            torch.nn.GELU(),
            torch.nn.Linear(n, d),
            torch.nn.Dropout(mlp_drop),
        ]

        super().__init__(*modules)
        torch.nn.init.normal_(self[0].weight, std=0.02)
        torch.nn.init.zeros_(self[0].bias)
        torch.nn.init.normal_(self[2].weight, std=0.02)
        torch.nn.init.zeros_(self[2].bias)


class Attention(torch.nn.Module):
    # big attention has dropout, too, sometimes qkv w/o bias
    def __init__(self, d, n_heads, att_drop=0.1, out_drop=0.1):
        super().__init__()
        self.d = d
        self.n_heads = n_heads
        self.scale = (d // n_heads) ** -0.5
        self.qkv = torch.nn.Linear(d, 3 * d)
        self.dropout_attn = torch.nn.Dropout(att_drop)
        self.proj = torch.nn.Linear(d, d)
        self.dropout_out = torch.nn.Dropout(out_drop)

        torch.nn.init.normal_(self.qkv.weight, std=0.02)
        torch.nn.init.zeros_(self.qkv.bias)
        torch.nn.init.normal_(self.proj.weight, std=0.02)
        torch.nn.init.zeros_(self.proj.bias)

    def forward(self, x):
        B, E, d = x.shape  # E = T + 1
        n_heads = self.n_heads

        # this is a trick to do q, k, v in one large linear for efficiency
        # unbind splits a tensor into a tuple of tensors
        q, k, v = self.qkv(x).view(B, E, 3, n_heads, -1).unbind(dim=2)

        logits = torch.einsum("bthc,bshc->bhts", q, k)
        logits *= self.scale  # normalize against staturation
        attn = logits.softmax(-1)
        attn = self.dropout_attn(attn)
        output = torch.einsum("bhts,bshc->bthc", attn, v)  # target source
        output = output.reshape(B, E, d)  # recombine
        output = self.proj(output)
        output = self.dropout_out(output)
        return output


class Block(torch.nn.Module):
    # works for GPT and DeiT
    def __init__(self, d, n_heads, n, att_drop=0.1, out_drop=0.1, mlp_drop=0.1, ln_eps=1e-6):
        super().__init__()
        self.norm1 = torch.nn.LayerNorm(d, eps=ln_eps)
        self.attn = Attention(d, n_heads, att_drop, out_drop)
        self.norm2 = torch.nn.LayerNorm(d, eps=ln_eps)
        self.mlp = MLP(d, n, mlp_drop)
        torch.nn.init.ones_(self.norm1.weight)
        torch.nn.init.zeros_(self.norm1.bias)
        torch.nn.init.ones_(self.norm2.weight)
        torch.nn.init.zeros_(self.norm2.bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class BinaryBERT(torch.nn.Module):
    def __init__(self, T: int, d: int, n_heads: int, n: int):
        super().__init__()
        self.T = T  # context length
        self.d = d  # embedding size
        self.n_heads = n_heads
        self.n = n  # number of hidden units
        assert d % n_heads == 0, "embedding size must be divisible by number of heads"
        self.v = 2  # vocabulary size
        n_blocks = 1
        self.embedding = BinaryEmbedding(self.T, self.d, self.v)
        self.blocks = torch.nn.Sequential(*[Block(d, n_heads, n, ln_eps=1e-6) for _ in range(n_blocks)])
        self.norm = torch.nn.LayerNorm(d, eps=1e-6)
        self.head = torch.nn.Linear(d, 1)
        torch.nn.init.ones_(self.norm.weight)
        torch.nn.init.zeros_(self.norm.bias)
        torch.nn.init.normal_(self.head.weight, std=0.02)  # scale with size?
        torch.nn.init.zeros_(self.head.bias)

    def forward(self, x):
        x = self.embedding(x)
        x = self.blocks(x)
        x = self.norm(x)
        x = self.head(x[:, -1])
        return x


def bin_acc(y_hat, y):
    y_ = y_hat.round()
    TP_TN = (y_ == y).float().sum().item()
    FP_FN = (y_ != y).float().sum().item()
    assert TP_TN + FP_FN == y.numel(), f"{TP_TN + FP_FN} != {y.numel()}"
    return TP_TN / y.numel()


def save_model(model):
    torch.save(model.state_dict(), 'model_states.pt')


def load_model(model):
    model_states = torch.load('model_states.pt')
    model.load_state_dict(model_states)


def evaluator(model, criterion, X_v, y_v, device="cpu"):
    model.to(device)
    model.eval()
    X_v, y_v = X_v.to(device), y_v.to(device)
    with torch.no_grad():
        y_hat = model(X_v)
        loss = criterion(y_hat.squeeze(), y_v.squeeze())
        acc = bin_acc(y_hat, y_v)
    return loss.item(), acc


def trainer(model, optimizer, criterion, n_epochs, X_t, y_t, X_v, y_v, device="cpu", verbose=False):
    # # book keeping
    train_loss, valid_loss = [], []
    train_acc, valid_acc = [], []
    model.to(device)
    model.train()
    X_t, y_t = X_t.to(device), y_t.to(device)
    X_v, y_v = X_v.to(device), y_v.to(device)
    for i in range(n_epochs):
        optimizer.zero_grad(set_to_none=True)
        y_hat = model(X_t)
        loss_t = criterion(y_hat.squeeze(), y_t.squeeze())
        loss_t.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        if (i + 1) % 10 == 0 or i == 0:
            train_loss.append(loss_t.item())
            train_acc.append(bin_acc(y_hat, y_t))
            model.eval()
            loss_v, acc_v = evaluator(model, criterion, X_v, y_v, device)
            valid_loss.append(loss_v)
            valid_acc.append(acc_v)
            verbose and print(f"Epoch {i + 1:04d}:"
            f" Train loss: {train_loss[-1]:.6f} acc: {train_acc[-1]:.3f}"
            f" Valid loss: {valid_loss[-1]:.6f} acc: {valid_acc[-1]:.3f}")
            model.train()
        # if valid_acc[-1] >= 0.99:
        #     break
    model.eval()
    return train_loss, valid_loss, train_acc, valid_acc

if __name__ == "__main__":

    # # create folder for results
    nma_dir = "./results"
    exp_name = "nma_"
    fold_id = str(int(time.time()))
    res_fold = os.path.join(nma_dir, "{}_{}".format(exp_name, fold_id))
    os.makedirs(res_fold)
    logger = setup_logger(logger_name=exp_name, logger_dir=res_fold, logger_id=fold_id)

    if torch.cuda.is_available():
        device = torch.device("cuda:0")  # NVIDIA GPU
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")  # Apple Silicon (Metal)
    else:
        device = torch.device("cpu")
    print(f"Device set to {device}")
    logger.info(f"Device set to {device}")

    T = 300  # context length
    s = 3  # sparsity
    B_t = 80  # batch size for training
    B_v = 500  # batch size for validation
    data_gen = OneDAnd(T, s)
    X_t, y_t = data_gen.generate(B_t, split="train", verbose=True)
    X_v, y_v = data_gen.generate(B_v, split="valid", verbose=True)


    d = 64  # embedding size
    n_heads = 16  # number of heads
    n = 64  # number of hidden units
    model = BinaryBERT(T, d, n_heads, n)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999), weight_decay=1e-4)
    criterion = torch.nn.BCELoss()

    T_list = [10, 20, 40, 60, 80, 100, 200, 300, 400, 500]
    B_list = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    n_epochs = 1000
    n_trials = 100
    s = 3
    B_v = 1000
    d = 64
    n_heads = 16
    n = 64
    criterion = torch.nn.BCELoss()
    full_t = time.time()
    train_log, valid_log = {}, {}
    for T in T_list:
        for B_t in B_list:
            train_log[f"{T}_{B_t}"] = []
            valid_log[f"{T}_{B_t}"] = []
            for i in range(n_trials):
                start_t = time.time()
                data_gen = OneDAnd(T, s)
                X_t, y_t = data_gen.generate(B_t, split="train", verbose=False)
                X_v, y_v = data_gen.generate(B_v, split="valid", verbose=False)

                model = BinaryBERT(T, d, n_heads, n)
                optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999), weight_decay=1e-4)

                train_loss, valid_loss, train_acc, valid_acc = trainer(model, optimizer, criterion, n_epochs, X_t, y_t, X_v, y_v, device=device, verbose=False)

                train_log[f"{T}_{B_t}"].append((train_loss, train_acc))
                valid_log[f"{T}_{B_t}"].append((valid_loss, valid_acc))
                print(f"Trial: {i+1:02d}   "
                    f"Time: {round(time.time() - start_t):02d}s   "
                    f"T: {T:03d}   "
                    f"B: {B_t:03d}   "
                    f"Train loss: {train_loss[-1]:.6f}   "
                    f"acc: {train_acc[-1]:.3f}   "
                    f"Valid loss: {valid_loss[-1]:.6f}   "
                    f"acc: {valid_acc[-1]:.3f}")
                logger.info(f"Trial: {i+1:02d}   "
                            f"Time: {round(time.time() - start_t):02d}s   "
                            f"T: {T:03d}   "
                            f"B: {B_t:03d}   "
                            f"Train loss: {train_loss[-1]:.6f}   "
                            f"acc: {train_acc[-1]:.3f}   "
                            f"Valid loss: {valid_loss[-1]:.6f}   "
                            f"acc: {valid_acc[-1]:.3f}")
    logger.info(f"FULL Time: {round(time.time() - full_t)}s")

    torch.save(train_log, os.path.join(res_fold, "train_log.pt"))
    torch.save(valid_log, os.path.join(res_fold, "valid_log.pt"))
