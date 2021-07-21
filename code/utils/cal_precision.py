import torch

def cal_precision(pred,ture_label):
    """
    Args:
        pred : tensor, shape [N]
        ture_label : tensor true label shape [N]
    return :
        precision
    """
    # assume pred.shape == ture_label.shape
    pred = (pred > 0.5)
    res = (pred == ture_label)
    count = torch.mean(res.float()).item()
    return 100.0*count

def test():
    pred = torch.Tensor([0.6,0.7,0.1,0.4,0.6])
    true_label = torch.Tensor([1,0,0,0,1])
    print(cal_precision(pred,true_label))

if __name__ == "__main__":
    test()
