import os
import argparse
import torch
from timeit import default_timer as timer

from HPGT.utils.data import get_dataset
from torch_geometric.utils import degree, add_self_loops
import torch.nn as nn
import torch.nn.functional as F
from HPGT.utils.tools import set_random_seed, draw_loss, draw_acc, load_model, train_val_test_split, EarlyStopping, \
    select_mask
from PeSeGenerator_batch import get_batch_se


def load_args():
    parser = argparse.ArgumentParser(description='HPGT', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--seed', type=int, default=52, help='random seed')
    parser.add_argument('--dataset', type=str, default='texas', help='name of dataset')  # adjust
    parser.add_argument('--hidden-dim', type=int, default=64, help="hidden dimension of Transformer")  # adjust
    parser.add_argument('--num-walks', type=int, default=2, help='number of walks from one node')
    parser.add_argument('--walks-type', type=str, default='DBP', help='random walk type (RWP,utils)')
    parser.add_argument('--walks-length', type=int, default=10, help='the length of one walk')
    parser.add_argument('--walks-heads', type=int, default=1, help="number of multi-heads in walk")
    parser.add_argument('--use-se', type=bool, default=True, help="whether to use structure-encode?")
    parser.add_argument('--num-layers', type=int, default=2, help="number of transformer encoders")
    parser.add_argument('--transformer-heads', type=int, default=8, help="number of transformer multi-heads")
    parser.add_argument('--dropout', type=float, default=0.5, help="dropout-rate")
    parser.add_argument('--epochs', type=int, default=1000, help='number of epochs')
    parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate')  # adjust
    parser.add_argument('--weight-decay', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--use_warmup', type=bool, default=False, help='whether use warmup?')  # adjust
    parser.add_argument('--warmup', type=int, default=100, help="number of epochs for warmup")  # adjust
    parser.add_argument('--use_lr_schedule', type=bool, default=False, help='whether use warmup?')  # adjust
    parser.add_argument('--convergence_epoch', type=int, default=50, help="number of epochs for warmup")  # adjust
    parser.add_argument('--use_early_stopping', type=bool, default=False,
                        help="whether to use early stopping")  # adjust
    parser.add_argument('--patience', type=int, default=50, help='val_dataset loss increases or is stable '
                                                                 'before the maximum iterations')  # adjust
    parser.add_argument('--aggr', type=str, default='mean', help='Which aggregation to use?(sum,mean,max)')
    parser.add_argument('--model_name', type=str, default='DBPMLP', help='Which model to use?(DBPMLP,DBPGCN,'
                                                                         'DBPGIN,DBPGAT,DBPSAGE)')

    args = parser.parse_args()

    return args


def warmup_train(model, data, walks, loss_fn, optimizer, lr_scheduler, epoch, warmup, deg):
    model.train()
    if epoch <= warmup:
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr_scheduler(epoch)

    out = model(data, walks, deg)
    l = loss_fn(out[data.train_mask], data.y[data.train_mask])
    optimizer.zero_grad()
    l.backward()
    optimizer.step()

    pred = out.argmax(dim=1)
    train_correct_sum = (pred[data.train_mask] == data.y[data.train_mask]).sum()
    train_acc = train_correct_sum / int(data.train_mask.sum())

    return l.cpu().detach().numpy(), train_acc.cpu().detach().numpy()


def train(model, data, walks, loss_fn, optimizer, deg):
    model.train()
    out = model(data, walks, deg)
    l = loss_fn(out[data.train_mask], data.y[data.train_mask])
    optimizer.zero_grad()
    l.backward()
    optimizer.step()

    pred = out.argmax(dim=1)
    train_correct_sum = (pred[data.train_mask] == data.y[data.train_mask]).sum()
    train_acc = train_correct_sum / int(data.train_mask.sum())

    return l.cpu().detach().numpy(), train_acc.cpu().detach().numpy()


def validate(model, data, walks, loss_fn, deg):
    model.eval()
    with torch.no_grad():
        out = model(data, walks, deg)
        l = loss_fn(out[data.val_mask], data.y[data.val_mask])
        pred = out.argmax(dim=1)
        test_correct_sum = (pred[data.val_mask] == data.y[data.val_mask]).sum()
        test_acc = test_correct_sum / int(data.val_mask.sum())
    return l.cpu().detach().numpy(), test_acc.cpu().detach().numpy()


def test(model, data, walks, deg):
    model.eval()
    with torch.no_grad():
        out = model(data, walks, deg)
        pred = out.argmax(dim=1)
        test_correct_sum = (pred[data.test_mask] == data.y[data.test_mask]).sum()
        test_acc = test_correct_sum / int(data.test_mask.sum())
    return test_acc.cpu().detach().numpy(), pred[data.test_mask].cpu().detach().numpy()


def main():
    global args
    args = load_args()
    print(args)

    set_random_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    dataset = get_dataset(args.dataset, True)
    data = dataset[0]
    num_classes = dataset.num_classes

    data.train_mask, data.val_mask, data.test_mask = select_mask(7, data.train_mask, data.val_mask, data.test_mask)

    if args.use_se:
        walks_path = 'walks/' + args.dataset + 'Walks/' + args.walks_type + '_' + args.dataset + '_walks' + str(
            args.walks_length) + '.pt'
        if os.path.exists(walks_path):
            walks = torch.load(walks_path).to(device)
        else:
            get_batch_se(dataset_name=args.dataset, data=data,
                         num_walks=args.num_walks, walks_length=args.walks_length,
                         use_se=args.use_se, Type='utils')
            walks = torch.load(walks_path).to(device)
    else:
        walks = None

    deg = degree(data.edge_index[0])
    if sum(deg == 0) == 0:
        edge_index = data.edge_index
        deg = degree(edge_index[0])
    else:
        edge_index = add_self_loops(data.edge_index)[0]
        deg = degree(edge_index[0])
    deg = deg.to(device)
    data = data.to(device)

    model = load_model(args.model_name, num_nodes=data.x.shape[0], in_dim=data.x.shape[1], hidden_dim=args.hidden_dim,
                       out_dim=num_classes,
                       num_walks=args.num_walks, walks_length=args.walks_length, walks_heads=args.walks_heads,
                       num_layers=args.num_layers, transformer_heads=args.transformer_heads
                       , dropout=args.dropout, aggr=args.aggr, use_se=args.use_se).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    warmup = args.warmup

    lr = args.lr
    lr_steps = lr / warmup

    def warmup_lr_scheduler(s):
        lr = s * lr_steps
        return lr

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=50, gamma=0.5, last_epoch=-1)

    loss_fn = nn.CrossEntropyLoss().to(device)

    save_path = 'models/' + args.dataset  # 当前目录下
    early_stopping = EarlyStopping(save_path=save_path, patience=args.patience, verbose=False, delta=0)

    Train_Loss = []
    Train_acc = []
    Val_Loss = []
    Val_acc = []

    for i in range(args.epochs):
        if args.use_warmup:
            train_loss, train_acc = warmup_train(model=model, data=data, walks=walks, loss_fn=loss_fn,
                                                 optimizer=optimizer, lr_scheduler=warmup_lr_scheduler,
                                                 epoch=i, warmup=warmup, deg=deg)
        else:
            train_loss, train_acc = train(model=model, data=data, walks=walks, loss_fn=loss_fn, optimizer=optimizer,
                                          deg=deg)
        Train_Loss.append(train_loss)
        Train_acc.append(train_acc)
        val_loss, val_acc = validate(model=model, data=data, walks=walks, loss_fn=loss_fn, deg=deg)
        Val_acc.append(val_acc)
        Val_Loss.append(val_loss)
        test_acc, _ = test(model=model, data=data, walks=walks, deg=deg)

        if i % 10 == 0:
            print(
                'Epoch {:03d}'.format(i),
                '|| train',
                'loss : {:.3f}'.format(train_loss),
                ', accuracy : {:.2f}%'.format(train_acc * 100),
                '|| val',
                'loss : {:.3f}'.format(val_loss),
                ', accuracy : {:.2f}%'.format(val_acc * 100)
            )
        if i > args.convergence_epoch and args.use_lr_schedule:
            lr_scheduler.step()

        if args.use_early_stopping:
            early_stopping(val_loss, test_acc, model)
            if early_stopping.early_stop:
                print("Early stopping")
                print('Test_acc:{:.2f}'.format(early_stopping.test_acc * 100))
                break

    _, out = test(model=model, data=data, walks=walks, deg=deg)

    draw_loss(Train_Loss, len(Train_Loss), args.dataset, 'Train')
    draw_acc(Train_acc, len(Train_acc), args.dataset, 'Train')
    draw_loss(Val_Loss, len(Val_Loss), args.dataset, 'Val')
    # draw_acc(Val_acc, len(Val_acc), args.dataset, 'Val')
    print('test_acc:{:.2f}'.format(early_stopping.test_acc * 100))


if __name__ == "__main__":
    main()
