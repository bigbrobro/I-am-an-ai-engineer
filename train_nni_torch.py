import argparse
import logging
import nni
import torch
import torch.nn.functional as F
import torch.optim as optim
from utils import get_logger
from network import Net
import os
from data_loader import data_loader

logger = logging.getLogger('mnist_AutoML')
output_logger = get_logger('model_output', logging_mode='INFO')


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args['log_interval'] == 0:
            logger.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            # sum up batch loss
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            # get the index of the max log-probability
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    accuracy = 100. * correct / len(test_loader.dataset)

    logger.info('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset), accuracy))

    return accuracy


def main(args, experiment_id, trial_id):
    use_cuda = not args['no_cuda'] and torch.cuda.is_available()
    torch.set_num_threads(4)
    torch.manual_seed(args['seed'])
    device = torch.device("cuda" if use_cuda else "cpu")

    batch_size = args['batch_size']
    hidden_size = args['hidden_size']

    train_loader, test_loader = data_loader(batch_size)

    model = Net(hidden_size=hidden_size).to(device)
    optimizer = optim.SGD(model.parameters(), lr=args['lr'],
                          momentum=args['momentum'])

    for epoch in range(1, args['epochs'] + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test_acc = test(args, model, device, test_loader)

        # report intermediate result
        nni.report_intermediate_result(test_acc)
        logger.debug('test accuracy %g', test_acc)
        logger.debug('Pipe send intermediate result done.')
        torch.save(model.state_dict(), f'{os.path.join(os.getcwd())}/model_outputs/{experiment_id}-{trial_id}-model.pth')

    test_acc = test(args, model, device, test_loader)
    # report final result
    nni.report_final_result(test_acc)
    logger.debug('Final result is %g', test_acc)
    output_logger.info(f'{experiment_id}|{trial_id}|{params}|{test_acc:0.6f}')
    logger.debug('Send final result done.')


def get_params():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument("--data_dir", type=str,
                        default='/tmp/tensorflow/mnist/input_data', help="data directory")
    parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument("--hidden_size", type=int, default=512, metavar='N',
                        help='hidden layer size (default: 512)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--epochs', type=int, default=2, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--no_cuda', action='store_true', default=True,
                        help='disables CUDA training')
    parser.add_argument('--log_interval', type=int, default=1000, metavar='N',
                        help='how many batches to wait before logging training status')

    args, _ = parser.parse_known_args()
    return args


if __name__ == '__main__':
    try:
        # get parameters form tuner
        tuner_params = nni.get_next_parameter()
        experiment_id = nni.get_experiment_id()
        trial_id = nni.get_trial_id()
        logger.debug(tuner_params)
        params = vars(get_params())
        params.update(tuner_params)
        main(params, experiment_id, trial_id)
    except Exception as exception:
        logger.exception(exception)
        raise
