import torch


class NegativeMSE:

    def __call__(self, model, dataset):
        total = torch.tensor(0, dtype=torch.float)
        for x, y in dataset:
            y_hat = torch.squeeze(model(torch.unsqueeze(x, 0)))
            residue = y - y_hat
            square = residue**2
            total += square
        return -total / len(dataset)


class ClassificationAccuracy:

    def __call__(self, model, dataset):
        num_correct = 0
        for x, g in dataset:
            y_hat = torch.squeeze(model(torch.unsqueeze(x, 0)))
            g_hat = torch.argmax(y_hat)
            if g_hat == g:
                num_correct += 1
        return num_correct / len(dataset)
