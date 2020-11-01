from data import *
from model import *
import matplotlib.pyplot as plt

hidden_size = 256
n_iter = 100000
record_every = 5000
device = "cuda:0" if torch.cuda.is_available() else "cpu"

# generator = RNN(n_category, n_letters, hidden_size, n_letters)
generator = NameGenerationRNN(n_category, n_letters, hidden_size, n_letters)
criterion = torch.nn.NLLLoss()
optim = torch.optim.Adam(generator.parameters())
for name, param in generator.named_parameters():
    print(name, param.size())


def train():
    total_loss = []
    generator.to(device)
    for i in range(n_iter):
        category_tensor, name_tensor, target_tensor, category, name = random_training_example()
        hidden1 = generator.init_hidden1().to(device)
        hidden2 = generator.init_hidden2().to(device)
        loss = 0
        category_tensor = category_tensor.to(device)
        name_tensor = name_tensor.to(device)
        target_tensor = target_tensor.to(device)
        optim.zero_grad()
        rnn_outputs = []
        for j in range(name_tensor.size(0)):
            output, hidden1, hidden2 = generator(category_tensor, name_tensor[j], hidden1, hidden2)
            if i % record_every == (record_every-1):
                rnn_outputs.append(output)
            l = criterion(output, target_tensor[j].view(1))
            loss += l
        loss /= name_tensor.size(0) - 1
        if i % record_every == (record_every-1):
            total_loss.append(loss.item()/record_every)
            print_training_details(rnn_outputs, category, name)
        loss.backward()
        optim.step()
    return generator, total_loss


def print_training_details(output_list, category, name):
    print("category: %s,\t\t name:\t\t %s" % (category, name), end='\t\t output: ')
    for output_tensor in output_list:
        print(tensor_to_word(output_tensor), end='')
    print()


def tensor_to_word(t):
    _, idx = t.topk(1, dim=-1)
    if idx == n_letters - 1:
        return '<eos>'
    if idx == n_letters - 2:
        return '<sos>'
    return all_letters[idx]


if __name__ == '__main__':
    generator, loss = train()
    torch.save(generator.state_dict(), 'my_rnn.pt')
    plt.plot(loss)
    plt.show()
