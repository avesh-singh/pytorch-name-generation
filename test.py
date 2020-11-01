from main import RNN, hidden_size, device, NameGenerationRNN, tensor_to_word
from data import *


# generator = RNN(n_category, n_letters, hidden_size, n_letters)
generator = NameGenerationRNN(n_category, n_letters, hidden_size, n_letters)
generator.load_state_dict(torch.load('my_rnn.pt'))
generator.eval()
max_length = 10
print("device: %s" % device)
generator.to(device)


def sample(category):
    with torch.no_grad():
        category_tensor = category_to_tensor(category).to(device)
        start_tensor = get_start_tensor()
        hidden = generator.init_hidden().to(device)
        word_tensor = start_tensor
        name_generated = ''
        while len(name_generated) < max_length:
            output, hidden = generator(category_tensor, word_tensor[0], hidden)
            if is_eos(output):
                break
            word = tensor_to_word(output)
            name_generated += word
            word_tensor = word_to_tensor(word).to(device)

        return name_generated


def is_eos(output):
    _, idx = output.topk(1, dim=-1)
    return idx == n_letters - 1


def sample_names_from_generator(category, count=3):
    for c in range(count):
        print(sample(category))


def get_start_tensor():
    t = torch.zeros(1, 1, n_letters).to(device)
    t[0, 0, n_letters-2] = 1
    return t


for category in all_categories:
    sample_names_from_generator(category, 2)
