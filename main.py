import io
import math
import time

from torchtext.vocab import vocab
from torchtext.data.utils import get_tokenizer
from torch.utils.data import DataLoader
from collections import Counter
import torch
import spacy

de_tokenizer = get_tokenizer('spacy', language='de_core_news_sm')
en_tokenizer = get_tokenizer('spacy', language='en_core_web_sm')


def build_vocab(filepath: str, tokenizer) -> vocab:
    counter = Counter()
    with io.open(filepath, encoding='utf8') as file:
        for line in file:
            counter.update(tokenizer(line))
    return vocab(counter, specials=['<bos>', '<pad>', '<eos>', '<unk>'])


de_vocab = build_vocab('.data/train.de', de_tokenizer)
en_vocab = build_vocab('.data/train.en', en_tokenizer)


def data_preprocess() -> [tuple]:
    de_iter = iter(io.open('.data/train.de', encoding='utf8'))
    en_iter = iter(io.open('.data/train.en', encoding='utf8'))
    datas = []
    for (de_sentence, en_sentence) in zip(de_iter, en_iter):
        de_tensor = torch.tensor([de_vocab[token] for token in de_tokenizer(de_sentence.rstrip('\n'))])
        en_tensor = torch.tensor([en_vocab[token] for token in en_tokenizer(en_sentence.rstrip('\n'))])
        datas.append((de_tensor, en_tensor))
    return datas


train_data = data_preprocess()


def collate(tuples: [tuple]) -> ([torch.tensor], [torch.tensor]):
    de_batch, en_batch = [], []
    for (de_tensor, en_tensor) in tuples:
        de_batch.append(torch.cat([torch.tensor([de_vocab['<bos>']]),
                                   de_tensor,
                                   torch.tensor([de_vocab['<eos>']])],
                                  dim=0))
        en_batch.append(torch.cat([torch.tensor([en_vocab['<bos>']]),
                                   en_tensor,
                                   torch.tensor([en_vocab['<eos>']])],
                                  dim=0))
    de_batch = torch.nn.utils.rnn.pad_sequence(de_batch, padding_value=de_vocab['<pad>'])
    en_batch = torch.nn.utils.rnn.pad_sequence(en_batch, padding_value=en_vocab['<pad>'])

    return de_batch, en_batch


batch_size = 128

train_iter = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, collate_fn=collate)


class PositionalEncoding(torch.nn.Module):
    def __init__(self, emb_size: int, dropout, maxlen: int = 5000):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(- torch.arange(0, emb_size, 2) * math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = torch.nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding):
        return self.dropout(token_embedding +
                            self.pos_embedding[:token_embedding.size(0), :])


class TokenEmbedding(torch.nn.Module):
    def __init__(self, vocab_size: int, emb_size):
        super(TokenEmbedding, self).__init__()
        self.embedding = torch.nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size

    def forward(self, tokens):
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)


class Transformer(torch.nn.Module):
    def __init__(self, input_size: int, output_size: int, n_encoder_layers: int, n_decoder_layers: int):
        super(Transformer, self).__init__()

        self.input_embedding = TokenEmbedding(input_size, 512)
        self.output_embedding = TokenEmbedding(output_size, 512)
        self.positional_encoding = PositionalEncoding(512, dropout=0.1)

        self.encoder_layer = torch.nn.TransformerEncoderLayer(512, 8)
        self.decoder_layer = torch.nn.TransformerDecoderLayer(512, 8)

        self.encoder = torch.nn.TransformerEncoder(self.encoder_layer, n_encoder_layers)
        self.decoder = torch.nn.TransformerDecoder(self.decoder_layer, n_decoder_layers)

        self.linear = torch.nn.Linear(512, output_size)

    def forward(self, x, y, src_mask, trg_mask, src_padding_mask, trg_padding_mask, memory_key_padding_mask):
        embedded_x = self.positional_encoding(self.input_embedding(x))
        embedded_y = self.positional_encoding(self.output_embedding(y))

        memory = self.encoder(embedded_x, src_mask, src_padding_mask)

        outputs = self.decoder(embedded_y, memory, trg_mask, None, trg_padding_mask, memory_key_padding_mask)

        return self.linear(outputs)

    def encode(self, x, src_mask, src_padding_mask=None):
        return self.encoder(self.positional_encoding(self.input_embedding(x)), src_mask, src_padding_mask)

    def decode(self, y, memory, trg_mask, trg_padding_mask=None, memory_key_padding_mask=None):
        return self.decoder(self.positional_encoding(self.output_embedding(y)), memory,
                            trg_mask, trg_padding_mask, memory_key_padding_mask)


def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones((sz, sz))) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


def create_mask(src, tgt):
    src_seq_len = src.shape[0]
    tgt_seq_len = tgt.shape[0]

    tgt_mask = generate_square_subsequent_mask(tgt_seq_len)
    src_mask = torch.zeros((src_seq_len, src_seq_len)).type(torch.bool)

    src_padding_mask = (src == de_vocab['<pad>']).transpose(0, 1)
    tgt_padding_mask = (tgt == en_vocab['<pad>']).transpose(0, 1)
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask


de_vocab_size = len(de_vocab)
en_vocab_size = len(en_vocab)
n_epochs = 16
n_layers = 3
learning_rate = 0.01

transformer = Transformer(de_vocab_size, en_vocab_size, n_layers, n_layers)
criterion = torch.nn.CrossEntropyLoss(ignore_index=de_vocab['<pad>'])
optimizer = torch.optim.Adam(transformer.parameters(), lr=learning_rate, betas=(0.9, 0.98), eps=1e-9)

for p in transformer.parameters():
    if p.dim() > 1:
        torch.nn.init.xavier_uniform_(p)


def train_epoch(model: torch.nn.Module):
    model.train()
    tot_loss = 0.0
    for idx, (src, trg) in enumerate(train_iter):
        # 11111
        trg_input = trg[:-1, :]

        src_mask, trg_mask, src_padding_mask, trg_padding_mask = create_mask(src, trg_input)
        output = model(src, trg_input, src_mask, trg_mask, src_padding_mask, trg_padding_mask, src_padding_mask)

        output = output.view(-1, output.shape[-1])
        trg_output = trg[1:, :].view(-1)

        loss = criterion(output, trg_output)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        tot_loss += loss.item()
    return tot_loss / len(train_iter)


def greedy_decode(model, src, src_mask, max_len, start_symbol):
    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long)
    for i in range(max_len - 1):
        tgt_mask = generate_square_subsequent_mask(ys.size(0)).type(torch.bool)
        out = model.decode(ys, memory, tgt_mask)
        out = out.transpose(0, 1)
        prob = model.linear(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.item()

        ys = torch.cat([ys,
                        torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=0)
        if next_word == en_vocab['<eos>']:
            break
    return ys


def translate(model, src, src_vocab, tgt_vocab, src_tokenizer):
    model.eval()
    tokens = [de_vocab['<bos>']] + [src_vocab[tok] for tok in src_tokenizer(src)] + [de_vocab['<eos>']]
    num_tokens = len(tokens)
    src = (torch.LongTensor(tokens).reshape(num_tokens, 1))
    src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool)
    tgt_tokens = greedy_decode(model, src, src_mask, max_len=num_tokens + 5, start_symbol=en_vocab['<bos>']).flatten()
    return " ".join([tgt_vocab.get_itos()[tok] for tok in tgt_tokens]).replace("<bos>", "").replace("<eos>", "")


for epoch in range(1, n_epochs + 1):
    print(f'epoch = {epoch} / {n_epochs}')
    checkpoint = {
        'model': transformer.state_dict(),
        'optimizer': optimizer.state_dict()
    }
    torch.save(checkpoint, 'model.pth')
    print('checkpoint saved')

    print("Eine Gruppe von Menschen steht vor einem Iglu . is translated into: ")
    print(translate(transformer, "Eine Gruppe von Menschen steht vor einem Iglu .", de_vocab, en_vocab, de_tokenizer))

    start_time = time.time()
    losses = train_epoch(transformer)
    end_time = time.time()

    print(f'loss = {losses}, epoch time = {end_time - start_time}')


