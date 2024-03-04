import test1
import torch
import torch.nn as nn
import math


def masked_softmax(X, valid_lens):
    """Perform softmax operation by masking elements on the last axis."""
    # `X`: 3D tensor, `valid_lens`: 1D or 2D tensor
    if valid_lens is None:
        return nn.functional.softmax(X, dim=-1)
    else:
        shape = X.shape
        if valid_lens.dim() == 1:
            valid_lens = torch.repeat_interleave(valid_lens, shape[1])
        else:
            valid_lens = valid_lens.reshape(-1)
        # On the last axis, replace masked elements with a very large negative
        # value, whose exponentiation outputs 0
        X = test1.sequence_mask(X.reshape(-1, shape[-1]), valid_lens,
                              value=-1e6)
        return nn.functional.softmax(X.reshape(shape), dim=-1)


class Decoder(nn.Module):
    """The base decoder interface for the encoder-decoder architecture."""

    def __init__(self, ):
        super(Decoder, self).__init__()

    def init_state(self, enc_outputs, ):
        raise NotImplementedError

    def forward(self, X, state):
        raise NotImplementedError


class EncoderDecoder(nn.Module):
    """The base class for the encoder-decoder architecture."""

    def __init__(self, encoder, decoder):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, enc_X, dec_X):
        enc_outputs = self.encoder(enc_X)
        dec_state = self.decoder.init_state(enc_outputs)
        return self.decoder(dec_X, dec_state)


class AttentionDecoder(Decoder):
    """The base attention-based decoder interface."""

    def __init__(self):
        super(AttentionDecoder, self).__init__()

    @property
    def attention_weights(self):
        raise NotImplementedError


class DecoderBlock(nn.Module):
    """解码器中第i个块"""

    def __init__(self, key_size, query_size, value_size, num_hiddens,
                 norm_shape, ffn_num_input, ffn_num_hiddens, num_heads,
                 dropout, i, ):
        super(DecoderBlock, self).__init__()
        self.i = i
        self.attention1 = test1.MultiHeadAttention(
            key_size, query_size, value_size, num_hiddens, num_heads, dropout)
        self.addnorm1 = test1.AddNorm(norm_shape, dropout)
        self.attention2 = test1.MultiHeadAttention(
            key_size, query_size, value_size, num_hiddens, num_heads, dropout)
        self.addnorm2 = test1.AddNorm(norm_shape, dropout)
        self.ffn = test1.PositionWiseFFN(ffn_num_input, ffn_num_hiddens,
                                         num_hiddens)
        self.addnorm3 = test1.AddNorm(norm_shape, dropout)

    def forward(self, X, state):
        enc_outputs = state[0]
        # 训练阶段，输出序列的所有词元都在同一时间处理，
        # 因此state[2][self.i]初始化为None。
        # 预测阶段，输出序列是通过词元一个接着一个解码的，
        # 因此state[2][self.i]包含着直到当前时间步第i个块解码的输出表示
        if state[1][self.i] is None:
            key_values = X
        else:
            key_values = torch.cat((state[1][self.i], X), axis=1)
        state[1][self.i] = key_values

        # 自注意力
        X2 = self.attention1(X, key_values, key_values)
        Y = self.addnorm1(X, X2)
        # 编码器－解码器注意力。
        Y2 = self.attention2(Y, enc_outputs, enc_outputs)
        Z = self.addnorm2(Y, Y2)
        return self.addnorm3(Z, self.ffn(Z)), state


class TransformerDecoder(AttentionDecoder):
    def __init__(self, key_size, query_size, value_size,
                 num_hiddens, norm_shape, ffn_num_input, ffn_num_hiddens,
                 num_heads, num_layers, dropout, ):
        super(TransformerDecoder, self).__init__()
        self.num_hiddens = num_hiddens
        self.num_layers = num_layers
        self.pos_encoding = test1.PositionalEncoding(num_hiddens, dropout)
        self.blks = nn.Sequential()
        for i in range(num_layers):
            self.blks.add_module("block" + str(i),
                                 DecoderBlock(key_size, query_size, value_size, num_hiddens,
                                              norm_shape, ffn_num_input, ffn_num_hiddens,
                                              num_heads, dropout, i))

    def init_state(self, enc_outputs):
        return [enc_outputs, [None] * self.num_layers]

    def forward(self, X, state):
        X = self.pos_encoding(X * math.sqrt(self.num_hiddens))
        self._attention_weights = [[None] * len(self.blks) for _ in range(2)]
        for i, blk in enumerate(self.blks):
            X, state = blk(X, state)
            # 解码器自注意力权重
            self._attention_weights[0][
                i] = blk.attention1.attention.attention_weights
            # “编码器－解码器”自注意力权重
            self._attention_weights[1][
                i] = blk.attention2.attention.attention_weights
        return X, state

    def attention_weights(self):
        return self._attention_weights


if __name__ == "__main__":
    a = torch.randn(1, 16, 256)  # 数据
    b = torch.randn(1, 16, 256)
    num_hiddens, num_layers, dropout = 256, 1, 0.5
    ffn_num_input, ffn_num_hiddens, num_heads = 256, 128, 16
    key_size, query_size, value_size = 256, 256, 256
    norm_shape = [16, 256]
    encoder = test1.TransformerEncoder(
        key_size=256, query_size=256, value_size=256, num_hiddens=256,
        norm_shape=[16, 256], ffn_num_input=256, ffn_num_hiddens=128, num_heads=16,
        num_layers=1, dropout=0.5)
    decoder = TransformerDecoder(key_size=256, query_size=256, value_size=256, num_hiddens=256,
                                 norm_shape=[16, 256], ffn_num_input=256, ffn_num_hiddens=128, num_heads=16,
                                 num_layers=1, dropout=0.5)

    net = EncoderDecoder(encoder, decoder)
    out1, out2 = net(a, b)
    # out1输出 out2状态列表
    print(out1.shape)
    print(out2)
