import torch
import torch.nn as nn
from network_module import *
import torch.nn.functional as F
import math
import sys
# ----------------------------------------
#         Initialize the networks
# ----------------------------------------
def weights_init(net, init_type = 'normal', init_gain = 0.02):
    """Initialize network weights.
    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal
    In our paper, we choose the default setting: zero mean Gaussian distribution with a standard deviation of 0.02
    """
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and classname.find('Conv') != -1:
            if init_type == 'normal':
                torch.nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                torch.nn.init.xavier_normal_(m.weight.data, gain = init_gain)
            elif init_type == 'kaiming':
                torch.nn.init.kaiming_normal_(m.weight.data, a = 0, mode = 'fan_in')
            elif init_type == 'orthogonal':
                torch.nn.init.orthogonal_(m.weight.data, gain = init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
        elif classname.find('BatchNorm2d') != -1:
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(m.bias.data, 0.0)

    # apply the initialization function <init_func>
    print('initialize network with %s type' % init_type)
    net.apply(init_func)
def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class GeLU(nn.Module):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return gelu(x)


def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": gelu, "relu": torch.nn.functional.relu, "swish": swish}
BertLayerNorm = torch.nn.LayerNorm

class OrderEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings.
    """
    def __init__(self, opt):
        super(OrderEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(opt.max_position_embeddings, opt.hidden_size, padding_idx=1)
        self.position_embeddings = nn.Embedding(opt.max_position_embeddings, opt.hidden_size, padding_idx=1)
        self.token_type_embeddings = nn.Embedding(opt.max_position_embeddings, opt.hidden_size, padding_idx=0)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = BertLayerNorm(opt.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(opt.hidden_dropout_prob)

    def forward(self, input_ids, token_type_ids=None):
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = words_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

class ImageEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings.
    """
    def __init__(self, opt):
        super(ImageEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(64, opt.hidden_size, padding_idx=1)
        self.position_embeddings = nn.Embedding(64, opt.hidden_size, padding_idx=1)
        self.token_type_embeddings = nn.Embedding(64, opt.hidden_size, padding_idx=1)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = BertLayerNorm(opt.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(opt.hidden_dropout_prob)

    def forward(self, input_ids, token_type_ids=1):
        seq_length = input_ids.size(2)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.expand(8,64)

        token_type_ids = torch.ones(8,64).long().cuda()
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = input_ids.permute(0,2,1) + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

class BertAttention(nn.Module):
    def __init__(self, opt, ctx_dim=None):
        super().__init__()
        if opt.hidden_size % opt.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (opt.hidden_size, opt.num_attention_heads))
        self.num_attention_heads = opt.num_attention_heads
        self.attention_head_size = int(opt.hidden_size / opt.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # visual_dim = 2048
        if ctx_dim is None:
            ctx_dim =opt.hidden_size
        self.query = nn.Linear(opt.hidden_size, self.all_head_size)
        self.key = nn.Linear(ctx_dim, self.all_head_size)
        self.value = nn.Linear(ctx_dim, self.all_head_size)

        self.dropout = nn.Dropout(opt.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, context, attention_mask=None):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(context)
        mixed_value_layer = self.value(context)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer

class BertAttOutput(nn.Module):
    def __init__(self, opt):
        super(BertAttOutput, self).__init__()
        self.dense = nn.Linear(opt.hidden_size, opt.hidden_size)
        self.LayerNorm = BertLayerNorm(opt.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(opt.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

class BertSelfattLayer(nn.Module):
    def __init__(self, opt):
        super(BertSelfattLayer, self).__init__()
        self.self = BertAttention(opt)
        self.output = BertAttOutput(opt)

    def forward(self, input_tensor, attention_mask):
        # Self attention attends to itself, thus keys and querys are the same (input_tensor).
        self_output = self.self(input_tensor, input_tensor, attention_mask)
        attention_output = self.output(self_output, input_tensor)
        return attention_output

class BertIntermediate(nn.Module):
    def __init__(self, opt):
        super(BertIntermediate, self).__init__()
        self.dense = nn.Linear(opt.hidden_size, opt.intermediate_size)
        if isinstance(opt.hidden_act, str) or (sys.version_info[0] == 2 and isinstance(opt.hidden_act, unicode)):
            self.intermediate_act_fn = ACT2FN[opt.hidden_act]
        else:
            self.intermediate_act_fn = opt.hidden_act

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states

class BertOutput(nn.Module):
    def __init__(self, opt):
        super(BertOutput, self).__init__()
        self.dense = nn.Linear(opt.intermediate_size, opt.hidden_size)
        self.LayerNorm = BertLayerNorm(opt.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(opt.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

class BertLayer(nn.Module):
    def __init__(self, opt):
        super(BertLayer, self).__init__()
        self.attention = BertSelfattLayer(opt)
        self.intermediate = BertIntermediate(opt)
        self.output = BertOutput(opt)

    def forward(self, hidden_states, attention_mask):
        attention_output = self.attention(hidden_states, attention_mask)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output

class BertCrossattLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.att = BertAttention(config)
        self.output = BertAttOutput(config)

    def forward(self, input_tensor, ctx_tensor, ctx_att_mask=None):
        output = self.att(input_tensor, ctx_tensor, ctx_att_mask)
        attention_output = self.output(output, input_tensor)
        return attention_output

class LXRTXLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        # The cross-attention Layer
        self.visual_attention = BertCrossattLayer(config)

        # Self-attention Layers
        self.lang_self_att = BertSelfattLayer(config)
        self.visn_self_att = BertSelfattLayer(config)

        # Intermediate and Output Layers (FFNs)
        self.lang_inter = BertIntermediate(config)
        self.lang_output = BertOutput(config)
        self.visn_inter = BertIntermediate(config)
        self.visn_output = BertOutput(config)

    def cross_att(self, lang_input, lang_attention_mask, visn_input, visn_attention_mask):
        # Cross Attention
        lang_att_output = self.visual_attention(lang_input, visn_input, ctx_att_mask=visn_attention_mask)
        visn_att_output = self.visual_attention(visn_input, lang_input, ctx_att_mask=lang_attention_mask)
        return lang_att_output, visn_att_output

    def self_att(self, lang_input, lang_attention_mask, visn_input, visn_attention_mask):
        # Self Attention
        lang_att_output = self.lang_self_att(lang_input, lang_attention_mask)
        visn_att_output = self.visn_self_att(visn_input, visn_attention_mask)
        return lang_att_output, visn_att_output

    def output_fc(self, lang_input, visn_input):
        # FC layers
        lang_inter_output = self.lang_inter(lang_input)
        visn_inter_output = self.visn_inter(visn_input)

        # Layer output
        lang_output = self.lang_output(lang_inter_output, lang_input)
        visn_output = self.visn_output(visn_inter_output, visn_input)
        return lang_output, visn_output

    def forward(self, lang_feats, lang_attention_mask,
                      visn_feats, visn_attention_mask):
        lang_att_output = lang_feats
        visn_att_output = visn_feats

        lang_att_output, visn_att_output = self.cross_att(lang_att_output, lang_attention_mask,
                                                          visn_att_output, visn_attention_mask)
        lang_att_output, visn_att_output = self.self_att(lang_att_output, lang_attention_mask,
                                                         visn_att_output, visn_attention_mask)
        lang_output, visn_output = self.output_fc(lang_att_output, visn_att_output)

        return lang_output, visn_output
# ----------------------------------------
#               Discriminator
# ----------------------------------------
# PatchDiscriminator70: PatchGAN discriminator for Pix2Pix
# Usage: Initialize PatchGAN in training code like:
#        discriminator = PatchDiscriminator70()
# This is a kind of PatchGAN. Patch is implied in the output. This is 70 * 70 PatchGAN
class PatchDiscriminator70(nn.Module):
    def __init__(self, opt):
        super(PatchDiscriminator70, self).__init__()
        # Down sampling
        self.block1 = Conv2dLayer(opt.in_channels + opt.out_channels, opt.start_channels, 3, 2, 1, pad_type = opt.pad, activation = opt.activ_d, norm = 'none', sn = True)
        self.block2 = Conv2dLayer(opt.start_channels, opt.start_channels * 2, 3, 2, 1, pad_type = opt.pad, activation = opt.activ_d, norm = opt.norm, sn = True)
        self.block3 = Conv2dLayer(opt.start_channels * 2, opt.start_channels * 4, 3, 2, 1, pad_type = opt.pad, activation = opt.activ_d, norm = opt.norm, sn = True)
        # Final output, implemention of 70 * 70 PatchGAN
        self.final1 = Conv2dLayer(opt.start_channels * 4, opt.start_channels * 8, 4, 1, 1, pad_type = opt.pad, activation = opt.activ_d, norm = opt.norm, sn = True)
        self.final2 = Conv2dLayer(opt.start_channels * 8, 1, 4, 1, 1, pad_type = opt.pad, activation = 'none', norm = 'none', sn = True)

    def forward(self, img_A, img_B):
        # Concatenate image and condition image by channels to produce input
        # img_A: grayscale input; img_B: ab embedding output
        x = torch.cat((img_A, img_B), 1)                        # out: batch * 3 * 256 * 256
        x = self.block1(x)                                      # out: batch * 64 * 128 * 128
        x = self.block2(x)                                      # out: batch * 128 * 64 * 64
        x = self.block3(x)                                      # out: batch * 256 * 32 * 32
        x = self.final1(x)                                      # out: batch * 512 * 31 * 31
        x = self.final2(x)                                      # out: batch * 1 * 30 * 30
        return x


class xixi_recon(nn.Module):
    def __init__(self, opt):
        super(xixi_recon, self).__init__()
        # The generator is U shaped
        # It means: input -> downsample -> upsample -> output
        # Encoder
        self.E1 = Conv2dLayer(opt.in_channels, opt.start_channels, 7, 1, 3, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm)
        self.E2 = Conv2dLayer(opt.start_channels, opt.start_channels * 2, 3, 2, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm)
        self.E3 = Conv2dLayer(opt.start_channels * 2, opt.start_channels * 4, 3, 2, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm)
        self.E4 = Conv2dLayer(opt.start_channels * 4, opt.start_channels * 4, 3, 2, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm)
        self.E5 = Conv2dLayer(opt.start_channels * 4, opt.start_channels * 4, 3, 2, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm)
        self.E6 = Conv2dLayer(opt.start_channels * 4, opt.start_channels * 4, 3, 2, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm)
        self.E7 = Conv2dLayer(opt.start_channels * 4, opt.start_channels * 4, 3, 2, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm)
        self.E8 = Conv2dLayer(opt.start_channels * 4, opt.start_channels * 4, 3, 4, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm)
        # Bottleneck
        # self.T1 = ResConv2dLayer(opt.start_channels * 4, opt.start_channels * 4, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm)
        # self.T2 = ResConv2dLayer(opt.start_channels * 4, opt.start_channels * 4, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm)
        # self.T3 = ResConv2dLayer(opt.start_channels * 4, opt.start_channels * 4, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm)
        # self.T4 = ResConv2dLayer(opt.start_channels * 4, opt.start_channels * 4, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm)
        # self.T5 = ResConv2dLayer(opt.start_channels * 4, opt.start_channels * 4, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm)
        # self.T6 = ResConv2dLayer(opt.start_channels * 4, opt.start_channels * 4, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm)
        # self.T7 = ResConv2dLayer(opt.start_channels * 4, opt.start_channels * 4, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm)
        # self.T8 = ResConv2dLayer(opt.start_channels * 4, opt.start_channels * 4, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm)
        # self.T9 = ResConv2dLayer(opt.start_channels * 4, opt.start_channels * 4, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm)
        # Decoder
        self.D1 = TransposeConv2dLayer(opt.start_channels * 4, opt.start_channels * 4, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm, scale_factor = 4)
        self.D2 = TransposeConv2dLayer(opt.start_channels * 4, opt.start_channels * 4, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm, scale_factor = 2)
        self.D3 = TransposeConv2dLayer(opt.start_channels * 4, opt.start_channels * 4, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm, scale_factor = 2)
        self.D4 = TransposeConv2dLayer(opt.start_channels * 4, opt.start_channels * 4, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm, scale_factor = 2)
        self.D5 = TransposeConv2dLayer(opt.start_channels * 4, opt.start_channels * 4, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm, scale_factor = 2)
        # self.D6 = TransposeConv2dLayer(opt.start_channels * 4, opt.start_channels * 4, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm, scale_factor = 2)
        self.D6 = TransposeConv2dLayer(opt.start_channels * 4, opt.start_channels * 2, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm, scale_factor = 2)
        self.D7 = TransposeConv2dLayer(opt.start_channels * 2, opt.start_channels, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm, scale_factor = 2)
        self.D8 = Conv2dLayer(opt.start_channels, opt.out_channels, 3, 1, 1, pad_type = opt.pad, norm = 'none', activation = 'none')

    def forward(self, x):

        # print(x.shape)
        x = self.E1(x)                                          # out: batch * 64 * 256 * 256
        x = self.E2(x)                                          # out: batch * 128 * 128 * 128
        x = self.E3(x)                                          # out: batch * 256 * 64 * 64
        x = self.E4(x)                                          # out: batch *
        x = self.E5(x)                                          # out: batch * 
        # x = self.E6(x)                                          # out: batch *
        # x = self.E7(x)                                          # out: batch * 
        # x = self.E8(x)                                          # out: batch * 
        # print(x.shape)
        # x = self.D1(x)                                          # out: batch * 
        # x = self.D2(x)                                          # out: batch * 
        # x = self.D3(x)                                      
        x = self.D4(x)                                         
        
        x = self.D5(x)                                        
        x = self.D6(x)                                          
        x = self.D7(x)
        out = self.D8(x)                                         

        return out

class MyDNN(nn.Module):
    def __init__(self, opt):
        super(MyDNN, self).__init__()
        # The generator is U shaped
        # It means: input -> downsample -> upsample -> output
        # Encoder
        self.E1 = Conv2dLayer(opt.in_channels, opt.start_channels, 7, 1, 3, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm)
        self.E2 = Conv2dLayer(opt.start_channels, opt.start_channels * 2, 3, 2, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm)
        self.E3 = Conv2dLayer(opt.start_channels * 2, opt.start_channels * 4, 3, 2, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm)
        self.E4 = Conv2dLayer(opt.start_channels * 4, opt.start_channels * 4, 3, 2, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm)
        self.E5 = Conv2dLayer(opt.start_channels * 4, opt.start_channels * 4, 3, 2, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm)
        self.E6 = Conv2dLayer(opt.start_channels * 4, opt.start_channels * 8, 3, 2, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm)
        self.E7 = Conv2dLayer(opt.start_channels * 8, opt.start_channels * 8, 3, 2, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm)
        # self.E8 = Conv2dLayer(opt.start_channels * 16, opt.start_channels * 32, 3, 4, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm)
        # Bottleneck
        # self.T1 = ResConv2dLayer(opt.start_channels * 4, opt.start_channels * 4, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm)
        # self.T2 = ResConv2dLayer(opt.start_channels * 4, opt.start_channels * 4, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm)
        # self.T3 = ResConv2dLayer(opt.start_channels * 4, opt.start_channels * 4, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm)
        # self.T4 = ResConv2dLayer(opt.start_channels * 4, opt.start_channels * 4, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm)
        # self.T5 = ResConv2dLayer(opt.start_channels * 4, opt.start_channels * 4, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm)
        # self.T6 = ResConv2dLayer(opt.start_channels * 4, opt.start_channels * 4, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm)
        # self.T7 = ResConv2dLayer(opt.start_channels * 4, opt.start_channels * 4, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm)
        # self.T8 = ResConv2dLayer(opt.start_channels * 4, opt.start_channels * 4, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm)
        # self.T9 = ResConv2dLayer(opt.start_channels * 4, opt.start_channels * 4, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm)
        # Decoder
        # self.D1 = TransposeConv2dLayer(opt.start_channels * 32, opt.start_channels * 16, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm, scale_factor = 4)
        self.D2 = TransposeConv2dLayer(opt.start_channels * 8, opt.start_channels * 8, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm, scale_factor = 2)
        self.D3 = TransposeConv2dLayer(opt.start_channels * 8, opt.start_channels * 4, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm, scale_factor = 2)
        self.D4 = TransposeConv2dLayer(opt.start_channels * 4, opt.start_channels * 4, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm, scale_factor = 2)
        self.D5 = TransposeConv2dLayer(opt.start_channels * 4, opt.start_channels * 4, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm, scale_factor = 2)
        # self.D6 = TransposeConv2dLayer(opt.start_channels * 4, opt.start_channels * 4, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm, scale_factor = 2)
        self.D6 = TransposeConv2dLayer(opt.start_channels * 4, opt.start_channels * 2, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm, scale_factor = 2)
        self.D7 = TransposeConv2dLayer(opt.start_channels * 2, opt.start_channels, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm, scale_factor = 2)
        self.D8 = Conv2dLayer(opt.start_channels, opt.out_channels, 3, 1, 1, pad_type = opt.pad, norm = 'none', activation = 'none')

    def forward(self, x):

        # print(x.shape)
        x = self.E1(x)                                          # out: batch * 64 * 256 * 256
        x = self.E2(x)                                          # out: batch * 128 * 128 * 128
        x = self.E3(x)                                          # out: batch * 256 * 64 * 64
        x = self.E4(x)                                          # out: batch *
        x = self.E5(x)                                          # out: batch * 
        x = self.E6(x)                                          # out: batch *
        x = self.E7(x)                                          # out: batch * 
        # x = self.E8(x)                                          # out: batch * 
        print(x.shape)
        # x = self.D1(x)                                          # out: batch * 
        x = self.D2(x)                                          # out: batch * 
        x = self.D3(x)                                      
        x = self.D4(x)                                         
        
        x = self.D5(x)                                        
        x = self.D6(x)                                          
        x = self.D7(x)
        out = self.D8(x)                                         

        return out

class LXMERT(nn.Module):
    def __init__(self, opt):
        super(LXMERT, self).__init__()
        # The generator is U shaped
        # Encoder
        self.E1 = Conv2dLayer(in_channels = 3,  out_channels = opt.start_channels, kernel_size=3, stride = 1, padding = 1, dilation = 1,  pad_type = opt.pad, activation = 'relu', norm = opt.norm)
        self.E2 = Conv2dLayer(in_channels = opt.start_channels, out_channels = 2*opt.start_channels, kernel_size=3, stride = 2, padding = 1, dilation = 1,  pad_type = opt.pad, activation = 'relu',norm = opt.norm)
        self.E3 = Conv2dLayer(in_channels = 2*opt.start_channels, out_channels = 4*opt.start_channels, kernel_size=3, stride = 2, padding = 1, dilation = 1,  pad_type = opt.pad, activation = 'relu',norm = opt.norm)
        self.E4 = Conv2dLayer(in_channels = 4*opt.start_channels, out_channels = 4*opt.start_channels, kernel_size=3, stride = 2, padding = 1, dilation = 1,  pad_type = opt.pad, activation = 'relu',norm = opt.norm)
        self.E5 = Conv2dLayer(in_channels = 4*opt.start_channels, out_channels = 4*opt.start_channels, kernel_size=3, stride = 2, padding = 1, dilation = 1,  pad_type = opt.pad, activation = 'relu',norm = opt.norm)
        self.E6 = Conv2dLayer(in_channels = 4*opt.start_channels, out_channels = 4*opt.start_channels, kernel_size=3, stride = 2, padding = 1, dilation = 1,  pad_type = opt.pad, activation = 'relu',norm = opt.norm)
        # self.E7 = Conv2dLayer(in_channels = 4*opt.start_channels, out_channels = 4*opt.start_channels, kernel_size=3, stride = 2, padding = 1, dilation = 1,  pad_type = opt.pad, activation = 'relu',norm = opt.norm)
        # self.E8 = Conv2dLayer(in_channels = 4*opt.start_channels, out_channels = 4*opt.start_channels, kernel_size=4, stride = 1, padding = 0, dilation = 1,  pad_type = opt.pad, activation = 'relu',norm = opt.norm)

        self.stroke_embedding =  OrderEmbeddings(opt)
        self.img_embedding =  ImageEmbeddings(opt)

        # Layers
        # Using self.layer instead of self.l_layer to support loading BERT weights.
        self.layer = nn.ModuleList(
            [BertLayer(opt) for _ in range(2)]
        )
        self.x_layers = nn.ModuleList(
            [LXRTXLayer(opt) for _ in range(3)]
        )
        self.r_layers = nn.ModuleList(
            [BertLayer(opt) for _ in range(2)]
        )
        # Decoder
        self.D1 = TransposeConv2dLayer(opt.start_channels * 4, opt.start_channels * 4, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm, scale_factor = 4)
        self.D2 = TransposeConv2dLayer(opt.start_channels * 4, opt.start_channels * 4, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm, scale_factor = 2)
        # self.D3 = TransposeConv2dLayer(opt.start_channels * 4, opt.start_channels * 4, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm, scale_factor = 2)
        self.D4 = TransposeConv2dLayer(opt.start_channels * 4, opt.start_channels * 2, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm, scale_factor = 2)
        self.D5 = TransposeConv2dLayer(opt.start_channels * 2, opt.start_channels, 3, 1, 1, pad_type = opt.pad, activation = opt.activ_g, norm = opt.norm, scale_factor = 2)
        self.D6 = Conv2dLayer(opt.start_channels, opt.out_channels, 3, 1, 1, pad_type = opt.pad, norm = 'none', activation = 'none')
    def forward(self, x,stroke_order):
        x = self.E1(x)
        x = self.E2(x)
        x = self.E3(x)
        x = self.E4(x)
        x = self.E5(x)
        x = self.E6(x)
        x = torch.flatten(x, start_dim=2, end_dim=3)
        
        img_embedding = self.img_embedding(x)
        stroke_embedding = self.stroke_embedding(stroke_order)
        # Run stroke layers
        for layer_module in self.layer:
            stroke_feats = layer_module(stroke_embedding, attention_mask=None)

        # Run relational layers
        for layer_module in self.r_layers:
            img_feats = layer_module(img_embedding, attention_mask=None)

        # Run cross-modality layers
        for layer_module in self.x_layers:
            stroke_feats, img_feats = layer_module(stroke_feats, None,img_feats, visn_attention_mask=None)
            
        # print(img_feats.shape)
        x = img_feats.permute(0,2,1)
        x = x.reshape(8,512,8,8)
        x = self.D1(x)
        x = self.D2(x)
        # x = self.D3(x)
        x = self.D4(x)
        x = self.D5(x)
        x = self.D6(x)
        return x