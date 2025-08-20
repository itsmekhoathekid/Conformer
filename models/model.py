import torch
import torch.nn as nn
from .encoder import build_encoder
from .decoder import build_decoder
import torch.nn.functional as F

class JointNet(nn.Module):
    def __init__(self, input_size, inner_dim, vocab_size):
        super(JointNet, self).__init__()
        self.forward_layer = nn.Linear(input_size, inner_dim, bias=True)
        self.tanh = nn.Tanh()
        self.project_layer = nn.Linear(inner_dim, vocab_size, bias=True)

    def forward(self, enc_state, dec_state):
        if enc_state.dim() == 3 and dec_state.dim() == 3:
            dec_state = dec_state.unsqueeze(1)
            enc_state = enc_state.unsqueeze(2)

            t = enc_state.size(1)
            u = dec_state.size(2)

            enc_state = enc_state.repeat([1, 1, u, 1])
            dec_state = dec_state.repeat([1, t, 1, 1])
        else:
            assert enc_state.dim() == dec_state.dim()

        concat_state = torch.cat((enc_state, dec_state), dim=-1)
        outputs = self.forward_layer(concat_state)
        outputs = self.tanh(outputs)
        outputs = self.project_layer(outputs)

        return outputs


class ConformerTransducer(nn.Module):
    def __init__(self, config):
        super(ConformerTransducer, self).__init__()
        self.encoder = build_encoder(config)
        self.decoder = build_decoder(config["decoder_params"])

        self.joint = JointNet(
            input_size=config["joint"]["input_size"],
            inner_dim=config["joint"]["inner_size"],
            vocab_size=config["joint"]["vocab_size"]
        )
        self.sos = config["vocab"]["sos"]
        self.eos = config["vocab"]["eos"]
        self.blank = config["vocab"]["blank"]

    def forward(self, inputs, inputs_length, targets, targets_length, training = True):
        enc_state, fbank_len = self.encoder(inputs, inputs_length, training)
        dec_state, _ = self.decoder(targets, targets_length)
        joint_outputs = self.joint(enc_state, dec_state)
        return joint_outputs, fbank_len

    def recognize(self, inputs, inputs_length):
        batch_size = inputs.size(0)

        enc_states, inputs_length = self.encoder(inputs, inputs_length)
        zero_token = torch.LongTensor([[self.sos]]) 

        
        if inputs.is_cuda:
            zero_token = zero_token.cuda()
        def decode(enc_state, lengths):
            token_list = []
            dec_state, hidden = self.decoder(zero_token)
            for t in range(lengths):
                enc_step = enc_states[:, t, :]
                dec_proj = dec_state[:, -1, :]
                logits = self.joint(enc_step, dec_proj)
                logits = F.softmax(logits.squeeze(1).squeeze(1), dim=-1) 
                pred = torch.argmax(logits, dim=-1).item()

                if pred == self.eos: # eos
                    break

                if pred not in [self.eos, self.blank, self.sos]:
                    token_list.append(pred)
                    token = torch.LongTensor([[pred]])
                    if enc_state.is_cuda:
                        token = token.cuda()
                    dec_state, hidden = self.decoder(token, hidden=hidden)

            return token_list

        results = [decode(enc_states[i], inputs_length[i]) for i in range(batch_size)]
        return results