import torch as th
import warnings

warnings.filterwarnings('ignore')
from torch.nn.utils.rnn import PackedSequence, pad_packed_sequence
import torch.nn as nn

class PITONN(th.nn.Module):
    def __init__(self,
                 num_bins = 129,
                 rnn="blstm",
                 num_spks= 2,
                 num_layer= 3,
                 hidden_size = 256,
                 dropout = 0.0,
                 non_linear="relu",
                 bidirectional=True):

        super(PITONN, self).__init__()
        self.num_spks = num_spks

        self.blstm = nn.LSTM(input_size= num_bins, hidden_size=hidden_size,num_layers=num_layer, batch_first=True,
                             dropout=dropout, bidirectional=bidirectional)
        
        ONN_input = 2*hidden_size if bidirectional else hidden_size

        self.ONN_speaker_1 = nn.Linear(ONN_input,ONN_input,bias=False)
        self.ONN_speaker_2 = nn.Linear(ONN_input,ONN_input,bias=False)
        self.drops = th.nn.Dropout(p=dropout)
        self.__set_parameter()

        self.RELU = nn.ReLU()

        self.Linear_1 =  nn.Linear(ONN_input, int(ONN_input / 2) , bias=False)
        self.Linear_2 =  nn.Linear(int(ONN_input / 2), num_bins , bias=False)


    
    def __set_parameter(self):
        nn.init.orthogonal(self.ONN_speaker_1.weight)
        nn.init.orthogonal(self.ONN_speaker_2.weight)

    def forward(self,x,status,per_train = False):
        is_packed = isinstance(x, PackedSequence)
        if not is_packed and x.dim() != 3:
            x = th.unsqueeze(x, 0)
        
        x,_ = self.blstm(x)

        if is_packed:
            x, _ = pad_packed_sequence(x, batch_first=True)
        
        x = self.drops(x)

        speaker_1 =  self.RELU(self.ONN_speaker_1(x))
        speaker_2 =  self.RELU(self.ONN_speaker_2(x))
        if per_train:
            speaker_1 = speaker_1 * status[:,0].reshape(-1,1)
            speaker_2 = speaker_2 * status[:,1].reshape(-1,1)

        mix = speaker_1 + speaker_2

        spk1_spec = self.RELU(self.Linear_2(self.RELU(self.Linear_1(speaker_1))))
        spk2_spec = self.RELU(self.Linear_2(self.RELU(self.Linear_1(speaker_2))))
        mix_spec = self.RELU(self.Linear_2(self.RELU(self.Linear_1(mix))))

        r1 = self.ONN_speaker_1.weight / th.norm(self.ONN_speaker_1.weight)
        r2 = self.ONN_speaker_2.weight / th.norm(self.ONN_speaker_2.weight)

        Orth_const = th.mm(th.t(r1),r2)

        return mix_spec,spk1_spec,spk2_spec,speaker_1,speaker_2,Orth_const