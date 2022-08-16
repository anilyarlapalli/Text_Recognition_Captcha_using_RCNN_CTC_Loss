from unicodedata import bidirectional
import torch
import torchvision
from torch import dropout, nn 
from torch.nn import functional as F
from torchvision import models

class TextRecognition(nn.Module):
    def __init__(self, num_chars):
        super(TextRecognition, self).__init__()
        self.conv1 = nn.Conv2d(3, 128, kernel_size=(2,2), padding=(1,1))
        self.maxpool1 = nn.AvgPool2d(kernel_size=(2,2))
        self.conv2 = nn.Conv2d(128, 64, kernel_size=(2,2), padding=(1,1))
        self.maxpool2 = nn.AvgPool2d(kernel_size=(2,2))

        self.linear1 = nn.Linear(4800, 128)
        self.drop1 = nn.Dropout(0.2)

        self.gru = nn.GRU(128, 64, bidirectional = True, num_layers = 2, )
        self.output = nn.Linear(128, num_chars+1)

    def forward(self, images, targets = None):
        bs, c, h, w = images.size()
        # print(bs, c, h, w)
        x = F.relu(self.conv1(images))
        # print(x.size())
        x = self.maxpool1(x)
        # print(x.size())

        x = F.relu(self.conv2(x))
        # print(x.size())
        x = self.maxpool2(x)
        # print(x.size())  # ([1, 64, 18, 75])

        #Modifying input to feed it to RNN
        x = x.permute(0, 3, 1, 2) # ([1, 75, 64, 18])
        # print(x.size())

        x = x.view(bs, x.size(1), -1) # (N - Batch size,L-sequence length,D∗H out) when batch_first=True 
        # print(x.size()) # ([1, 75, 1152])

        x = self.linear1(x)
        x = self.drop1(x)
        # print(x.size()) # ([1, 75, 64])

        x, _ = self.gru(x)
        # print(x.size()) # ([1, 75, 64])

        x = self.output(x)
        # print(x.size()) # ([6, 75, 109])

        x = x.permute(1, 0, 2)
        # print(x.size()) # ([75, 6, 109])

        if targets is not None:
            log_softmax_values = F.log_softmax(x, 2)

            input_lengths = torch.full(
                    size=(bs,), fill_value=log_softmax_values.size(0), dtype = torch.int32
            )
            # print(input_lengths) # [75, 75, 75, 75, 75, 75]

            target_lengths = torch.full(
                    size=(bs,), fill_value=targets.size(1), dtype = torch.int32
            )
            # print(target_lengths) # [16, 16, 16, 16, 16, 16]

            loss = nn.CTCLoss(blank=0)(log_softmax_values, targets, input_lengths, target_lengths)

            return x, loss
        return x, None


if __name__ == "__main__":
    tr = TextRecognition(108)
    img = torch.rand(6, 3, 75, 300)
    target = torch.randint(1, 20, (6, 16))
    x, loss = tr(img, target)

