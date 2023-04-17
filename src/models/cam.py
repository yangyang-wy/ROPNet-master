
import torch
import torch.nn as nn

class CAM(nn.Module):
    def __init__(self, C):
        super(CAM, self).__init__()
        self.dim = C   ##这个是什么C= ？
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        b, c, n = x.shape

        out1 = x.view(b, c, -1)  # b,c,n
        out2 = x.view(b, c, -1).permute(0, 2, 1) # b,n,c
        attention_matrix = torch.bmm(out1, out2) # b,c,c
        attention_matrix = self.softmax(torch.max(attention_matrix, -1, keepdim=True)[0].expand_as(attention_matrix) - attention_matrix) # b,c,c

        out3 = x.view(b, c, -1) # b,c,n

        out = torch.bmm(attention_matrix, out3) # b,c,n
        out = self.gamma * out.view(b, c, n) + x

        return out

class PAM(nn.Module):
    def __init__(self, C):
        super(PAM, self).__init__()
        self.dim = C
        self.conv1 = nn.Conv1d(in_channels = C, out_channels=C // 8, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels = C, out_channels=C // 8, kernel_size=1)
        self.conv3 = nn.Conv1d(in_channels = C, out_channels=C, kernel_size=1)

        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self,x):
        b, c, n = x.shape

        out1 = self.conv1(x).view(b, -1, n).permute(0, 2, 1) # b, n, c/latent

        out2 = self.conv2(x).view(b, -1, n) # b,c/latent,n

        attention_matrix = self.softmax(torch.bmm(out1, out2)) # b,n,n

        out3 = self.conv3(x).view(b, -1, n) # b,c,n

        attention = torch.bmm(out3, attention_matrix.permute(0, 2, 1))

        out = self.gamma * attention.view(b, c, n) + x
        return  out

if __name__ == "__main__":

    cbam = CAM(256)
    a=torch.randn(1, 256, 32)
    t = cbam(a)
    print(t.shape)