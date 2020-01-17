from efficientnet_pytorch import EfficientNet
from common import *
from dataset import *

# Gets the GPU if there is one, otherwise the cpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

IMAGE_RGB_MEAN = [0.485, 0.456, 0.406]
IMAGE_RGB_STD = [0.229, 0.224, 0.225]


def criterion(prediction, mask, regr, weight = [1, 1], size_average=True):
    # prediction: mask, 'x', 'y', 'z', 'yaw', 'pitch', 'roll'
    #  prediction[batch_size, 8, 40, 128]
    # Focal loss
    pred_mask = torch.sigmoid(prediction[:, 0]) #[batch_size, 40, 128]
    #     mask_loss = mask * (1 - pred_mask)**2 * torch.log(pred_mask + 1e-12) + (1 - mask) * pred_mask**2 * torch.log(1 - pred_mask + 1e-12)
    # mask_loss = mask * torch.log(pred_mask + 1e-12) + (1 - mask) * torch.log(1 - pred_mask + 1e-12)  # [4, 40, 128] cross engtropy
    # mask_loss = nn.BCEWithLogitsLoss()(pred_mask, mask)
    # mask_loss = -mask_loss.mean(0).sum() # sum of batch loss
    mask_loss = FocalLoss()(pred_mask, mask)

    # Regression L1 loss
    pred_regr = prediction[:, 1:] # [batch_size, 7, 40, 128]
    regr_loss = (torch.abs(pred_regr - regr).sum(1) * mask).sum(1).sum(1) / mask.sum(1).sum(1)  # regr_loss per pixel in mask. [batch_size]
    regr_loss = regr_loss.mean(0)

    # Sum
    loss = (mask_loss * weight[0] + regr_loss * weight[1]) / (weight[0] + weight[1])
    if not size_average:
        loss *= prediction.shape[0]
    return loss, mask_loss, regr_loss

def metric_hit(logit, truth, threshold=0.5):
    num_class = 1
    batch_size, H, W = logit.shape

    with torch.no_grad():
        logit = logit.view(batch_size, num_class, -1)
        truth = truth.view(batch_size, -1)

        probability = torch.softmax(logit, 1)
        p = torch.max(probability, 1)[1]
        t = truth.long()
        correct = (p == t)

        index0 = t == 0
        index1 = t == 1

        num_neg = index0.sum().item()
        num_pos1 = index1.sum().item()

        neg = correct[index0].sum().item() / (num_neg + 1e-12)
        pos1 = correct[index1].sum().item() / (num_pos1 + 1e-12)

        num_pos = [num_pos1]
        tn = neg
        tp = [pos1]

    return tn, tp, num_neg, num_pos

class double_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''

    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True):
        super(up, self).__init__()

        #  would be a nice idea if the upsampling could be learned too,
        #  but my machine do not have enough memory to handle all those weights
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, 2, stride=2)

        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x1, x2=None):
        x1 = self.up(x1)

        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2))

        # for padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd

        if x2 is not None:
            x = torch.cat([x2, x1], dim=1)
        else:
            x = x1
        x = self.conv(x)
        return x


def get_mesh(batch_size, shape_x, shape_y):
    mg_x, mg_y = np.meshgrid(np.linspace(0, 1, shape_y), np.linspace(0, 1, shape_x))
    mg_x = np.tile(mg_x[None, None, :, :], [batch_size, 1, 1, 1]).astype('float32')
    mg_y = np.tile(mg_y[None, None, :, :], [batch_size, 1, 1, 1]).astype('float32')
    mesh = torch.cat([torch.tensor(mg_x).to(device), torch.tensor(mg_y).to(device)], 1)
    return mesh

class MyUNet(nn.Module):
    '''Mixture of previous classes'''

    def __init__(self, n_classes):
        super(MyUNet, self).__init__()
        self.base_model = EfficientNet.from_pretrained('efficientnet-b0')

        self.conv0 = double_conv(5, 64)
        self.conv1 = double_conv(64, 128)
        self.conv2 = double_conv(128, 512)
        self.conv3 = double_conv(512, 1024)

        self.mp = nn.MaxPool2d(2)

        self.up1 = up(1282 + 1024, 512)
        self.up2 = up(512 + 512, 256)
        self.outc = nn.Conv2d(256, n_classes, 1)

    def forward(self, x):
        batch_size = x.shape[0]
        mesh1 = get_mesh(batch_size, x.shape[2], x.shape[3])
        x0 = torch.cat([x, mesh1], 1)
        x1 = self.mp(self.conv0(x0))
        x2 = self.mp(self.conv1(x1))
        x3 = self.mp(self.conv2(x2))
        x4 = self.mp(self.conv3(x3))

        x_center = x[:, :, :, IMG_WIDTH // 8: -IMG_WIDTH // 8]
        feats = self.base_model.extract_features(x_center)
        bg = torch.zeros([feats.shape[0], feats.shape[1], feats.shape[2], feats.shape[3] // 8]).to(device)
        feats = torch.cat([bg, feats, bg], 3)

        # Add positional info
        mesh2 = get_mesh(batch_size, feats.shape[2], feats.shape[3])
        feats = torch.cat([feats, mesh2], 1)

        x = self.up1(feats, x4)
        x = self.up2(x, x3)
        x = self.outc(x)
        return x

##############################################################################################
def make_dummy_data(folder='1024x320', batch_size=8):
    image_file = glob.glob(os.path.join(config.data_dir, 'dump/%s/image/*.png' % folder))  # 32
    image_file = sorted(image_file)

    input = []
    truth_mask = []
    truth_label = []
    for b in range(0, batch_size):
        i = b % len(image_file)
        image = cv2.imread(image_file[i], cv2.IMREAD_COLOR)
        mask = np.load(image_file[i].replace('/image/', '/mask/').replace('.png', '.npy'))
        label = (mask.reshape(4, -1).sum(1) > 0).astype(np.int32)

        num_class, H, W = mask.shape
        mask = mask.transpose(1, 2, 0) * [1, 2, 3, 4]
        mask = mask.reshape(-1, 4)
        mask = mask.max(-1).reshape(1, H, W)

        input.append(image)
        truth_mask.append(mask)
        truth_label.append(label)

    input = np.array(input)
    input = image_to_input(input, IMAGE_RGB_MEAN, IMAGE_RGB_STD)

    truth_mask = np.array(truth_mask)
    truth_label = np.array(truth_label)

    infor = None

    return input, truth_mask, truth_label, infor

def run_check_net():
    batch_size = 1
    C, H, W = 3, IMG_HEIGHT, IMG_WIDTH

    input = np.random.uniform(-1, 1, (batch_size, C, H, W))
    input = torch.from_numpy(input).float().cuda()

    net = MyUNet(n_classes = 34).cuda()
    net.eval()

    with torch.no_grad():
        logit = net(input)

    print('')
    print('input: ', input.shape)
    print('logit: ', logit.shape)
    print(net)

def run_check_train():
    loss_weight = [5, 5, 2, 5]
    if 1:
        input, truth_mask, truth_label, infor = make_dummy_data(folder='256x256', batch_size=2)
        batch_size, C, H, W = input.shape

        print('input shape:{}'.format(input.shape))
        print("truth label shape: {}".format(truth_label.shape))
        print("truth mask shape: {}".format(truth_mask.shape))
        print("truth label.sum :{}".format(truth_label.sum(0)))

    # ---
    truth_mask = torch.from_numpy(truth_mask).long().cuda()
    truth_label = torch.from_numpy(truth_label).float().cuda()
    input = torch.from_numpy(input).float().cuda()

    net = MyUNet(config.num_classes).cuda()
    net = net.eval()

    with torch.no_grad():
        logit = net(input)
        loss = criterion(logit, truth_mask)
        # tn, tp, num_neg, num_pos = metric_hit(logit, truth_mask)
        # dn, dp, num_neg, num_pos = metric_dice(logit, truth_mask)

        print('loss = %0.5f' % loss.item())
        # print('tn,tp = %0.5f, [%0.5f,%0.5f,%0.5f,%0.5f] ' % (tn, tp[0], tp[1], tp[2], tp[3]))
        # print('dn,dp = %0.5f, [%0.5f,%0.5f,%0.5f,%0.5f] ' % (dn, dp[0], dp[1], dp[2], dp[3]))
        # print('num_pos,num_neg = %d, [%d,%d,%d,%d] ' % (num_neg, num_pos[0], num_pos[1], num_pos[2], num_pos[3]))
        print('')

    # exit(0)
    # dummy sgd to see if it can converge ...
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()),
                          lr=0.1, momentum=0.9, weight_decay=0.0001)

    # optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()),lr=0.001)

    print('batch_size =', batch_size)
    print('----------------------------------------------------------------------')
    print('[iter ]  loss     |  tn, [tp1,tp2,tp3,tp4]  |  dn, [dp1,dp2,dp3,dp4]  ')
    print('----------------------------------------------------------------------')
    # [00000]  0.70383  | 0.00000, 0.46449

    i = 0
    optimizer.zero_grad()
    while i <= 200:

        net.train()
        optimizer.zero_grad()

        logit = net(input)
        loss = criterion(logit, truth_mask, loss_weight)
        # tn, tp, num_neg, num_pos = metric_hit(logit, truth_mask)
        # dn, dp, num_neg, num_pos = metric_dice(logit, truth_mask)

        (loss).backward()
        optimizer.step()

        if i % 10 == 0:
            print('[%05d] %8.5f  | %0.5f, [%0.5f,%0.5f,%0.5f,%0.5f]  | %0.5f, [%0.5f,%0.5f,%0.5f,%0.5f] ' % (
                i,
                loss.item(),
                tn, tp[0], tp[1], tp[2], tp[3],
                dn, dp[0], dp[1], dp[2], dp[3],
            ))
        i = i + 1
    print('')

    if 1:
        # net.eval()
        logit = net(input)
        probability = torch.softmax(logit, 1)
        probability = one_hot_encode_predict(probability)
        truth_mask = one_hot_encode_truth(truth_mask)

        probability_mask = probability.data.cpu().numpy()
        truth_label = truth_label.data.cpu().numpy()
        truth_mask = truth_mask.data.cpu().numpy()
        image = input_to_image(input, IMAGE_RGB_MEAN, IMAGE_RGB_STD)

        for b in range(batch_size):
            print('%2d ------ ' % (b))
            result = draw_predict_result(image[b], truth_mask[b], truth_label[b], probability_mask[b])
            image_show('result', result, resize=0.5)
            cv2.waitKey(0)

# main #################################################################
if __name__ == '__main__':
    print('%s: calling main function ... ' % os.path.basename(__file__))

    # run_check_net()
    run_check_train()

    print('\nsucess!')
