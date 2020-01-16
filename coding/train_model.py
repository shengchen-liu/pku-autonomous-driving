import os, sys
from common import *
from dataset import *
from efficientnet import *

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


# Root directory of the project : coding
ROOT_DIR = os.path.abspath("../")
CURRENT_DIR = os.path.abspath(".")
NUM_CLASSES = config.num_classes

# Import
sys.path.append(ROOT_DIR)  # To find local version of the library
sys.path.append(CURRENT_DIR) # current path



### loss ###################################################################

def one_hot_encode_truth(truth, num_class=NUM_CLASSES):
    one_hot = truth.repeat(1, num_class, 1, 1)
    arange = torch.arange(1, num_class + 1).view(1, num_class, 1, 1).to(truth.device) #with values from the interval [start, end)
    one_hot = (one_hot == arange).float()
    return one_hot


def one_hot_encode_predict(predict, num_class=NUM_CLASSES):
    value, index = torch.max(predict, 1, keepdim=True)

    value = value.repeat(1, num_class, 1, 1)
    index = index.repeat(1, num_class, 1, 1)
    arange = torch.arange(1, num_class + 1).view(1, num_class, 1, 1).to(predict.device)

    one_hot = (index == arange).float()
    value = value * one_hot
    return value

def run_train():
    out_dir = os.path.join(config.results, config.model_name)

    if config.initial_checkpoint is not None:
        initial_checkpoint = os.path.join(config.results, config.model_name, 'checkpoint', config.initial_checkpoint)
    else:
        initial_checkpoint = None

    schduler = NullScheduler(lr=0.001)
    batch_size = config.batch_size  # 8
    iter_accum = 4

    # loss_weight = [0.2] + [1.0] * (len(classes)-1)  #
    train_sampler = RandomSampler  # RandomSampler, FourBalanceClassSampler

    ## setup  -----------------------------------------------------------------------------
    for f in ['checkpoint', 'train', 'valid', 'backup']: os.makedirs(out_dir + '/' + f, exist_ok=True)
    # backup_project_as_zip(PROJECT_PATH, out_dir + '/backup/code.train.%s.zip' % IDENTIFIER)

    log = Logger()
    log.open(out_dir + '/log.train.txt', mode='a')
    log.write('\n--- [START %s] %s\n\n' % (IDENTIFIER, '-' * 64))
    log.write('\t%s\n' % COMMON_STRING)
    log.write('\n')

    log.write('\tSEED         = %u\n' % SEED)
    log.write('\tPROJECT_PATH = %s\n' % PROJECT_PATH)
    log.write('\t__file__     = %s\n' % CURRENT_DIR)
    log.write('\tout_dir      = %s\n' % out_dir)
    log.write('\n')

    # tensorboard
    writer = SummaryWriter(log_dir = os.path.join(config.logs, config.model_name))

    ## dataset ----------------------------------------
    log.write('** dataset setting **\n')
    train_images_dir = config.data_dir + '/train_images/{}.jpg'
    train = pd.read_csv(os.path.join(config.data_dir, 'train.csv'))
    df_train, df_eval = train_test_split(train, test_size=0.01, random_state=42)
    train_dataset = CarDataset(
        df_train,
        train_images_dir,
        training=True)
    valid_dataset = CarDataset(
        df_eval,
        train_images_dir,
        training=True)
    # print(train_dataset)

    train_loader = DataLoader(
        dataset=train_dataset,
        sampler=RandomSampler(train_dataset),
        batch_size=4,
        num_workers=4)

    valid_loader = DataLoader(
        dataset = valid_dataset,
        sampler=SequentialSampler(valid_dataset),
        batch_size=4,
        drop_last=False,
        num_workers=os.cpu_count()
    )
    # collate_fn?

    assert (len(train_dataset) >= batch_size)
    log.write('batch_size = %d\n' % (batch_size))
    log.write('train_dataset : \n%s\n' % (train_dataset))
    log.write('valid_dataset : \n%s\n' % (valid_dataset))
    log.write('\n')

    ## net ----------------------------------------
    log.write('** net setting **\n')
    net = MyUNet(config.num_classes).cuda()
    log.write('\tinitial_checkpoint = %s\n' % initial_checkpoint)

    if initial_checkpoint is not None:
        state_dict = torch.load(initial_checkpoint, map_location=lambda storage, loc: storage)
        ##for k in ['logit.weight','logit.bias']: state_dict.pop(k, None) #tramsfer sigmoid feature to softmax network
        ##net.load_state_dict(state_dict,strict=False)
        net.load_state_dict(state_dict, strict=False)

    # else:
    #     net.load_pretrain(skip=['logit'], is_print=False)

    log.write('%s\n' % (type(net)))
    # log.write('\tloss_weight = %s\n' % str(loss_weight))
    log.write('\ttrain_loader.sampler = %s\n' % str(train_loader.sampler))
    log.write('\n')

    ## optimiser ----------------------------------
    # if 0: ##freeze
    #     for p in net.encoder1.parameters(): p.requires_grad = False
    #     pass

    # net.set_mode('train',is_freeze_bn=True)
    # -----------------------------------------------

    # optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()),lr=schduler(0))
    # optimizer = torch.optim.RMSprop(net.parameters(), lr =0.0005, alpha = 0.95)
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=schduler(0), momentum=0.9,
                                weight_decay=0.0001)

    num_iters = 3000 * 1000
    iter_smooth = 50
    iter_log = 500
    iter_valid = 1500
    iter_save = [0, num_iters - 1] \
                + list(range(0, num_iters, 1500))  # 1*1000

    start_iter = 0
    start_epoch = 0
    rate = 0
    if initial_checkpoint is not None:
        initial_optimizer = initial_checkpoint.replace('_model.pth', '_optimizer.pth')
        if os.path.exists(initial_optimizer):
            checkpoint = torch.load(initial_optimizer)
            start_iter = checkpoint['iter']
            start_epoch = checkpoint['epoch']
            # optimizer.load_state_dict(checkpoint['optimizer'])
        pass

    log.write('optimizer\n  %s\n' % (optimizer))
    log.write('schduler\n  %s\n' % (schduler))
    log.write('\n')

    ## start training here! ##############################################

    log.write('** start training here! **\n')
    log.write('   batch_size=%d,  iter_accum=%d\n' % (batch_size, iter_accum))
    log.write('   experiment  = %s\n' % CURRENT_DIR.split('/')[-2])
    log.write(
        '                      |---- VALID----------------------|---------- TRAIN/BATCH ------------------------------\n')
    log.write(
        'rate     iter   epoch |  loss   |  loss   | time         \n')
    log.write(
        '------------------------------------------------------------------------------------------------------------------------------------------------\n')
    # 0.00000    0.0*   0.0 |  0.690   0.50 [0.00,1.00,0.00,1.00]   0.44 [0.00,0.02,0.00,0.15]  |  0.000   0.00 [0.00,0.00,0.00,0.00]  |  0 hr 00 min

    train_loss = np.zeros(3, np.float32)
    valid_loss = np.zeros(3, np.float32)
    batch_loss = np.zeros(3, np.float32)
    iter = 0
    i = 0

    start = timer()
    while iter < num_iters:
        sum_train_loss = np.zeros(3, np.float32)
        sum = np.zeros(3, np.float32)

        optimizer.zero_grad()
        for t, (input, truth_mask, regr_batch, id) in enumerate(train_loader):

            batch_size = input.shape[0]
            iter = i + start_iter
            epoch = (iter - start_iter) * batch_size / len(train_dataset) + start_epoch

            # Validation
            if (iter % iter_valid == 0):
                valid_loss = do_valid(net, valid_loader, out_dir)  #
                # tensorboard
                # loss    hit_neg, pos1
                writer.add_scalars('Loss/loss', {'valid': valid_loss[0]}, iter)
                writer.add_scalars('Loss/hit_neg', {'valid': valid_loss[1]}, iter)
                writer.add_scalars('Loss/pos1',  {'valid': valid_loss[2]}, iter)

                # # # dice_neg,pos1,2,3,4
                # writer.add_scalars('Valid_dice_neg/loss', {'valid': valid_loss[6]}, iter)
                # writer.add_scalars('Valid_dice_neg/pos1',  {'valid': valid_loss[7]}, iter)
                # writer.add_scalars('Valid_dice_neg/pos2',  {'valid': valid_loss[8]}, iter)
                # writer.add_scalars('Valid_dice_neg/pos3',  {'valid': valid_loss[9]}, iter)
                # writer.add_scalars('Valid_dice_neg/pos4',  {'valid': valid_loss[10]}, iter)
                # pass

            # Logging
            if (iter % iter_log == 0):
                print('\r', end='', flush=True)
                asterisk = '*' if iter in iter_save else ' '
                log.write(
                    '%0.5f  %5.1f%s %5.1f |  %5.3f  |  %5.3f ' % ( \
                        rate, iter / 1000, asterisk, epoch,
                        valid_loss[0],
                        train_loss[0])
                    )
                log.write('\n')
                # tensorboard
                # loss    hit_neg,pos1,2,3,4
                writer.add_scalars('Loss/loss', {'train': train_loss[0]}, iter)
                # writer.add_scalars('Loss/hit_neg', {'train': train_loss[1]}, iter)
                # writer.add_scalars('Loss/pos1',  {'train': train_loss[2]}, iter)
                # writer.add_scalars('Loss/pos2',  {'train': train_loss[3]}, iter)
                # writer.add_scalars('Loss/pos3',  {'train': train_loss[4]}, iter)
                # writer.add_scalars('Loss/pos4',  {'train': train_loss[5]}, iter)

            # Saving
            if iter in iter_save:
                torch.save(net.state_dict(), out_dir + '/checkpoint/%08d_model.pth' % (iter))
                torch.save({
                    # 'optimizer': optimizer.state_dict(),
                    'iter': iter,
                    'epoch': epoch,
                }, out_dir + '/checkpoint/%08d_optimizer.pth' % (iter))
                pass

            # learning rate schduler -------------
            lr = schduler(iter)
            if lr < 0: break
            adjust_learning_rate(optimizer, lr)
            rate = get_learning_rate(optimizer)

            writer.add_scalar('Learning_rate/rate', rate, iter)
            # one iteration update  -------------
            # net.set_mode('train',is_freeze_bn=True)

            net.train()
            input = input.cuda()
            truth_mask = truth_mask.cuda()
            regr_batch = regr_batch.cuda()
            logit = data_parallel(net, input)  # net(input)
            loss = criterion(logit, truth_mask, regr_batch)
            tn, tp, num_neg, num_pos = metric_hit(logit[:, 0], truth_mask)

            (loss / iter_accum).backward()
            if (iter % iter_accum) == 0:
                optimizer.step()
                optimizer.zero_grad()

            # print statistics  ------------
            l = np.array([loss.item(), tn, *tp])
            n = np.array([batch_size, num_neg, *num_pos])

            batch_loss = l
            sum_train_loss += l * n
            sum += n
            if iter % iter_smooth == 0:
                train_loss = sum_train_loss / (sum + 1e-12)
                sum_train_loss[...] = 0
                sum[...] = 0

            print('\r', end='', flush=True)
            asterisk = ' '
            print(
                '%0.5f  %5.1f%s %5.1f |  %5.3f  |  %5.3f ' % ( \
                    rate, iter / 1000, asterisk, epoch,
                    valid_loss[0],
                    batch_loss[0])
                , end='', flush=True)
            i = i + 1
        pass  # -- end of one data loader --
    pass  # -- end of all iterations --

    log.write('\n')

def do_valid(net, valid_loader, out_dir=None):
    # out_dir=None
    valid_num = np.zeros(3, np.float32)
    valid_loss = np.zeros(3, np.float32)

    for t, (input, truth_mask, regr_batch, id) in enumerate(valid_loader):
        # if b==5: break
        net.eval()
        input = input.cuda()
        truth_mask = truth_mask.cuda()
        regr_batch = regr_batch.cuda()

        with torch.no_grad():
            logit = data_parallel(net, input)  # net(input)
            loss = criterion(logit, truth_mask, regr_batch, size_average=False)
            tn, tp, num_neg, num_pos = metric_hit(logit[:, 0], truth_mask)
            # dn, dp, num_neg, num_pos = metric_dice(logit, truth_mask, threshold=0.5, sum_threshold=100)

            # zz=0
        # ---
        batch_size = input.shape[0]
        l = np.array([loss.item(), tn, *tp])
        n = np.array([batch_size, num_neg, *num_pos])
        valid_loss += l * n
        valid_num += n

        # debug-----------------------------
        if config.debug:
            probability = torch.softmax(logit, 1)
            image = input_to_image(input, IMAGE_RGB_MEAN, IMAGE_RGB_STD)

            probability = one_hot_encode_predict(probability)
            truth_mask = one_hot_encode_truth(truth_mask)

            probability_mask = probability.data.cpu().numpy()
            truth_label = truth_label.data.cpu().numpy()
            truth_mask = truth_mask.data.cpu().numpy()

            for b in range(0, batch_size, 4):
                # image_id = infor[b].image_id[:-4]
                result = draw_predict_result(image[b], truth_mask[b], truth_label[b], probability_mask[b],
                                             stack='vertical')
                draw_shadow_text(result, '%05d    %s.jpg' % (valid_num[0] - batch_size + b, image_id), (5, 24), 1,
                                 [255, 255, 255], 2)

                image_show('result', result, resize=1)
                cv2.imwrite(out_dir + '/valid/%s.png' % (infor[b].image_id[:-4]), result)
                cv2.waitKey(1)
                pass
        # debug-----------------------------

        # print(valid_loss)
        # print('\r %8d /%8d' % (valid_num[0], len(valid_loader.dataset)), end='', flush=True)

        pass  # -- end of one data loader --
    # assert (valid_num[0] == len(valid_loader.dataset))
    valid_loss = valid_loss / valid_num

    return valid_loss



# main #################################################################
if __name__ == '__main__':
    print('%s: calling main function ... ' % os.path.basename(__file__))

    run_train()
