from common import *
from kaggle_utils import *
from config import *

DATA_DIR = config.data_dir
IMG_WIDTH = 1024
IMG_HEIGHT = IMG_WIDTH // 16 * 5 # 320
MODEL_SCALE = 8

# Our code will generate data, visualization and model checkpoints, they will be persisted to disk in this folder
INPUT_FOLDER= "../input"
os.makedirs(INPUT_FOLDER, exist_ok=True)

# From camera.zip
camera_matrix = np.array([[2304.5479, 0,  1686.2379],
                          [0, 2305.8757, 1354.9849],
                          [0, 0, 1]], dtype=np.float32)
camera_matrix_inv = np.linalg.inv(camera_matrix)

class CarDataset(Dataset):
    """Car dataset."""

    def __init__(self, dataframe, root_dir, training=True, transform=None):
        self.df = dataframe
        self.root_dir = root_dir
        self.transform = transform
        self.training = training

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Get image name
        idx, labels = self.df.values[idx]
        # img_name = self.root_dir.format(idx)
        # labels:
        # '2 0.146265 -0.048006 -3.09404 -2.82929 5.06246 23.5584
        # 70 0.160453 3.1192 -3.10969 -7.25741 7.01096 36.5986
        # 35 0.169212 3.11246 -3.11718 -10.3985 6.65688 34.0345
        # 12 0.145069 -0.368432 3.05084 -3.64239 2.89103 6.13721'

        # Augmentation
        flip = False
        if self.training:
            flip = np.random.randint(10) == 1

        # Read image
        # img0 = imread(img_name, True)
        img0 = createMaskImages(idx) # B G R [2710, 3384, 3] -> [1255, 3384, 3] cut the top half because it is sky, no cars
        img = preprocess_image(img0, flip=flip) # [320, 1024,3]
        img = np.rollaxis(img, 2, 0) # [3, 320, 1024]

        # Get mask and regression maps
        mask, regr = get_mask_and_regr(img0, labels, flip=flip)
        regr = np.rollaxis(regr, 2, 0)

        return [img, mask, regr, idx]


def createMaskImages(imageName):
    trainimage = cv2.imread(INPUT_FOLDER + "/train_images/" + imageName + '.jpg')
    imagemask = cv2.imread(INPUT_FOLDER + "/train_masks/" + imageName + ".jpg", 0)
    try:
        imagemaskinv = cv2.bitwise_not(imagemask) # opposite of the mask
        res = cv2.bitwise_and(trainimage, trainimage, mask=imagemaskinv) # src1, src2
        # dst = src1 && src2 if mask != 0

        # cut upper half,because it doesn't contain cars.
        res = res[res.shape[0] // 2:]
        return res
    except:
        trainimage = trainimage[trainimage.shape[0] // 2:]
        return trainimage


def imread(path, fast_mode=False):
    img = cv2.imread(path)  # B G R
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) # RGB
    if not fast_mode and img is not None and len(img.shape) == 3:
        img = np.array(img[:, :, ::-1])
    return img

# image processing
def _regr_preprocess(regr_dict, flip=False):
    if flip:
        for k in ['x', 'pitch', 'roll']:
            regr_dict[k] = -regr_dict[k]
    for name in ['x', 'y', 'z']:
        regr_dict[name] = regr_dict[name] / 100
    regr_dict['roll'] = regr_dict['roll']
    regr_dict['pitch_sin'] = sin(regr_dict['pitch'])
    regr_dict['pitch_cos'] = cos(regr_dict['pitch'])
    regr_dict.pop('pitch')
    regr_dict.pop('id')
    return regr_dict


def _regr_back(regr_dict):
    for name in ['x', 'y', 'z']:
        regr_dict[name] = regr_dict[name] * 100
    regr_dict['roll'] = regr_dict['roll']

    pitch_sin = regr_dict['pitch_sin'] / np.sqrt(regr_dict['pitch_sin'] ** 2 + regr_dict['pitch_cos'] ** 2)
    pitch_cos = regr_dict['pitch_cos'] / np.sqrt(regr_dict['pitch_sin'] ** 2 + regr_dict['pitch_cos'] ** 2)
    regr_dict['pitch'] = np.arccos(pitch_cos) * np.sign(pitch_sin)
    return regr_dict

def preprocess_image(img, flip=False):
    img = img[img.shape[0] // 2:]
    bg = np.ones_like(img) * img.mean(1, keepdims=True).astype(img.dtype)
    bg = bg[:, :img.shape[1] // 6]
    img = np.concatenate([bg, img, bg], 1)
    img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
    if flip:
        img = img[:,::-1]
    return (img / 255).astype('float32')

def get_mask_and_regr(img, labels, flip=False):
    mask = np.zeros([IMG_HEIGHT // MODEL_SCALE, IMG_WIDTH // MODEL_SCALE], dtype='float32')
    regr_names = ['x', 'y', 'z', 'yaw', 'pitch', 'roll']
    regr = np.zeros([IMG_HEIGHT // MODEL_SCALE, IMG_WIDTH // MODEL_SCALE, 7], dtype='float32')
    coords = str2coords(labels)
    xs, ys = get_img_coords(labels)
    for x, y, regr_dict in zip(xs, ys, coords):
        x, y = y, x
        x = (x - img.shape[0] // 2) * IMG_HEIGHT / (img.shape[0] // 2) / MODEL_SCALE
        x = np.round(x).astype('int')
        y = (y + img.shape[1] // 6) * IMG_WIDTH / (img.shape[1] * 4/3) / MODEL_SCALE
        y = np.round(y).astype('int')
        if x >= 0 and x < IMG_HEIGHT // MODEL_SCALE and y >= 0 and y < IMG_WIDTH // MODEL_SCALE:
            mask[x, y] = 1
            regr_dict = _regr_preprocess(regr_dict, flip)
            regr[x, y] = [regr_dict[n] for n in sorted(regr_dict)]
    if flip:
        mask = np.array(mask[:,::-1])
        regr = np.array(regr[:,::-1])
    return mask, regr

def str2coords(s, names=['id', 'yaw', 'pitch', 'roll', 'x', 'y', 'z']):
    '''
    Input:
        s: PredictionString (e.g. from train dataframe)
        names: array of what to extract from the string
    Output:
        list of dicts with keys from `names`
    '''
    coords = []
    for l in np.array(s.split()).reshape([-1, 7]):
        coords.append(dict(zip(names, l.astype('float'))))
        if 'id' in coords[-1]:
            coords[-1]['id'] = int(coords[-1]['id'])
    return coords

def get_img_coords(s):
    '''
    Input is a PredictionString (e.g. from train dataframe)
    Output is two arrays:
        xs: x coordinates in the image (row)
        ys: y coordinates in the image (column)
    '''
    coords = str2coords(s)
    xs = [c['x'] for c in coords]
    ys = [c['y'] for c in coords]
    zs = [c['z'] for c in coords]
    P = np.array(list(zip(xs, ys, zs))).T
    img_p = np.dot(camera_matrix, P).T
    img_p[:, 0] /= img_p[:, 2]
    img_p[:, 1] /= img_p[:, 2]
    img_xs = img_p[:, 0]
    img_ys = img_p[:, 1]
    img_zs = img_p[:, 2] # z = Distance from the camera
    return img_xs, img_ys

'''
test_dataset : 
	mode    = train
	split   = ['valid0_500.npy']
	csv     = ['train.csv']
		len   =   500
		neg   =   212  0.424
		pos   =   288  0.576
		pos1  =    35  0.070  0.122
		pos2  =     5  0.010  0.017
		pos3  =   213  0.426  0.740
		pos4  =    35  0.070  0.122
		

train_dataset : 
	mode    = train
	split   = ['train0_12068.npy']
	csv     = ['train.csv']
		len   = 12068
		neg   =  5261  0.436
		pos   =  6807  0.564
		pos1  =   862  0.071  0.127
		pos2  =   242  0.020  0.036
		pos3  =  4937  0.409  0.725
		pos4  =   766  0.063  0.113

		
'''


def null_collate(batch):
    batch_size = len(batch)

    input = []
    truth = []
    infor = []
    for b in range(batch_size):
        input.append(batch[b][0])
        truth.append(batch[b][1])
        infor.append(batch[b][2])

    input = np.stack(input).astype(np.float32) / 255
    input = input.transpose(0, 3, 1, 2)
    truth = np.stack(truth)
    truth = (truth > 0.5).astype(np.float32)

    input = torch.from_numpy(input).float()
    truth = torch.from_numpy(truth).float()

    return input, truth, infor


class FourBalanceClassSampler(Sampler):

    def __init__(self, dataset):
        self.dataset = dataset

        label = (self.dataset.df['Label'].values)
        label = label.reshape(-1, 4)
        label = np.hstack([label.sum(1, keepdims=True) == 0, label]).T

        self.neg_index = np.where(label[0])[0]
        self.pos1_index = np.where(label[1])[0]
        self.pos2_index = np.where(label[2])[0]
        self.pos3_index = np.where(label[3])[0]
        self.pos4_index = np.where(label[4])[0]

        # assume we know neg is majority class
        num_neg = len(self.neg_index)
        self.length = 4 * num_neg

    def __iter__(self):
        neg = self.neg_index.copy()
        random.shuffle(neg)
        num_neg = len(self.neg_index)

        pos1 = np.random.choice(self.pos1_index, num_neg, replace=True)
        pos2 = np.random.choice(self.pos2_index, num_neg, replace=True)
        pos3 = np.random.choice(self.pos3_index, num_neg, replace=True)
        pos4 = np.random.choice(self.pos4_index, num_neg, replace=True)

        l = np.stack([neg, pos1, pos2, pos3, pos4]).T
        l = l.reshape(-1)
        return iter(l)

    def __len__(self):
        return self.length


class FixedSampler(Sampler):

    def __init__(self, dataset, index):
        self.dataset = dataset
        self.index = index
        self.length = len(index)

    def __iter__(self):
        return iter(self.index)

    def __len__(self):
        return self.length


##############################################################
#
# class BalanceClassSampler(Sampler):
#
#     def __init__(self, dataset, length=None):
#         self.dataset = dataset
#
#         if length is None:
#             length = len(self.dataset)
#
#         self.length = length
#
#
#
#     def __iter__(self):
#
#         df = self.dataset.df
#         df['Class'] = df['ImageId_ClassId'].str[-1].astype(np.int32)
#         df['Label'] = (df['EncodedPixels']!='').astype(np.int32)
#
#
#         label = df['Label'].values*df['Class'].values
#         unique, count = np.unique(label, return_counts=True)
#         L = len(label)//5
#
#
#
#
#         pos_index = np.where(self.dataset.label==1)[0]
#         neg_index = np.where(self.dataset.label==0)[0]
#         half = self.length//2 + 1
#
#
#         neg  = np.random.choice(label==0, [L,6], replace=True)
#         pos1 = np.random.choice(label==1, L, replace=True)
#         pos2 = np.random.choice(label==2, L, replace=True)
#         pos3 = np.random.choice(label==3, L, replace=True)
#         pos4 = np.random.choice(label==4, L, replace=True)
#
#
#         l = np.stack([neg.reshape,pos1,pos2,pos3,pos3,pos4]).T
#         l = l.reshape(-1)
#         l = l[:self.length]
#         return iter(l)
#
#     def __len__(self):
#         return self.length


##############################################################

def image_to_input(image, rbg_mean, rbg_std):  # , rbg_mean=[0,0,0], rbg_std=[1,1,1]):
    input = image.astype(np.float32)
    input = input[..., ::-1] / 255
    input = input.transpose(0, 3, 1, 2)
    input[:, 0] = (input[:, 0] - rbg_mean[0]) / rbg_std[0]
    input[:, 1] = (input[:, 1] - rbg_mean[1]) / rbg_std[1]
    input[:, 2] = (input[:, 2] - rbg_mean[2]) / rbg_std[2]
    return input


def input_to_image(input, rbg_mean, rbg_std):  # , rbg_mean=[0,0,0], rbg_std=[1,1,1]):
    input = input.data.cpu().numpy()
    input[:, 0] = (input[:, 0] * rbg_std[0] + rbg_mean[0])
    input[:, 1] = (input[:, 1] * rbg_std[1] + rbg_mean[1])
    input[:, 2] = (input[:, 2] * rbg_std[2] + rbg_mean[2])
    input = input.transpose(0, 2, 3, 1)
    input = input[..., ::-1]
    image = (input * 255).astype(np.uint8)
    return image


##############################################################

def do_random_crop(image, mask, w, h):
    height, width = image.shape[:2]
    x, y = 0, 0
    if width > w:
        x = np.random.choice(width - w)
    if height > h:
        y = np.random.choice(height - h)
    image = image[y:y + h, x:x + w]
    mask = mask[:, y:y + h, x:x + w]
    return image, mask


def do_random_crop_rescale(image, mask, w, h):
    height, width = image.shape[:2]
    x, y = 0, 0
    if width > w:
        x = np.random.choice(width - w)
    if height > h:
        y = np.random.choice(height - h)
    image = image[y:y + h, x:x + w]
    mask = mask[:, y:y + h, x:x + w]

    # ---
    if (w, h) != (width, height):
        image = cv2.resize(image, dsize=(width, height), interpolation=cv2.INTER_LINEAR)

        mask = mask.transpose(1, 2, 0)
        mask = cv2.resize(mask, dsize=(width, height), interpolation=cv2.INTER_NEAREST)
        mask = mask.transpose(2, 0, 1)

    return image, mask


def do_flip_lr(image, mask):
    image = cv2.flip(image, 1)
    mask = mask[:, :, ::-1]
    return image, mask


def do_flip_ud(image, mask):
    image = cv2.flip(image, 0)
    mask = mask[:, ::-1, :]
    return image, mask


def do_random_scale_rotate(image, mask, w, h):
    H, W = image.shape[:2]

    # dangle = np.random.uniform(-2.5, 2.5)
    # dscale = np.random.uniform(-0.10,0.10,2)
    dangle = np.random.uniform(-5, 5)
    dscale = np.random.uniform(-0.15, 0.15, 2)
    dshift = np.random.uniform(0, 1, 2)
    cos = np.cos(dangle / 180 * PI)
    sin = np.sin(dangle / 180 * PI)
    sx, sy = 1 + dscale  # 1,1 #
    tx, ty = dshift

    src = np.array([[-w / 2, -h / 2], [w / 2, -h / 2], [w / 2, h / 2], [-w / 2, h / 2]], np.float32)
    src = src * [sx, sy]
    x = (src * [cos, -sin]).sum(1)
    y = (src * [sin, cos]).sum(1)
    x = x - x.min()
    y = y - y.min()
    x = x + (W - x.max()) * tx
    y = y + (H - y.max()) * ty

    if 0:
        overlay = image.copy()
        for i in range(4):
            cv2.line(overlay, int_tuple([x[i], y[i]]), int_tuple([x[(i + 1) % 4], y[(i + 1) % 4]]), (0, 0, 255), 5)
        image_show('overlay', overlay)
        cv2.waitKey(0)

    src = np.column_stack([x, y])
    dst = np.array([[0, 0], [w, 0], [w, h], [0, h]])
    s = src.astype(np.float32)
    d = dst.astype(np.float32)
    transform = cv2.getPerspectiveTransform(s, d)

    image = cv2.warpPerspective(image, transform, (w, h),
                                flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))

    mask = mask.transpose(1, 2, 0)
    mask = cv2.warpPerspective(mask, transform, (w, h),
                               flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
    mask = mask.transpose(2, 0, 1)
    mask = (mask > 0.5).astype(np.float32)

    return image, mask


def do_random_crop_rotate_rescale(image, mask, w, h):
    H, W = image.shape[:2]

    # dangle = np.random.uniform(-2.5, 2.5)
    # dscale = np.random.uniform(-0.10,0.10,2)
    dangle = np.random.uniform(-8, 8)
    dshift = np.random.uniform(-0.1, 0.1, 2)

    dscale_x = np.random.uniform(-0.00075, 0.00075)
    dscale_y = np.random.uniform(-0.25, 0.25)

    cos = np.cos(dangle / 180 * PI)
    sin = np.sin(dangle / 180 * PI)
    sx, sy = 1 + dscale_x, 1 + dscale_y  # 1,1 #
    tx, ty = dshift * min(H, W)

    src = np.array([[-w / 2, -h / 2], [w / 2, -h / 2], [w / 2, h / 2], [-w / 2, h / 2]], np.float32)
    src = src * [sx, sy]
    x = (src * [cos, -sin]).sum(1) + W / 2
    y = (src * [sin, cos]).sum(1) + H / 2
    # x = x-x.min()
    # y = y-y.min()
    # x = x + (W-x.max())*tx
    # y = y + (H-y.max())*ty

    if 0:
        overlay = image.copy()
        for i in range(4):
            cv2.line(overlay, int_tuple([x[i], y[i]]), int_tuple([x[(i + 1) % 4], y[(i + 1) % 4]]), (0, 0, 255), 5)
        image_show('overlay', overlay)
        cv2.waitKey(0)

    src = np.column_stack([x, y])
    dst = np.array([[0, 0], [w, 0], [w, h], [0, h]])
    s = src.astype(np.float32)
    d = dst.astype(np.float32)
    transform = cv2.getPerspectiveTransform(s, d)

    image = cv2.warpPerspective(image, transform, (W, H),
                                flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))

    mask = mask.transpose(1, 2, 0)
    mask = cv2.warpPerspective(mask, transform, (W, H),
                               flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
    mask = mask.transpose(2, 0, 1)

    return image, mask


def do_random_log_contast(image):
    gain = np.random.uniform(0.70, 1.30, 1)
    inverse = np.random.choice(2, 1)

    image = image.astype(np.float32) / 255
    if inverse == 0:
        image = gain * np.log(image + 1)
    else:
        image = gain * (2 ** image - 1)

    image = np.clip(image * 255, 0, 255).astype(np.uint8)
    return image


def do_noise(image, mask, noise=8):
    H, W = image.shape[:2]
    image = image + np.random.uniform(-1, 1, (H, W, 1)) * noise
    image = np.clip(image, 0, 255).astype(np.uint8)
    return image, mask

# convert euler angle to rotation matrix
def euler_to_Rot(yaw, pitch, roll):
    Y = np.array([[cos(yaw), 0, sin(yaw)],
                  [0, 1, 0],
                  [-sin(yaw), 0, cos(yaw)]])
    P = np.array([[1, 0, 0],
                  [0, cos(pitch), -sin(pitch)],
                  [0, sin(pitch), cos(pitch)]])
    R = np.array([[cos(roll), -sin(roll), 0],
                  [sin(roll), cos(roll), 0],
                  [0, 0, 1]])
    return np.dot(Y, np.dot(P, R))

def draw_line(image, points):
    color = (255, 0, 0)
    cv2.line(image, tuple(points[0][:2]), tuple(points[3][:2]), color, 16)
    cv2.line(image, tuple(points[0][:2]), tuple(points[1][:2]), color, 16)
    cv2.line(image, tuple(points[1][:2]), tuple(points[2][:2]), color, 16)
    cv2.line(image, tuple(points[2][:2]), tuple(points[3][:2]), color, 16)
    return image


def draw_points(image, points):
    for (p_x, p_y, p_z) in points:
        cv2.circle(image, (p_x, p_y), int(1000 / p_z), (0, 255, 0), -1)
#         if p_x > image.shape[1] or p_y > image.shape[0]:
#             print('Point', p_x, p_y, 'is out of image with shape', image.shape)
    return image

def visualize(img, coords):
    # You will also need functions from the previous cells
    x_l = 1.02
    y_l = 0.80
    z_l = 2.31

    img = img.copy()
    for point in coords:
        # Get values
        x, y, z = point['x'], point['y'], point['z']
        yaw, pitch, roll = -point['pitch'], -point['yaw'], -point['roll']
        # Math
        Rt = np.eye(4)
        t = np.array([x, y, z])
        Rt[:3, 3] = t
        Rt[:3, :3] = euler_to_Rot(yaw, pitch, roll).T
        Rt = Rt[:3, :]
        P = np.array([[x_l, -y_l, -z_l, 1],
                      [x_l, -y_l, z_l, 1],
                      [-x_l, -y_l, z_l, 1],
                      [-x_l, -y_l, -z_l, 1],
                      [0, 0, 0, 1]]).T
        img_cor_points = np.dot(camera_matrix, np.dot(Rt, P))
        img_cor_points = img_cor_points.T
        img_cor_points[:, 0] /= img_cor_points[:, 2]
        img_cor_points[:, 1] /= img_cor_points[:, 2]
        img_cor_points = img_cor_points.astype(int)
        # Drawing
        img = draw_line(img, img_cor_points)
        img = draw_points(img, img_cor_points[-1:])

    return img

##############################################################

def prepare_training_data_for_scene(first_sample_token, dataset, output_folder, bev_shape, voxel_size, z_offset, box_scale, map_mask):
    """
    Given a first sample token (in a scene), output rasterized input volumes and targets in birds-eye-view perspective.

    """
    sample_token = first_sample_token

    while sample_token:

        sample = dataset.get("sample", sample_token)

        sample_lidar_token = sample["data"]["LIDAR_TOP"]
        lidar_data = dataset.get("sample_data", sample_lidar_token)
        lidar_filepath = dataset.get_sample_data_path(sample_lidar_token)

        ego_pose = dataset.get("ego_pose", lidar_data["ego_pose_token"])
        calibrated_sensor = dataset.get("calibrated_sensor", lidar_data["calibrated_sensor_token"])

        # global_from_car = transform_matrix(ego_pose['translation'],
        #                                    Quaternion(ego_pose['rotation']), inverse=False)

        car_from_sensor = transform_matrix(calibrated_sensor['translation'], Quaternion(calibrated_sensor['rotation']),
                                           inverse=False)

        try:
            lidar_pointcloud = LidarPointCloud.from_file(lidar_filepath)
            lidar_pointcloud.transform(car_from_sensor)
        except Exception as e:
            print("Failed to load Lidar Pointcloud for {}: {}:".format(sample_token, e))
            sample_token = sample["next"]
            continue

        bev = create_voxel_pointcloud(lidar_pointcloud.points, bev_shape, voxel_size=voxel_size, z_offset=z_offset)
        bev = normalize_voxel_intensities(bev)

        boxes = dataset.get_boxes(sample_lidar_token)

        target = np.zeros_like(bev)

        # move_boxes_to_car_space(boxes, ego_pose)
        # scale_boxes(boxes, box_scale)
        # draw_boxes(target, voxel_size, boxes=boxes, classes=classes, z_offset=z_offset)

        # bev_im = np.round(bev * 255).astype(np.uint8)
        target_im = target[:, :, 0]  # take one channel only

        semantic_im = get_semantic_map_around_ego(map_mask, ego_pose, voxel_size[0], target_im.shape)
        semantic_im = np.round(semantic_im * 255).astype(np.uint8)

        # cv2.imwrite(os.path.join(output_folder, "{}_input.png".format(sample_token)), bev_im)
        # cv2.imwrite(os.path.join(output_folder, "{}_target.png".format(sample_token)), target_im)
        cv2.imwrite(os.path.join(output_folder, "{}_map.png".format(sample_token)), semantic_im)

        sample_token = sample["next"]

def run_check_train_data():
    train_images_dir = INPUT_FOLDER + '/train_images/{}.jpg'
    train = pd.read_csv(os.path.join(INPUT_FOLDER, 'train.csv'))
    df_train, df_eval = train_test_split(train, test_size=0.01, random_state=42)

    train_dataset = CarDataset(df_train, train_images_dir, training=True)

    # im, target, sample_token = train_dataset[1]
    # im = im.numpy()
    # target = target.numpy()
    # plt.figure(figsize=(16, 8))
    # target_as_rgb = np.repeat(target[..., None], 3, 2)
    # # Transpose the input volume CXY to XYC order, which is what matplotlib requires.
    # plt.imshow(np.hstack((im.transpose(1, 2, 0)[..., :3], target_as_rgb)))
    # plt.title(sample_token)
    # plt.show()

    for n in range(0, len(train_dataset)):
        i = n  # i = np.random.choice(len(dataset))

        image, mask, regr, id = train_dataset[n]

        fig = plt.figure(figsize=(16, 16))
        plt.imshow(np.rollaxis(image, 0, 3))
        plt.title(id)
        print(id)
        plt.show()

        plt.figure(figsize=(16, 16))
        plt.imshow(mask)
        plt.title(id)
        plt.show()

        plt.figure(figsize=(16, 16))
        plt.imshow(regr[-2])
        plt.title(id)
        plt.show()


def run_check_test_dataset():
    dataset = SteelDataset(
        mode='test',
        csv=['sample_submission.csv', ],
        split=['test_1801.npy', ],
        augment=None,  #
    )
    print(dataset)
    # exit(0)

    for n in range(0, len(dataset)):
        i = n  # i = np.random.choice(len(dataset))

        image, mask, infor = dataset[i]
        overlay = np.vstack([m for m in mask])

        # ----
        print('%05d : %s' % (i, infor.image_id))
        image_show('image', image, 0.5)
        image_show_norm('mask', overlay, 0, 1, 0.5)
        cv2.waitKey(0)


def run_check_data_loader():
    train_images_dir = INPUT_FOLDER + '/train_images/{}.jpg'
    train = pd.read_csv(os.path.join(INPUT_FOLDER, 'train.csv'))
    df_train, df_eval = train_test_split(train, test_size=0.01, random_state=42)
    dataset = CarDataset(
        df_train,
        train_images_dir,
        training=True)
    print(dataset)
    loader = DataLoader(
        dataset=dataset,
        sampler=RandomSampler(dataset),
        batch_size=4,
        num_workers=4)

    for t, (input, truth, regr, id) in enumerate(loader):

        print('----t=%d---' % t)
        print('')
        print('input', input.shape)
        print('truth', truth.shape)
        print('id', id)
        print('')

        if 1:
            batch_size = input.shape[0]
            input = input.data.cpu().numpy()
            truth = truth.data.cpu().numpy()
            for b in range(batch_size):
                fig, axes = plt.subplots(3, figsize=(20, 20))
                # img = input[b]
                # axes[0].imshow(np.rollaxis(img, 0, 3))
                # temp = train.loc[train['ImageId'] == id[b]]['PredictionString'].values[0]
                # img_vis = visualize(img, str2coords(temp))
                # axes[1].imshow(np.rollaxis(img_vis, 0, 3))
                # plt.show()

                image = input[b]
                mask = truth[b]
                reg = regr[b]
                fig.suptitle(id[b])
                axes[0].imshow(np.rollaxis(image, 0, 3))
                axes[0].set_title('image')
                axes[1].imshow(mask)
                axes[1].set_title('mask')
                axes[2].imshow(reg[-2])
                axes[2].set_title('yaw')
                plt.show()

                # plt.imshow(np.rollaxis(image, 0, 3))
                # plt.title(id[b])
                # plt.show()
                #
                # plt.figure(figsize=(16, 16))
                # plt.imshow(mask)
                # plt.title(id[b])
                # plt.show()
                #
                # plt.figure(figsize=(16, 16))
                # plt.imshow(reg[-2])
                # plt.title(id[b])
                # plt.show()
                #


def run_check_augment():
    def augment(image, mask, infor):
        # image, mask = do_random_scale_rotate(image, mask)
        # image = do_random_log_contast(image)

        # if np.random.rand()<0.5:
        #     image, mask = do_flip_ud(image, mask)

        # image, mask = do_noise(image, mask, noise=8)
        # image, mask = do_random_crop_rescale(image,mask,1600-(256-224),224)
        image, mask = do_random_crop_rotate_rescale(image, mask, 1600 - (256 - 224), 224)

        # image, mask = do_random_scale_rotate(image, mask, 224*2, 224)
        return image, mask, infor

    dataset = SteelDataset(
        mode='train',
        csv=['train.csv', ],
        split=['train0_12068.npy', ],
        augment=None,  # None
    )
    print(dataset)

    for t in range(len(dataset)):
        image, mask, infor = dataset[t]

        overlay = image.copy()
        overlay = draw_contour_overlay(overlay, mask[0], (0, 0, 255), thickness=2)
        overlay = draw_contour_overlay(overlay, mask[1], (0, 255, 0), thickness=2)
        overlay = draw_contour_overlay(overlay, mask[2], (255, 0, 0), thickness=2)
        overlay = draw_contour_overlay(overlay, mask[3], (0, 255, 255), thickness=2)

        print('----t=%d---' % t)
        print('')
        print('infor\n', infor)
        print(image.shape)
        print(mask.shape)
        print('')

        # image_show('original_mask',mask,  resize=0.25)
        image_show('original_image', image, resize=0.5)
        image_show('original_overlay', overlay, resize=0.5)
        cv2.waitKey(1)

        if 1:
            for i in range(100):
                image1, mask1, infor1 = augment(image.copy(), mask.copy(), infor)

                overlay1 = image1.copy()
                overlay1 = draw_contour_overlay(overlay1, mask1[0], (0, 0, 255), thickness=2)
                overlay1 = draw_contour_overlay(overlay1, mask1[1], (0, 255, 0), thickness=2)
                overlay1 = draw_contour_overlay(overlay1, mask1[2], (255, 0, 0), thickness=2)
                overlay1 = draw_contour_overlay(overlay1, mask1[3], (0, 255, 255), thickness=2)

                # image_show_norm('mask',mask1,  resize=0.25)
                image_show('image1', image1, resize=0.5)
                image_show('overlay1', overlay1, resize=0.5)
                cv2.waitKey(1)


# main #################################################################
if __name__ == '__main__':
    print('%s: calling main function ... ' % os.path.basename(__file__))
    # generate_bev_data()
    # run_check_train_data()

    # run_check_test_dataset()

    run_check_data_loader()
    # run_check_augment()
