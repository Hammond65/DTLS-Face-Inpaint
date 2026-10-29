import copy
import torch.nn.functional as F
import numpy as np
import shutil
import os
import errno
import torch
import lpips
import wandb
import util.dnnlib as dnnlib
import scipy.linalg
import random

from typing import List
from torch import nn
from functools import partial
from torch.utils import data
from pathlib import Path
from torch.optim import Adam
from torchvision import transforms, utils
from PIL import Image
from util.discriminator import NLayerDiscriminator as discr
from torchmetrics.image import StructuralSimilarityIndexMeasure
from skimage.metrics.simple_metrics import peak_signal_noise_ratio as compare_psnr
from util.yaml import load_yaml
from util.maskGenerator import MixedMaskGenerator
from util.r1 import NonSaturatingWithR1

try:
    from apex import amp
    APEX_AVAILABLE = True
except:
    APEX_AVAILABLE = False

####### helpers functions

def cycle(dl):
    while True:
        for data in dl:
            yield data

def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr

def loss_backwards(fp16, loss, optimizer, scaler, **kwargs):
    if fp16:
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward(**kwargs)
    else:
        scaler.scale(loss).backward(**kwargs)

def normalize(img):
    return ((img * 2) - 1)
def unnormalize(img):
    return ((img + 1) / 2)
# small helper modules

class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

class FeatureStats:
    def __init__(self, capture_all=False, capture_mean_cov=False, max_items=None):
        self.capture_all = capture_all
        self.capture_mean_cov = capture_mean_cov
        self.max_items = max_items
        self.num_items = 0
        self.num_features = None
        self.all_features = None
        self.raw_mean = None
        self.raw_cov = None

    def set_num_features(self, num_features):
        if self.num_features is not None:
            assert num_features == self.num_features
        else:
            self.num_features = num_features
            self.all_features = []
            self.raw_mean = np.zeros([num_features], dtype=np.float64)
            self.raw_cov = np.zeros([num_features, num_features], dtype=np.float64)

    def is_full(self):
        return (self.max_items is not None) and (self.num_items >= self.max_items)

    def append(self, x):
        x = np.asarray(x, dtype=np.float32)
        assert x.ndim == 2
        if (self.max_items is not None) and (self.num_items + x.shape[0] > self.max_items):
            if self.num_items >= self.max_items:
                return
            x = x[:self.max_items - self.num_items]

        self.set_num_features(x.shape[1])
        self.num_items += x.shape[0]
        if self.capture_all:
            self.all_features.append(x)
        if self.capture_mean_cov:
            x64 = x.astype(np.float64)
            self.raw_mean += x64.sum(axis=0)
            self.raw_cov += x64.T @ x64

    def append_torch(self, x, num_gpus=1, rank=0):
        assert isinstance(x, torch.Tensor) and x.ndim == 2
        assert 0 <= rank < num_gpus
        if num_gpus > 1:
            ys = []
            for src in range(num_gpus):
                y = x.clone()
                torch.distributed.broadcast(y, src=src)
                ys.append(y)
            x = torch.stack(ys, dim=1).flatten(0, 1) # interleave samples
        self.append(x.cpu().numpy())

    def get_all(self):
        assert self.capture_all
        return np.concatenate(self.all_features, axis=0)

    def get_all_torch(self):
        return torch.from_numpy(self.get_all())

    def get_mean_cov(self):
        assert self.capture_mean_cov
        mean = self.raw_mean / self.num_items
        cov = self.raw_cov / self.num_items
        cov = cov - np.outer(mean, mean)
        return mean, cov

class DTLS(nn.Module):
    def __init__(
        self,
        denoise_fn,
        *,
        image_size,
        timesteps,
        device,
        stochastic=False,
        loss_config
    ):
        super().__init__()
        self.image_size = image_size
        self.denoise_fn = denoise_fn
        self.num_timesteps = int(timesteps)
        self.device = device
        self.lpips_loss_func = lpips.LPIPS(net='vgg').to(self.device)
        self.loss_config = loss_config
        if loss_config.r1 is not None:
            self.r1 = NonSaturatingWithR1(**loss_config.r1)
        else:
            self.r1 = None
        
    def ratio(self, img, length, currentStep):
        #new_masks = []
        ratio = 1/length
        # 0 -> 32 4 -> 16
        keep = ratio * currentStep
        img = img * keep
        return img
        
    def feature_matching_loss(self, fake_features: List[torch.Tensor], target_features: List[torch.Tensor], mask=None):
        if mask is None:
            res = torch.stack([F.mse_loss(fake_feat, target_feat)
                            for fake_feat, target_feat in zip(fake_features, target_features)]).mean()
        else:
            res = 0
            norm = 0
            for fake_feat, target_feat in zip(fake_features, target_features):
                cur_mask = F.interpolate(mask, size=fake_feat.shape[-2:], mode='bilinear', align_corners=False)
                error_weights = 1 - cur_mask
                cur_val = ((fake_feat - target_feat).pow(2) * error_weights).mean()
                res = res + cur_val
                norm += 1
            res = res / norm
        return res
    
    def high_pass_filter(self, image, sigma=2):
        blurred = transforms.GaussianBlur(kernel_size=133, sigma=sigma)(image)
        high_pass = image - blurred
        return high_pass

    # SAPL loss (L2 between high-pass filtered images)
    def sapl_loss(self, y_true, y_pred):
        true_sharp = self.high_pass_filter(y_true)
        pred_sharp = self.high_pass_filter(y_pred)
        return F.smooth_l1_loss(true_sharp, pred_sharp)
        
    def masked_l2_loss(self, pred, target, mask, weight_known, weight_missing):
        per_pixel_l2 = F.mse_loss(pred, target, reduction='none')
        pixel_weights = (1-mask) * weight_missing + mask * weight_known
        return (pixel_weights * per_pixel_l2).mean()
        
    def masked_l1_loss(self, pred=None, target=None, mask=None, weight_known=10, weight_missing=0):
        per_pixel_l1 = F.l1_loss(pred, target, reduction='none')
        pixel_weights = (1-mask) * weight_missing + mask * weight_known
        return (pixel_weights * per_pixel_l1).mean()  
    
    @torch.no_grad()
    def sample(self, batch_size=16, img=None, t=None, gt=None, mask=None, idx=None, result_folder=None, resample=0):
        if t == None:
            t = self.num_timesteps
        img_t = img.clone()
        inverted_mask = 1 - mask
        previous_x_s0 = None
        momentum = 0
        img_list = []
        ## Domain Transfer
        if resample == 0:
            while (t):
                step = torch.full((batch_size,), t, dtype=torch.long).to(self.device)
                
                if self.denoise_fn.channels == 4:
                    img_t = torch.cat((img_t, (1-mask)), dim=1)
                if previous_x_s0 is None:
                    R_x = self.denoise_fn(img_t, step)
                    previous_x_s0 = R_x
                else:
                    R_x = self.denoise_fn(img_t, step)
  
                momentum += previous_x_s0 - R_x
                inpaint_part = (R_x * inverted_mask) + (momentum * inverted_mask)
                recon_x_percentage = self.ratio(img=((R_x.clone()*mask)+(momentum*mask)), length=self.num_timesteps, currentStep=t-1)
                original_percentage = self.ratio(img=gt.clone(), length=self.num_timesteps, currentStep=(self.num_timesteps-(t-1)))
                inpaint_part = self.ratio(img=inpaint_part.clone(), length=self.num_timesteps, currentStep=(self.num_timesteps-(t-1)))
                img_t = recon_x_percentage + original_percentage + inpaint_part
                img_list.append(img_t.clone())
                previous_x_s0 = R_x
                t -= 1
        else:
            for _ in range(resample):
                while (t):
                    step = torch.full((batch_size,), t, dtype=torch.long).to(self.device)
                    R_x = self.denoise_fn(img_t, step)
                    t -= 1
                t = self.num_timesteps
        return img_t, img_list

    def p_losses(self, x_start, t, input_data, mask):
        input_percentage = input_data.clone()
        x_start_percentage = x_start.clone()
        current_img = x_start.clone()
        for i in range(t.shape[0]):
            input_percentage[i] = self.ratio(img=input_data.clone()[i], length=self.num_timesteps, currentStep=t[i].item())
            x_start_percentage[i] = self.ratio(img=x_start.clone()[i], length=self.num_timesteps, currentStep=(self.num_timesteps - t[i].item()))
            if t[i] == 20:
                current_img[i] = input_data.clone()[i]
            else:
                current_img[i] = input_percentage[i] + x_start_percentage[i]

        if self.denoise_fn.channels == 4:
            current_img = torch.cat((current_img, (1-mask)), dim=1)
        x_recon = self.denoise_fn(current_img, t)

        mse_loss = None
        l1_value = None
        lpips_loss = None
        sapl_loss = None
        # Compute losses
        if self.loss_config.l2_loss:
            mse_loss = self.masked_l2_loss(pred=x_recon, target=x_start, mask=mask, weight_known=0, weight_missing=self.loss_config.l2_weight)
        if self.loss_config.l1_loss:
            l1_value = self.masked_l1_loss(pred=x_recon, target=x_start, mask=mask, weight_known=self.loss_config.l1_weight, weight_missing=0)
        if self.loss_config.lpips_loss:
            lpips_loss = self.lpips_loss_func(x_recon, x_start).to(self.device).mean() * self.loss_config.lpips_weight
        if self.loss_config.sapl_loss:
            sapl_loss = self.sapl_loss(x_recon, x_start) * self.loss_config.sapl_weight
            
        losses = {
            'l1_value': l1_value,
            'mse_loss': mse_loss,
            'lpips_loss': lpips_loss,
            'sapl_loss': sapl_loss,
        }
        output_setting = {
            'x_recon': x_recon,
            'lpips_loss_func': self.lpips_loss_func,
        }
    
        return losses, output_setting
    def forward(self, x, input_data, mask, *args, **kwargs):
        b, c, h, w, device, img_size, = *x.shape, x.device, self.image_size
        assert h == img_size and w == img_size, f'height and width of image must be {img_size}'
        t1 = torch.randint(1, self.num_timesteps + 1, (b,), device=device).long()
        t2 = torch.randint(20, self.num_timesteps + 1, (b,), device=device).long()
        t = random.choice([t1, t2, t2])
        return self.p_losses(x, t, input_data, mask, *args, **kwargs)

class Dataset(data.Dataset):
    def __init__(self, folder=None, mask_folder=None, image_size=None, exts = ['jpg', 'jpeg', 'png']):
        super().__init__()
        self.folder = folder
        self.mask_folder = mask_folder
        self.image_size = image_size
        if self.folder != None:
            self.paths = [p for ext in exts for p in Path(f'{folder}').glob(f'**/*.{ext}')]
        if self.mask_folder!= None:
            self.paths_mask = [p for ext in exts for p in Path(f'{mask_folder}').glob(f'**/*.{ext}')]

        self.transform = transforms.Compose([
            transforms.Resize(int(image_size)),
            transforms.RandomCrop(int(image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.3, hue=0.05),
            transforms.ToTensor(),
            transforms.Lambda(lambda t: (t * 2) - 1)
        ])
        
        self.mask_transform = transforms.Compose([
            transforms.Resize((int(image_size), int(image_size))),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.ToTensor(),
        ])

    def __len__(self):
        if self.mask_folder != None:
            return len(self.paths_mask)
        if self.folder != None:
            return len(self.paths)

    def __getitem__(self, index):
        if self.folder != None:
            path = self.paths[index]
            img = Image.open(path)
        if self.mask_folder != None:
            path = self.paths_mask[index]
            mask = Image.open(path)
        
        if self.folder == None:
            return self.mask_transform(mask)
        elif self.mask_folder == None:
            img = self.transform(img)
            # img = img['image']
            # img = self.normalize(img)
            if img.size(0) == 1:  # Grayscale image
                img = img.expand(3, -1, -1)  # Expand to 3 channels
            return img
        return self.transform(image=img)['image'], self.mask_transform(mask)


# trainer class

class Trainer(object):
    def __init__(
        self,
        diffusion_model,
        data_path,
        mask_path,
        *,
        ema_decay = 0.995,
        image_size = 128,
        train_batch_size = 32,
        train_lr = 2e-5,
        train_num_steps = 100000,
        gradient_accumulate_every = 1,
        fp16 = False,
        step_start_ema = 2000,
        update_ema_every = 10,
        save_and_sample_every = 1000,
        input_image,
        results_folder,
        load_path = None,
        sr_load_path = None,
        inpaint_load_path = None,
        shuffle=True,
        device,
        mask_generator_config,
        discriminator_config,
        eval_config=None,
        fast_test=False,
    ):
        super().__init__()
        self.mask_generator = MixedMaskGenerator(**load_yaml(mask_generator_config).mask_generator_kwargs)
        if discriminator_config != None:
            discriminator_config = load_yaml(discriminator_config)
        train_lr = float(train_lr)
        self.eval_config = eval_config
        self.model = diffusion_model
        self.fast_test = fast_test
        self.ema = EMA(ema_decay)

        self.image_size = diffusion_model.image_size

        if discriminator_config != None:    
            self.discriminator = discr(**discriminator_config.settings).to(device)

        self.ema_model = copy.deepcopy(self.model)
        self.update_ema_every = update_ema_every

        self.step_start_ema = step_start_ema
        self.save_and_sample_every = save_and_sample_every

        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every
        self.train_num_steps = train_num_steps
        self.nrow = 5
        self.metrics_list = []
        self.input_image = input_image

        self.ds_img = Dataset(folder=data_path, image_size=image_size)

        self.dl_img = cycle(data.DataLoader(self.ds_img, batch_size = train_batch_size, shuffle=shuffle, pin_memory=False, num_workers=30, drop_last=True))
        self.opt = Adam(diffusion_model.parameters(), lr=train_lr, betas=(0.9, 0.999), eps=1e-8)
        if discriminator_config != None:    
           self.opt_d = Adam(self.discriminator.parameters(), lr=train_lr, betas=(0.9, 0.999), eps=1e-8)
           
        self.BCE_loss = torch.nn.BCEWithLogitsLoss(size_average=True,reduction='none')

        self.scaler = torch.cuda.amp.GradScaler()
        self.step = 0

        self.device = device
        assert not fp16 or fp16 and APEX_AVAILABLE, 'Apex must be installed in order for mixed precision training to be turned on'

        self.fp16 = fp16
        if fp16:
            (self.model, self.ema_model), self.opt = amp.initialize([self.model, self.ema_model], self.opt, opt_level='O1')

        self.lpips_loss_func = lpips.LPIPS(net='vgg').to(self.device)
        self.ssim = StructuralSimilarityIndexMeasure().to(self.device)

        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(parents=True, exist_ok=True)

        self.reset_parameters()
        
        self.best_quality = 0
        if load_path != None:
            self.load(load_path)

    def reset_parameters(self):
        self.ema_model.load_state_dict(self.model.state_dict())

    def step_ema(self):
        if self.step < self.step_start_ema:
            self.reset_parameters()
            return
        self.ema.update_model_average(self.ema_model, self.model)

    def save_best(self):
        data = {
            'step': self.step,
            'model': self.model.state_dict(),
            'ema': self.ema_model.state_dict(),
        }
        if hasattr(self, 'discriminator'):
            data['dis'] = self.discriminator.state_dict()
        torch.save(data, str(self.results_folder / f'model_best.pt'))

    def save_last(self):
        data = {
            'step': self.step,
            'model': self.model.state_dict(),
            'ema': self.ema_model.state_dict(),
        }
        if hasattr(self, 'discriminator'):
            data['dis'] = self.discriminator.state_dict()
        torch.save(data, str(self.results_folder / f'model_last.pt'))

    def load(self, load_path):
        print("Loading : ", load_path)
        data = torch.load(load_path, map_location=self.device)
        self.step = data['step']
        self.model.load_state_dict(data['model'], strict=False)
        self.ema_model.load_state_dict(data['ema'], strict=False)
        
    def img2tensor(self, img_path=None, img: Image = None):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda t: (t * 2) - 1)
        ])
        mask_transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        if img_path is not None:
            img = Image.open(img_path)
            if img.mode == 'L':
                img = img.convert('L')
                tensor_img = mask_transform(img)
            elif img.mode == 'RGB':
                img = img.convert('RGB')
                tensor_img = transform(img)
            else:
                img = img.convert('RGB')
                tensor_img = transform(img)
        else:
            if img.mode == 'L':
                img = img.convert('L')
                tensor_img = mask_transform(img)
            elif img.mode == 'RGB':
                img = img.convert('RGB')
                tensor_img = transform(img)
            else:
                img = img.convert('RGB')
                tensor_img = transform(img)
        return tensor_img
    
    def log_losses(self, loss_dict):
        clean_losses = {k: v for k, v in loss_dict.items() if v is not None}
        
        if clean_losses:
            wandb.log(clean_losses)
            
    def get_feature_detector(self, url, device=torch.device('cpu'), num_gpus=1, rank=0, verbose=False, feature_detector_cache=dict()):
        assert 0 <= rank < num_gpus
        key = (url, device)
        if key not in feature_detector_cache:
            is_leader = (rank == 0)
            if not is_leader and num_gpus > 1:
                torch.distributed.barrier()
            with dnnlib.util.open_url(url, verbose=(verbose and is_leader)) as f:
                feature_detector_cache[key] = torch.jit.load(f).eval().to(device)
            if is_leader and num_gpus > 1:
                torch.distributed.barrier()
        return feature_detector_cache[key]
            
    def fid_get_stat(self, img, gt, stat1, stat2, feature_detector_cache):
        detector_url = 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/inception-2015-12-05.pt'
        detector_kwargs = dict(return_features=True)
        device = img.device
        detector = self.get_feature_detector(url=detector_url, device=device, num_gpus=1, rank=0, verbose=False, feature_detector_cache=feature_detector_cache)
        detector.eval()
        fea1 = detector(img, **detector_kwargs)
        stat1.append_torch(fea1, num_gpus=1, rank=0)
        fea2 = detector(gt, **detector_kwargs)
        stat2.append_torch(fea2, num_gpus=1, rank=0)
        
        return stat1, stat2

    def fid_calculate(self, stat1, stat2):
        mu1, sigma1 = stat1.get_mean_cov()
        mu2, sigma2 = stat2.get_mean_cov()
        m = np.square(mu1 - mu2).sum()
        s, _ = scipy.linalg.sqrtm(np.dot(sigma1, sigma2), disp=False)
        fid = np.real(m + np.trace(sigma1 + sigma2 - s * 2))
        return fid

    def train(self):
        backwards = partial(loss_backwards, self.fp16)

        acc_loss = 0
        best_lpips = 100
        while self.step < self.train_num_steps:
            if hasattr(self, 'discriminator'):
                if self.model.r1 is not None:
                    actual_losses=dict()
                    for i in range(self.gradient_accumulate_every):
                        data = next(self.dl_img)
                        data = data.to(self.device)
                        mask_data = []
                        for _ in range(data.shape[0]):
                            mask_data.append(torch.from_numpy(self.mask_generator(data[0].unsqueeze(0))))
                        mask_data = torch.stack(mask_data).to(self.device)
                        input_data = data * mask_data
                        
                        losses = None
                        losses, output_setting = self.model(data, input_data, mask_data)
                        x_recon = output_setting['x_recon']
                        
                        if self.discriminator.channels == 7:
                            input = torch.cat([data, input_data, mask_data], dim=1)
                            d_generated = torch.cat([x_recon.detach(), input_data, mask_data], dim=1)
                        else:
                            input = data
                            d_generated = x_recon.detach()
                            
                        self.model.r1.pre_discriminator_step(input)
                        score_true, _ = self.discriminator(input)
                        score_false, _ = self.discriminator(d_generated)
                        
                        loss_dis, _ = self.model.r1.discriminator_loss(input,d_generated,score_true,score_false, (1-mask_data))
                        
                        backwards(loss_dis/ self.gradient_accumulate_every, self.opt_d, scaler=self.scaler)
                        if self.discriminator.channels == 7:
                            g_generated = torch.cat([x_recon, input_data, mask_data], dim=1)
                        else:
                            g_generated = x_recon
                            
                        _, gt_feature = self.discriminator(input)
                        score_fake, fake_feature = self.discriminator(g_generated)
                        loss_gen, _ = self.model.r1.generator_loss(score_fake)
                        feature_matching_loss = self.model.feature_matching_loss(fake_feature, gt_feature) * self.model.loss_config.feature_matching_weight
                        
                        losses['feature_matching_loss'] = feature_matching_loss
                        losses['gen_loss'] = loss_gen
                        
                        total_loss = 0.0
                        for loss_name, loss_value in losses.items():
                            if loss_value is not None:
                                total_loss += loss_value

                        total_loss = (total_loss/self.gradient_accumulate_every)
                        backwards(total_loss, self.opt, scaler=self.scaler)
                        
                        for param in output_setting['lpips_loss_func'].parameters():
                            param.grad = None
                        
                        losses['loss_dis'] = loss_dis
                        
                        for loss_name, loss_value in losses.items():
                                if loss_value is not None:
                                    if loss_name not in actual_losses:
                                        actual_losses[loss_name] = (loss_value / self.gradient_accumulate_every)
                                    else:
                                        actual_losses[loss_name] += (loss_value / self.gradient_accumulate_every)
                        
                    output = ', '.join(f"{key}: {value}" for key, value in actual_losses.items())
                    print(f"step: {self.step}", output)
                    try:
                        self.log_losses(actual_losses)
                    except:
                        pass
                    self.scaler.step(self.opt)
                    self.scaler.step(self.opt_d)
                    self.scaler.update()
                    self.opt.zero_grad()
                    self.opt_d.zero_grad()
                else:
                    actual_losses=dict()
                    for i in range(self.gradient_accumulate_every):
                        data = next(self.dl_img)
                        data = data.to(self.device)
                        mask_data = []
                        for _ in range(data.shape[0]):
                            mask_data.append(torch.from_numpy(self.mask_generator(data[0].unsqueeze(0))))
                        mask_data = torch.stack(mask_data).to(self.device)
                        input_data = data * mask_data
                        
                        input = torch.cat([data, input_data, mask_data], dim=1)
                        score_true, _ = self.discriminator(input)
                        
                        GAN_true = torch.ones_like(score_true)
                        loss_dis_true = self.BCE_loss(score_true, GAN_true)
                            
                        backwards(loss_dis_true/ self.gradient_accumulate_every, self.opt_d, scaler=self.scaler)
                        losses = None
                        losses, output_setting = self.model(data, input_data, mask_data)
                        
                        x_recon = output_setting['x_recon']
                        
                        d_generated = torch.cat([x_recon.detach(), input_data, mask_data], dim=1)
                        
                        score_false, _ = self.discriminator(d_generated)
                        
                        GAN_false = torch.zeros_like(score_false)
                        loss_dis_false = self.BCE_loss(score_false, GAN_false)
                            
                        backwards(loss_dis_false/ self.gradient_accumulate_every, self.opt_d, scaler=self.scaler)
                        _, gt_feature = self.discriminator(input)
                        
                        g_generated = torch.cat([x_recon, input_data, mask_data], dim=1)
                        
                        score_fake, fake_feature = self.discriminator(g_generated)
                        GAN_fake = torch.ones_like(score_fake)
                        loss_gen = self.BCE_loss(score_fake, GAN_fake) * 1e-1
                        feature_matching_loss = F.smooth_l1_loss(fake_feature[-1],gt_feature[-1]) * 0.2
                        losses['feature_matching_loss'] = feature_matching_loss
                        losses['gen_loss'] = loss_gen
                        
                        total_loss = 0.0
                        for loss_name, loss_value in losses.items():
                            if loss_value is not None:
                                total_loss += loss_value
                        total_loss = (total_loss/self.gradient_accumulate_every)
                        backwards(total_loss, self.opt, scaler=self.scaler)
                        
                        for param in output_setting['lpips_loss_func'].parameters():
                            param.grad = None
                        
                        losses['dis_gt_loss'] = loss_dis_true
                        losses['dis_fake_loss'] = loss_dis_false
                        
                        for loss_name, loss_value in losses.items():
                                if loss_value is not None:
                                    if loss_name not in actual_losses:
                                        actual_losses[loss_name] = (loss_value / self.gradient_accumulate_every)
                                    else:
                                        actual_losses[loss_name] += (loss_value / self.gradient_accumulate_every)
                        
                    output = ', '.join(f"{key}: {value}" for key, value in actual_losses.items())
                    print(f"step: {self.step}", output)
                    try:
                        self.log_losses(actual_losses)
                    except:
                        print("wandb is not initialized. Skipping wandb logging.")
                    self.scaler.step(self.opt)
                    self.scaler.step(self.opt_d)
                    self.scaler.update()
                    self.opt.zero_grad()
                    self.opt_d.zero_grad()
            else:
                with torch.cuda.amp.autocast():
                    data = next(self.dl_img)
                    data = data.to(self.device)
                    mask_data = []
                    for _ in range(data.shape[0]):
                        mask_data.append(torch.from_numpy(self.mask_generator(data[0].unsqueeze(0))))
                    mask_data = torch.stack(mask_data).to(self.device)
                    input_data = data * mask_data

                    losses, output_setting = self.model(data, input_data, mask_data)
                    
                    total_loss = 0.0
                    for loss_name, loss_value in losses.items():
                        if loss_value is not None:
                            total_loss += loss_value
                    total_loss = (total_loss/self.gradient_accumulate_every)
                backwards(total_loss, self.opt, scaler=self.scaler)
                
                for param in output_setting['lpips_loss_func'].parameters():
                    param.grad = None
                    
                output = ', '.join(f"{key}: {value}" for key, value in losses.items())
                print(f"step: {self.step}", output)
                try:
                    self.log_losses(losses)
                except:
                    print("wandb is not initialized. Skipping wandb logging.")
                self.scaler.step(self.opt)
                self.scaler.update()
                self.opt.zero_grad()
                
            if self.step % self.update_ema_every == 0:
                self.step_ema()

            if self.step == 0 or self.step % self.save_and_sample_every == 0:
                list_img = []
                list_mask = []
                list_img = [os.path.join(self.input_image,img) for img in os.listdir(self.input_image) if (img.startswith("image"))]
                list_mask = [os.path.join(self.input_image,img) for img in os.listdir(self.input_image) if (img.startswith("mask"))]
                list_img.sort()
                list_mask.sort()
                total_psnr = 0
                total_ssim = 0
                total_lpips = 0
                averagePSNR = 0
                averagessim = 0
                averagelpips = 0
                hq_img_set = torch.tensor([])
                gt_img_set = torch.tensor([])
                mask_img_set = torch.tensor([])
                stat1 = FeatureStats(capture_all=True, capture_mean_cov=True, max_items=len(list_img))
                stat2 = FeatureStats(capture_all=True, capture_mean_cov=True, max_items=len(list_img))
                feature_detector_cache = dict()
                for idx, (img_path, mask_path) in enumerate(zip(list_img, list_mask)):
                    imgname = os.path.splitext(os.path.basename(img_path))[0]
                    maskname = os.path.splitext(os.path.basename(mask_path))[0]
                    if (idx % 10) == 0:
                        print(imgname, maskname)
                    # read image
                    gt = self.img2tensor(img_path).unsqueeze(0).to(self.device)
                    mask = self.img2tensor(mask_path).unsqueeze(0).to(self.device)
                    img = gt
                    masked_img = img * mask
                    with torch.no_grad():
                        img_sr, _ = self.ema_model.sample(batch_size=1, img=masked_img, gt=masked_img, mask=mask, idx=idx, result_folder=self.results_folder)

                        img_sr = unnormalize(img_sr)
                        if idx <= 32:
                            hq_img_set = torch.cat((hq_img_set, img_sr.to("cpu")), dim=0)
                            gt_img_set = torch.cat((gt_img_set, unnormalize(gt).to("cpu")), dim=0)
                            mask_img_set = torch.cat((mask_img_set, unnormalize(masked_img).to("cpu")), dim=0)
                        
                        psnrScore = compare_psnr(np.clip(unnormalize(gt).detach().cpu().numpy().astype(np.float32),0,1), np.clip(img_sr.detach().cpu().numpy().astype(np.float32),0,1))
                        ssimScore = self.ssim(img_sr,unnormalize(gt))
                        lpipScore = self.lpips_loss_func(normalize(img_sr),(gt))
                        stat1, stat2 = self.fid_get_stat((img_sr), unnormalize(gt), stat1, stat2, feature_detector_cache)
                        total_psnr += psnrScore
                        total_ssim += ssimScore
                        total_lpips += lpipScore
                        gt = None
                        if self.fast_test:
                            if idx == 2:
                                break

                averagePSNR = total_psnr / len(list_img)
                averagessim = total_ssim / len(list_img)
                averagelpips = total_lpips / len(list_img)
                averagefid = self.fid_calculate(stat1,stat2)
                utils.save_image(hq_img_set, str(self.results_folder / f'{self.step}_Output.png'), nrow=hq_img_set.shape[0])
                utils.save_image(mask_img_set, str(self.results_folder / f'{self.step}_MASK.png'), nrow=hq_img_set.shape[0])
                utils.save_image(gt_img_set, str(self.results_folder / f'{self.step}_GT.png'), nrow=hq_img_set.shape[0])
                hq_img_set = None
                gt_img_set = None
                mask_img_set = None
                try:
                    wandb.log({"PSNR": averagePSNR, "SSIM": averagessim, "LPIPS": averagelpips, "FID": averagefid})
                except:
                    print("wandb is not initialized. Skipping wandb logging.")
                self.metrics_list.append(f"PSNR: {averagePSNR}, SSIM: {averagessim}, LPIPS: {averagelpips.item()}, FID: {averagefid}")
                if averagelpips < best_lpips:
                    best_lpips = averagelpips
                    self.save_best()
                file = open(f"{self.results_folder}/quality.txt", 'w')
                for line in self.metrics_list:
                    file.write(line + "\n")
                file.close()

                print(f'Mean of last {self.step}: {acc_loss}')
                self.save_last()
                acc_loss = 0

            self.step += 1
        print('training completed')

    def evaluation(self):
        list_img = []
        list_mask = []
        list_img = [os.path.join(self.input_image,img) for img in os.listdir(self.input_image) if (img.startswith("image"))]
        list_mask = [os.path.join(self.input_image,img) for img in os.listdir(self.input_image) if (img.startswith("mask"))]
        list_img.sort()
        list_mask.sort()
        total_psnr = 0
        total_ssim = 0
        total_lpips = 0
        averagePSNR = 0
        averagessim = 0
        averagelpips = 0
        hq_img_set = torch.tensor([])
        gt_img_set = torch.tensor([])
        mask_img_set = torch.tensor([])
        lpips_loss_func = lpips.LPIPS(net='vgg').to(self.device)
        ssim = StructuralSimilarityIndexMeasure().to(self.device)
        for idx, (img_path, mask_path) in enumerate(zip(list_img, list_mask)):
            imgname = os.path.splitext(os.path.basename(img_path))[0]
            maskname = os.path.splitext(os.path.basename(mask_path))[0]
            print(imgname, maskname)
            img = self.img2tensor(img_path).unsqueeze(0).to(self.device)
            mask = self.img2tensor(mask_path).unsqueeze(0).to(self.device)
            gt = img
            masked_img = img * mask
            with torch.no_grad():
                refine_img, _ = self.ema_model.sample(batch_size=1, img=masked_img, gt=masked_img, mask=mask, idx=idx, result_folder=self.results_folder)
                lpips_refine_img = refine_img
                refine_img = unnormalize(refine_img)
                
                hq_img_set = torch.cat((hq_img_set, refine_img.to("cpu")), dim=0)
                if self.eval_config.save_final_results == True:
                    gt_img_set = torch.cat((gt_img_set, unnormalize(gt).to("cpu")), dim=0)
                    mask_img_set = torch.cat((mask_img_set, unnormalize(masked_img).to("cpu")), dim=0)
                utils.save_image(refine_img, str(self.results_folder / f'{imgname}.png'))
                utils.save_image((unnormalize(img)* mask), str(self.results_folder / f'{maskname}.png'))
                ssimScore = ssim(refine_img,unnormalize(gt))
                lpipScore = lpips_loss_func(lpips_refine_img,gt)
                psnrScore = compare_psnr(np.clip(unnormalize(gt).detach().cpu().numpy().astype(np.float32),0,1), np.clip(refine_img.detach().cpu().numpy().astype(np.float32),0,1))
                total_psnr += psnrScore
                total_ssim += ssimScore
                total_lpips += lpipScore
                gt = None

        averagePSNR = total_psnr / hq_img_set.shape[0]
        averagessim = total_ssim / hq_img_set.shape[0]
        averagelpips = total_lpips / hq_img_set.shape[0]
        if self.eval_config.save_final_results == True:
            utils.save_image(hq_img_set, str(self.results_folder / f'{self.step}_FFHQ.png'), nrow=hq_img_set.shape[0])
            utils.save_image(mask_img_set, str(self.results_folder / f'{self.step}_MASK.png'), nrow=hq_img_set.shape[0])
            utils.save_image(gt_img_set, str(self.results_folder / f'{self.step}_GT.png'), nrow=hq_img_set.shape[0])

        self.metrics_list.append(f"PSNR: {averagePSNR}, SSIM: {averagessim}, LPIPS: {averagelpips.item()}")
        file = open(f"{self.results_folder}/quality.txt", 'w')
        for line in self.metrics_list:
            file.write(line + "\n")
        file.close()
    
    def single_evaluation(self):
        img = sorted([os.path.join(self.input_image,img) for img in os.listdir(self.input_image) if (img.startswith("image"))])[0]
        mask = sorted([os.path.join(self.input_image,img) for img in os.listdir(self.input_image) if (img.startswith("mask"))])[0]
        
        lpips_loss_func = lpips.LPIPS(net='vgg').to(self.device)
        ssim = StructuralSimilarityIndexMeasure().to(self.device)
        imgname = os.path.splitext(os.path.basename(img))[0]
        maskname = os.path.splitext(os.path.basename(mask))[0]
        print(imgname, maskname)
        # read image
        img = self.img2tensor(img).unsqueeze(0).to(self.device)
        mask = self.img2tensor(mask).unsqueeze(0).to(self.device)
        
        gt = img
        masked_img = img * mask
        with torch.no_grad():
            refine_img, _ = self.ema_model.sample(batch_size=1, img=masked_img, gt=masked_img, mask=mask, result_folder=self.results_folder)
            refine_img = unnormalize(refine_img)
            utils.save_image(refine_img, str(self.results_folder / f'{imgname}.png'))
            utils.save_image(unnormalize(img)*mask, str(self.results_folder / f'{imgname}_mask.png'))
            ssimScore = ssim(refine_img,unnormalize(gt))
            lpipScore = lpips_loss_func(refine_img,unnormalize(gt))
            psnrScore = compare_psnr(np.clip(unnormalize(gt).detach().cpu().numpy().astype(np.float32),0,1), np.clip(refine_img.detach().cpu().numpy().astype(np.float32),0,1))
            print(f"PSNR: {psnrScore} SSIM: {ssimScore} LPIPS: {lpipScore}")

    def test(self, input_image, mask):
        img = self.img2tensor(img=input_image).unsqueeze(0).to(self.device)
        mask = self.img2tensor(img=mask).unsqueeze(0).to(self.device)
        masked_img = img * mask
        with torch.no_grad():
            refine_img, img_list = self.ema_model.sample(batch_size=1, img=masked_img, gt=masked_img, mask=mask, result_folder="./results/")
        refine_img = unnormalize(refine_img)
        refine_img = torch.clamp(refine_img, min=0, max=1)
        unnorm_img_list = [unnormalize(img) for img in img_list]
        img_list = [torch.clamp(img, min=0, max=1) for img in unnorm_img_list]
        return transforms.ToPILImage()(refine_img.squeeze(0)), [transforms.ToPILImage()(img.squeeze(0)) for img in img_list]