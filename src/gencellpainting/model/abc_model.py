from gencellpainting.model.net.conv_modules import Conv2dStack,Conv2dTransposeStack
from gencellpainting.evaluation.clip_fih import FrechetCLIPDistance
import torch
import lightning as L
from torchmetrics import Recall,Precision


from gencellpainting.plotting import multi_channel_tensor_to_flat_matrix


class UnsupervisedImageGenerator(L.LightningModule):
    """
    Parent class for image generation, mainly including tensorboard image logging.
    """
    def __init__(self, epoch_monitoring_interval=None, n_images_monitoring=6, add_original=True):
        super(UnsupervisedImageGenerator, self).__init__()
        self.epoch_monitoring_interval = epoch_monitoring_interval
        self.last_monitored_epoch = -1
        self.n_images_monitoring = n_images_monitoring
        self.add_original = add_original
        # self.clip_frechet_distance = FrechetCLIPDistance()

    def training_step(self, batch, batch_idx):
        self.monitor_training(batch)

    def configure_optimizers(self):
        raise NotImplementedError("Method need to be implemented")

    def monitor_training(self, batch):
        if self.current_epoch % self.epoch_monitoring_interval == 0 and \
            self.last_monitored_epoch != self.current_epoch:
            self.last_monitored_epoch = self.current_epoch
            Nchannel = batch.size(1)
            min_dim = min(self.n_images_monitoring,batch.size(0))
            sub_batch = batch[:min_dim,:,:,:]
            images = self.generate_images(batch = sub_batch, n=min_dim)

            print("Monitored batch range {}-{} gen range {}-{}".format(sub_batch.min(),sub_batch.max(),images.min(),\
                                                                       images.max(),))

            # print("Flag value {}".format(self.add_original))
            if self.add_original: # we add the original
                # print("Batch shape {} gene images shape {}".format(batch.size(),images.size()))
                images = torch.cat([images,sub_batch],dim=1)

            B, C, H, W = images.size()

            images = images.view(B * C, H, W)

            self.logger.experiment.add_image("imgs",multi_channel_tensor_to_flat_matrix(images,nrow=Nchannel),self.current_epoch)
            
            # split_images = torch.chunk(images,images.size(0),dim=0)

            # # Saving the images
            # for i,img in enumerate(split_images):
            #     name_image = "img{}".format(i)
            #     self.logger.experiment.add_image(name_image,multi_channel_tensor_to_flat_matrix(img.squeeze(),nrow=Nchannel),self.current_epoch)
            
            self.compute_metrics(batch)

    def compute_metrics(self,batch):
        pass
        # nimages = batch.size(0)
        # fake_images = self.generate_images(batch = batch, n=nimages)
        # self.clip_frechet_distance.update(batch,is_real=True)
        # self.clip_frechet_distance.update(fake_images,is_real=False)
        # score = self.clip_frechet_distance.compute()
        # self.log("FrechetCLIPDistance",float(score))
        # self.clip_frechet_distance.reset()

    def generate_images(self, batch=None, n=6):
        raise NotImplementedError("Method need to be implemented")
    

class AbstractGAN(UnsupervisedImageGenerator):
    """Standard Unsupervised image Generator with the addition of discriminator specfic metrics"""
    def __init__(self, epoch_monitoring_interval=None, n_images_monitoring=6, add_original=True):
        super().__init__(epoch_monitoring_interval, n_images_monitoring, add_original)
        self.precision = Precision(task="binary")
        self.recall = Recall(task="binary")
    
    def training_step(self, batch, batch_idx, preds=None, targets=None):
        if preds is not None:
            self.precision.update(preds,targets)
            prec_score = self.precision.compute()
            self.log("Precision",float(prec_score))
            self.precision.reset()
            self.recall.update(preds,targets)
            rec_score = self.recall.compute()
            self.log("Recall",float(rec_score))
            self.precision.reset()
        self.monitor_training(batch)

