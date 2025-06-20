from conv_modules import Conv2dStack,Conv2dTransposeStack
import torch
import lightning as L

from plotting import multi_channel_tensor_to_flat_matrix


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
    

    def training_step(self, batch, batch_idx):
        self.monitor_training(batch)


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def monitor_training(self, batch):
        if self.current_epoch % self.epoch_monitoring_interval == 0 and \
            self.last_monitored_epoch != self.current_epoch:

            self.last_monitored_epoch = self.current_epoch

            Nchannel = batch.size(1)

            min_dim = min(self.n_images_monitoring,batch.size(0))

            sub_batch = batch[:min_dim,:,:,:]

            images = self.generate_images(batch = sub_batch, n=min_dim)

            # print("Flag value {}".format(self.add_original))
            if self.add_original: # we add the original
                # print("Batch shape {} gene images shape {}".format(batch.size(),images.size()))
                images = torch.cat([images,sub_batch],dim=1)
            split_images = torch.chunk(images,images.size(0),dim=0)

            # Saving the images
            for i,img in enumerate(split_images):
                name_image = "img{}".format(i)
                self.logger.experiment.add_image(name_image,multi_channel_tensor_to_flat_matrix(img.squeeze(),nrow=Nchannel),self.current_epoch)


    def generate_images(self, batch=None, n=6):
        raise NotImplementedError("Method need to be implemented")