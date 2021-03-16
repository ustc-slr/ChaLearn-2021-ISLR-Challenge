import cv2
import numpy as np
import torch
from torchvision.transforms import transforms
from .models.hrnet import HRNet
import torch.nn as nn


class SimpleHRNet2:
    """
    SimpleHRNet class.
    The class provides a simple and customizable method to load the HRNet network, load the official pre-trained
    weights, and predict the human pose on single images.
    Multi-person support with the YOLOv3 detector is also included (and enabled by default).
    """

    def __init__(self,
                    c,
                    nof_joints,
                    checkpoint_path,
                    resolution=(384, 288),
                    interpolation=cv2.INTER_CUBIC,
                    multiperson=True,
                    max_batch_size=32,
                    device=torch.device("cpu")):
        """
        Initializes a new SimpleHRNet object.
        HRNet (and YOLOv3) are initialized on the torch.device(``device``) and
        its (their) pretrained weights will be loaded from disk.

        Arguments:
            c (int): number of channels.
            nof_joints (int): number of joints.
            checkpoint_path (str): hrnet checkpoint path.
            resolution (tuple): hrnet input resolution - format: (height, width).
                Default: ``(384, 288)``
            interpolation (int): opencv interpolation algorithm.
                Default: ``cv2.INTER_CUBIC``
            multiperson (bool): if ``True``, multiperson detection will be enabled.
                This requires the use of a people detector (like YOLOv3).
                Default: ``True``
            max_batch_size (int): maximum batch size used in hrnet inference.
                Useless without multiperson=True.
                Default: ``16``
            device (:class:`torch.device`): the hrnet (and yolo) inference will be run on this device.
                Default: ``torch.device("cpu")``
        """

        self.c = c
        self.nof_joints = nof_joints
        self.checkpoint_path = checkpoint_path
        self.resolution = resolution  # in the form (height, width) as in the original implementation
        self.interpolation = interpolation
        self.multiperson = multiperson
        self.max_batch_size = max_batch_size
        self.device = device

        self.model = HRNet(c=c, nof_joints=nof_joints).to(device)
        self.model.load_state_dict(torch.load(checkpoint_path))
        self.model = nn.DataParallel(self.model)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def predict(self, images):
        """
        Predicts the human pose on a single image.

        Arguments:
            image (:class:`np.ndarray`):
                the image on which the human pose will be estimated.

                Image must be in the opencv format, i.e. shape=(height, width, BGR color channel).

        Returns:
            `:class:np.ndarray`:
                a numpy array containing human joints for each (detected) person.

                Format: shape=(# of people, # of joints (nof_joints), 3);  dtype=(np.float32).

                Each joint has 3 values: (x position, y position, joint confidence)
        """
        old_res = [384, 288]

        images = images.to(self.device)
        
        with torch.no_grad():
            if len(images) <= self.max_batch_size:
                out = self.model(images)

            else:
                out = torch.empty(
                    (images.shape[0], self.nof_joints, self.resolution[0] // 4, self.resolution[1] // 4)
                ).to(self.device)
                for i in range(0, len(images), self.max_batch_size):
                    out[i:i+self.max_batch_size] = self.model(images[i:i+self.max_batch_size])

        out = out.detach().cpu().numpy()
        pts = np.empty((out.shape[0], out.shape[1], 3), dtype=np.float32)
        # For each human, for each joint: x, y, confidence
        for i, human in enumerate(out):
            for j, joint in enumerate(human):
                pt = np.unravel_index(np.argmax(joint), shape=(self.resolution[0] // 4, self.resolution[1] // 4))
                # 0: pt_x / (width // 4) * (bb_x2 - bb_x1) + bb_x1
                # 1: pt_y / (height // 4) * (bb_y2 - bb_y1) + bb_y1
                # 2: confidences
                # 384 = 96*4, 288 = 72*4
                pts[i, j, 0] = pt[0] * 1. /96.0
                pts[i, j, 1] = pt[1] * 1. /72.0
                pts[i, j, 2] = joint[pt]

        return pts       