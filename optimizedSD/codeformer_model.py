import os
import sys
import traceback

import cv2
import torch
model_url = 'https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth'

from torchvision.transforms.functional import normalize
from codeformer.codeformer_arch import CodeFormer
from basicsr.utils.download_util import load_file_from_url
from basicsr.utils import imwrite, img2tensor, tensor2img
from facexlib.utils.face_restoration_helper import FaceRestoreHelper

from codeformer.codeformer_arch import CodeFormer

class FaceRestorerCodeFormer():

    def __init__(self, dirname="."):
        self.net = None
        self.face_helper = None
        self.cmd_dir = dirname

    def create_models(self):

        if self.net is not None and self.face_helper is not None:
            self.net.to("cuda")
            return self.net, self.face_helper
        net = CodeFormer(dim_embd=512, codebook_size=1024, n_head=8, n_layers=9, connect_list=['32', '64', '128', '256']).to("cuda")
        checkpoint = torch.load('./codeformer.ckpt')['params_ema']
        net.load_state_dict(checkpoint)
        net.eval()
        face_helper = FaceRestoreHelper(1, face_size=512, crop_ratio=(1, 1), det_model='retinaface_resnet50', save_ext='png', use_parse=True, device="cuda")

        self.net = net
        self.face_helper = face_helper

        return net, face_helper

    def send_model_to(self, device):
        self.net.to(device)
        self.face_helper.face_det.to(device)
        self.face_helper.face_parse.to(device)

    def restore(self, np_image, w=None):
        np_image = np_image[:, :, ::-1]

        original_resolution = np_image.shape[0:2]

        self.create_models()
        if self.net is None or self.face_helper is None:
            return np_image

        self.send_model_to("cuda")

        self.face_helper.clean_all()
        self.face_helper.read_image(np_image)
        self.face_helper.get_face_landmarks_5(only_center_face=False, resize=640, eye_dist_threshold=5)
        self.face_helper.align_warp_face()

        for idx, cropped_face in enumerate(self.face_helper.cropped_faces):
            cropped_face_t = img2tensor(cropped_face / 255., bgr2rgb=True, float32=True)
            normalize(cropped_face_t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
            cropped_face_t = cropped_face_t.unsqueeze(0).to("cuda")

            try:
                with torch.no_grad():
                    output = self.net(cropped_face_t, w=w if w is not None else 0.5, adain=True)[0]
                    restored_face = tensor2img(output, rgb2bgr=True, min_max=(-1, 1))
                del output
                torch.cuda.empty_cache()
            except Exception as error:
                print(f'\tFailed inference for CodeFormer: {error}', file=sys.stderr)
                restored_face = tensor2img(cropped_face_t, rgb2bgr=True, min_max=(-1, 1))

            restored_face = restored_face.astype('uint8')
            self.face_helper.add_restored_face(restored_face)

        self.face_helper.get_inverse_affine(None)

        restored_img = self.face_helper.paste_faces_to_input_image()
        restored_img = restored_img[:, :, ::-1]

        if original_resolution != restored_img.shape[0:2]:
            restored_img = cv2.resize(restored_img, (0, 0), fx=original_resolution[1]/restored_img.shape[1], fy=original_resolution[0]/restored_img.shape[0], interpolation=cv2.INTER_LINEAR)

        self.face_helper.clean_all()

        self.send_model_to("cpu")

        return restored_img


