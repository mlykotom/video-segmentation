__all__ = ['BaseModel', 'SegNet', 'MobileUNet']

import sys

if sys.version_info >= (3, 0):
    from models.base_model import BaseModel
    from models.mobile_unet import MobileUNet
    from models.segnet import SegNet
else:
    from base_model import BaseModel
    from mobile_unet import MobileUNet
    from segnet import SegNet
