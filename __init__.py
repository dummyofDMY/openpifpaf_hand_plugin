import openpifpaf

from . import hand_kp


def register():
    openpifpaf.DATAMODULES['hand'] = hand_kp.HandKp
    openpifpaf.CHECKPOINT_URLS['shufflenetv2k16-apollo-24'] = \
        "http://github.com/DuncanZauss/openpifpaf_assets/releases/" \
        "download/v0.1.0/shufflenetv2k16-201113-135121-apollo.pkl.epoch290"
    openpifpaf.CHECKPOINT_URLS['shufflenetv2k16-apollo-66'] = \
        "http://github.com/DuncanZauss/openpifpaf_assets/releases/" \
        "download/v0.1.0/sk16_apollo_66kp.pkl"
    openpifpaf.CHECKPOINT_URLS['shufflenetv2k30-apollo-66'] = \
        "http://github.com/DuncanZauss/openpifpaf_assets/releases/" \
        "download/v0.1.0/sk30_apollo_66kp.pkl"
