import os

from torch import nn

from ffcv.fields.decoders import SimpleRGBImageDecoder
from ffcv.loader import Loader, OrderOption


class DummyModule(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, img):
        return img


def main():
    img_pipeline = [SimpleRGBImageDecoder(), DummyModule()]
    pipelines = {
        'image': img_pipeline,
        'image2': img_pipeline.copy(),
    }

    loader = Loader(
        path=os.environ['BETON_PATH'],
        batch_size=256,
        num_workers=12,
        order=OrderOption.QUASI_RANDOM,
        pipelines=pipelines,
        custom_field_mapper={'image2': 'image'},
    )

    data_iter = iter(loader)
    data = next(data_iter)
    print(f"{len(data)=}")
    print(f"{[d.shape for d in data]}")


if __name__ == '__main__':
    main()
