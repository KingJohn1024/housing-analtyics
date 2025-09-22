import os
import json
import pandas as pd
import requests
import torch
import torchvision.models as models
import torchvision.transforms as T
from io import BytesIO
from PIL import Image
from housing_analytics.base import Base
from housing_analytics.data_collection import DataCollection


class DataProcessing(Base):
    def __init__(self, verbose=False):
        super().__init__(verbose=verbose)
        self.dc = DataCollection(verbose=verbose)

    def load_property_profile(self, zpid) -> dict:
        fpath = f'{self.root_path}/data/property_profiles/{zpid}.json'
        if not os.path.exists(fpath):
            self.dc.get_property_profile(zpid=zpid)

        try:
            with open(fpath, 'r') as f:
                pp = json.load(f)
        except FileNotFoundError as e:
            self.logger.error(f'Failed to load file: {fpath} | {e}')
        return pp

    def get_listing_pictures(self, zpid: int) -> list[str]:
        pp = self.load_property_profile(zpid=zpid)
        pics = pp.get('originalPhotos')

        images_urls = []
        for pic in pics:
            root = pic.get('mixedSources').get('jpeg')
            image_url = root[0].get('url')
            images_urls.append(image_url)
        return images_urls

    def image2vec(self, image_url) -> pd.DataFrame:
        """
        Converts an image to a vector using the pretrained Resnet model
        """

        # Load the images from the URLs
        response = requests.get(image_url)
        image_data = BytesIO(response.content)
        img = Image.open(image_data).convert('RGB')

        # Load pretrained ResNet
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        resnet.fc = torch.nn.Identity()  # remove final classification layer
        resnet.eval()

        # Image preprocessing
        transform = T.Compose(
            [
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        x = transform(img).unsqueeze(0)
        with torch.no_grad():
            emb = resnet(x).squeeze().numpy()

        # 512 features per image
        return pd.DataFrame(emb)

    def generate_listing_image_features(self, zpid: int) -> pd.DataFrame:
        """
        Generate image features for a listing images for a given zpid

        :param zpid: Zillow property identifier
        :return: Dataframe indexed by zpid x 512 with n-number of image feature columns
        """
        image_urls = self.get_listing_pictures(zpid=zpid)
        image_data = []
        for image_url in image_urls:
            image_data.append(self.image2vec(image_url=image_url))
        df = pd.concat(image_data, axis=1)
        df.columns = [f'Image{i}' for i in range(0, df.shape[1])]
        df.index = [zpid] * df.shape[0]
        return df

    def generate_all_listing_image_features(self):
        zpids = self.get_all_zpids()
        for zpid in zpids:
            fpath = f'{self.root_path}/data/image_data/{zpid}.pq'
            if not os.path.exists(fpath):
                df = self.generate_listing_image_features(zpid=zpid)
                df.to_parquet(fpath)
