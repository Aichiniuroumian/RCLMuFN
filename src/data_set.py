from torch.utils.data import Dataset
import logging
import os
from PIL import Image
import json
import transforms as T
from misc import nested_tensor_from_tensor_list
from torchvision import transforms



logger = logging.getLogger(__name__)

WORKING_PATH="./MMSD2.0dataset/data"
class MyDataset(Dataset):
    def __init__(self, mode, text_name, limit=None):
        self.text_name = text_name
        self.data = self.load_data(mode, limit)
        self.image_ids=list(self.data.keys())
        for id in self.data.keys():
            self.data[id]["image_path"] = os.path.join(WORKING_PATH,"dataset_image",str(id)+".jpg")
    
    def load_data(self, mode, limit):
        cnt = 0
        data_set=dict()
        if mode in ["train"]:
            f1= open(os.path.join(WORKING_PATH, self.text_name ,mode+".json"),'r',encoding='utf-8')
            datas = json.load(f1)
            for data in datas:
                if limit != None and cnt >= limit:
                    break

                image = data['image_id']
                sentence = data['text']
                label = data['label']
 
                if os.path.isfile(os.path.join(WORKING_PATH,"dataset_image",str(image)+".jpg")):
                    data_set[int(image)]={"text":sentence, 'label': label}
                    cnt += 1
                    
        
        if mode in ["test","valid"]:
            f1= open(os.path.join(WORKING_PATH, self.text_name ,mode+".json"),'r',encoding='utf-8')
            datas = json.load(f1)
            for data in datas:
                image = data['image_id']
                sentence = data['text']
                label = data['label']

                if os.path.isfile(os.path.join(WORKING_PATH,"dataset_image",str(image)+".jpg")):
                    data_set[int(image)]={"text":sentence, 'label': label}
                    cnt += 1
        return data_set


    def image_loader(self,id):
        return Image.open(self.data[id]["image_path"])
    def text_loader(self,id):
        return self.data[id]["text"]


    def __getitem__(self, index):
        id=self.image_ids[index]
        text = self.text_loader(id)
        image_feature = self.image_loader(id)
        label = self.data[id]["label"]
        return text,image_feature, label, id

    def __len__(self):
        return len(self.image_ids)

    @staticmethod
    def collate_func(batch_data):
        batch_size = len(batch_data)
 
        if batch_size == 0:
            return {}

        text_list = []
        image_list = []
        label_list = []
        id_list = []
        batches = []


        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        for instance in batch_data:
            a = instance[0]
            b = instance[1]
            c = instance[2]
            d = instance[3]
            text_list.append(instance[0])
            image_list.append(instance[1])
            label_list.append(instance[2])
            id_list.append(instance[3])
            samples = transform(instance[1])
            batches.append(samples)
        batch = tuple(batches)
        samples = nested_tensor_from_tensor_list(batch)
        return text_list, image_list, label_list, id_list, samples

