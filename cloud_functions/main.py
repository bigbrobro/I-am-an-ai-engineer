import os
import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F
from google.cloud import storage


# TODO : 항상 config/best_model_metadata.json에서 최신 모델 TAG 가져오도록하기
# TODO : hidden_size도 동적으로 갈 수 있도록
MODEL_TAG = 'FwLMbXZz-JCT13'


def handler(request):
    json_request = request.get_json()
    image_url = json_request.get('url')

    image = Image.open(image_url).convert("RGB")

    input_image = torch.tensor(image).float().reshape(-1, 1, 28, 28)

    os.mkdir('/tmp/save_model')
    client = storage.Client()
    bucket = client.get_bucket(bucket_or_name='geultto-functions')
    blobs = bucket.list_blobs(prefix=f'mnist_models/{MODEL_TAG}-model.pth')
    for blob in blobs:
        filename = os.path.basename(blob.name)
        blob.download_to_filename(f'/tmp/save_model/{MODEL_TAG}-model.pth')

    class Net(nn.Module):
        def __init__(self, hidden_size):
            super(Net, self).__init__()
            self.conv1 = nn.Conv2d(1, 20, 5, 1)
            self.conv2 = nn.Conv2d(20, 50, 5, 1)
            self.fc1 = nn.Linear(4 * 4 * 50, hidden_size)
            self.fc2 = nn.Linear(hidden_size, 10)

        def forward(self, x):
            x = F.relu(self.conv1(x))
            x = F.max_pool2d(x, 2, 2)
            x = F.relu(self.conv2(x))
            x = F.max_pool2d(x, 2, 2)
            x = x.view(-1, 4 * 4 * 50)
            x = F.relu(self.fc1(x))
            x = self.fc2(x)
            return F.log_softmax(x, dim=1)

    model = Net(1024)
    model.load_state_dict(torch.load(f'/tmp/save_model/{MODEL_TAG}-model.pth'))

    output = model(input_image)
    _, predicted = torch.max(output.data, 1)

    return predicted
