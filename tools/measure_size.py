import torch
from model.LungAttn import LungAttn
from model.LungAttn import get_mnist_loaders
from copy import deepcopy
import numpy as np
import time
class ModelCharacteristic:
    def __init__(self, model, dataloader):
        self.model = model
        self.dataloader = dataloader

    def calculate_size_mb(self):
        # source: https://discuss.pytorch.org/t/finding-model-size/130275
        param_size, buffer_size = 0, 0
        for param in self.model.parameters():
            param_size += param.nelement() * param.element_size()
        for buffer in self.model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()

        size_mb = (param_size + buffer_size) / 1024**2
        self.size_mb = size_mb
        print('model size: {:.3f}MB'.format(size_mb))

    def parameters_num(self):
        self.params_num = sum(param.numel() for param in self.model.parameters())
        print("model parameters num: {}".format(self.params_num))
    def measure_inference_time_gpu(self):
        # source: https://deci.ai/blog/measure-inference-time-deep-neural-networks/
        model = deepcopy(self.model)
        input_size = self.get_input_size()
        device = torch.device("cuda")
        model.to(device)
        model.eval()
        dummy_input = torch.randn(1, input_size[0], input_size[1], input_size[2], dtype=torch.float).to(device)
        data = next(iter(self.dataloader))[0].to(device)
        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        steps = 100
        times = np.zeros(steps)

        for _ in range(10):
            _ = model(dummy_input)
        with torch.no_grad():
            for step in range(steps):
                sample = data[step].unsqueeze(0)
                starter.record()
                _ = model(sample) #measure dummy or normal data?
                ender.record()
                torch.cuda.synchronize()
                curr_time = starter.elapsed_time(ender)
                times[step] = curr_time

        self.mean_infer_time_gpu = np.mean(times)
        self.std_infer_time_gpu = np.std(times)
        print("Mean inference time on GPU is {} ms with std {} ms".format(self.mean_infer_time_gpu, self.std_infer_time_gpu))

    def measure_inference_time_cpu(self):
        model = deepcopy(self.model)
        input_size = self.get_input_size()
        device = torch.device("cpu")
        model.to(device)
        model.eval()
        dummy_input = torch.randn(1, input_size[0], input_size[1], input_size[2], dtype=torch.float)
        data = next(iter(self.dataloader))[0]
        steps = 100
        times = np.zeros(steps)
        for _ in range(10):
            _ = model(dummy_input)
        with torch.no_grad():
            for step in range(steps):
                sample = data[step].unsqueeze(0)
                t1 = time.time() * 1000
                _ = model(sample)  # measure dummy or normal data?
                t2 = time.time() * 1000
                curr_time = t2 - t1
                times[step] = curr_time
        self.mean_infer_time_cpu = np.mean(times)
        self.std_infer_time_cpu = np.std(times)
        print("Mean inference time on CPU is {} ms with std {} ms".format(self.mean_infer_time_cpu, self.std_infer_time_cpu))

    def get_input_size(self):
        sample, labels = next(iter(self.dataloader))
        B, C, H, W = sample.size()
        self.input_size = (C, H, W)
        self.batch_size = B
        return self.input_size



if __name__ == "__main__":
    import os
    print(os.getcwd())
    state_dict_path = "./log/details/mix_224bs256lr0.1dp0.20.40.3dk20dv4nh2ep100wd0_12_08/saved_model_params"
    model = LungAttn()
    train_loader, train_eval_loader, test_loader = get_mnist_loaders(batch_size=128, test_batch_size=100, workers=12)
    model.load_state_dict(torch.load(state_dict_path))
    model_characteristics = ModelCharacteristic(model, test_loader)
    model_characteristics.calculate_size_mb()
    model_characteristics.parameters_num()
    model_characteristics.measure_inference_time_gpu()
    model_characteristics.measure_inference_time_cpu()