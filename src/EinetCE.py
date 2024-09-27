import torch
from torch import Tensor
import torch.optim as optim
import numpy as np 

from dataclasses import dataclass
from typing import List

from simple_einet.einet import Einet, EinetConfig
from simple_einet.layers.distributions.normal import Normal

@dataclass(frozen=True)
class EinetCardEstConfig:
    model_cfg: EinetConfig = None
    buffer_size: int = 10000
    train_epoch: int = 1000

class EinetCardinalityEstimator:
    def __init__(self, config: EinetCardEstConfig):
        self.model_cfg = config.model_cfg
        self.train_epoch = config.train_epoch
        self.buffer_size = config.buffer_size
        self.num_points = 0
        self.buffer: Tensor = None
        self.einets: List[Einet] = []

    def __train(self, data: Tensor, model: Einet):
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        for epoch in range(self.train_epoch):
            optimizer.zero_grad()
            log_prob = model(data)
            loss = -torch.mean(log_prob)
            loss.backward()
            optimizer.step()

    def __merge(self, model1: Einet, model2: Einet) -> Einet:
        num_samples = self.buffer_size // 2
        samples1 = model1.sample(num_samples)
        samples2 = model2.sample(num_samples)
        samples = torch.cat((samples1, samples2), 0).squeeze()

        model = Einet(self.model_cfg)
        self.__train(samples, model)
        return model
    
    def __batch_update(self, data: Tensor):
        assert len(data) == self.buffer_size, "Unexpected Error"
        model = Einet(self.model_cfg)
        self.__train(data, model)

        if len(self.einets) == 0:
            self.einets.append(model)
        else:
            for i in range(len(self.einets)):
                if self.einets[i] is None:
                    self.einets[i] = model
                    model = None
                    break
                else:
                    model = self.__merge(self.einets[i], model)
                    self.einets[i] = None
            if model is not None:
                self.einets.append(model)
        
    def update(self, data: Tensor):
        if data.dim() == 1:
            data = data.unsqueeze(0)
        
        assert data.dim() == 2, f"Wrong dimention of data with `data.dim() = {data.dim()}`"
        assert data.shape[1] == self.model_cfg.num_features, f"Wrong shape of data, expected shape (N, {self.model_cfg.num_features}), got (N, {data.shape[1]})"

        if self.buffer is None:
            self.buffer = data
        else:
            self.buffer = torch.vstack((self.buffer, data))

        while self.buffer is not None and len(self.buffer) >= self.buffer_size:
            train_data = None
            if len(self.buffer) == self.buffer_size:
                train_data = self.buffer
                self.buffer = None
            else:
                train_data = self.buffer[:self.buffer_size, :]
                self.buffer = self.buffer[self.buffer_size:, :]
            
            assert train_data is not None, "Unexpected Error"
            self.__batch_update(train_data)

    def query(self, interval: Tensor) -> float:
        assert interval.dim() == 2, f"Unexpected shape of query interval, expected (D, 2), got {interval.shape}"
        assert interval.shape[1] == 2, f"Unexpected shape of query interval, expected (D, 2), got {interval.shape}"

        weight = 1
        ans = 0
        for einet in self.einets:
            if einet is not None:
                ans += einet.integrate(interval).item() * weight
            weight *= 2
        ans *= self.buffer_size

        if self.buffer is not None:
            for point in self.buffer:
                flag = True
                dims = len(point)
                for i in range(dims):
                    x = point[i]
                    lb = interval[i][0]
                    ub = interval[i][1]
                    if x < lb or x > ub:
                        flag = False
                        break
                if flag:
                    ans += 1 
        
        return ans