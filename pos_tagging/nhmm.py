import logging
import wandb
from typing import Tuple

import torch
from torch import nn, Tensor
from datasets import Dataset
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

from pos_tagging.base import BaseUnsupervisedClassifier


logger = logging.getLogger()

wandb.init(
    mode="online",
    project="nhmm-GD", 
    config={               
        "architecture": "NHMM",
        "dataset": "PTB"
        
    }
)

class NeuralHMMClassifier(BaseUnsupervisedClassifier):
    def __init__(
        self,
        num_states: int,
        num_obs: int,
        hidden_dim: int = 128,
        device: str = "cpu",
        batch_size: int = 256,
        num_inner_loop_updates: int = 6,
    ):
        self.num_states = num_states
        self.num_obs = num_obs
        self.hidden_dim = hidden_dim
        self.device = device
        self.batch_size = batch_size
        self.num_inner_loop_updates = num_inner_loop_updates
        self.transition = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_states**2),
            nn.Unflatten(-1, (self.num_states, self.num_states)),
            nn.LogSoftmax(dim=-1),
        ).to(device)
        self.emission = nn.Sequential(
            nn.Embedding(num_states, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_obs),
            nn.LogSoftmax(dim=-1),
        ).to(device)
        self.start_probs = nn.Sequential(
            nn.Linear(1, num_states),
            nn.LogSoftmax(dim=-1),
        ).to(device)

        self.optimizer = torch.optim.Adam(
            params=list(self.transition.parameters())
            + list(self.emission.parameters())
            + list(self.start_probs.parameters()),
            lr=1e-3,
        )

    @staticmethod
    def _log_normalize(log_matrix):
        return log_matrix - torch.logsumexp(log_matrix, dim=-1, keepdim=True)

    def collate_batch(self, batch):
        input_list = [
            torch.tensor(item["input_ids"], device=self.device) for item in batch
        ]
        lengths = torch.tensor(
            [len(item["input_ids"]) for item in batch],
            dtype=torch.int32,
            device=self.device,
        )

        padded_inputs = pad_sequence(input_list, batch_first=True, padding_value=0)

        return padded_inputs, lengths

    def get_probabilities(self) -> Tuple[Tensor, Tensor, Tensor]:
        start_probs = self.start_probs(torch.tensor([1.0], device=self.device))
        transition_probs = self.transition(torch.tensor([1.0], device=self.device))
        emission_probs = self.emission(
            torch.arange(self.num_states, device=self.device)
        )

        return start_probs, transition_probs, emission_probs

    def train(
        self,
        inputs: Dataset,
        epochs: int = 5,
        method: str = "nhmm",
        continue_training=False,
    ) -> None:
        """
        Train a neural HMM.
        Reference: Unsupervised Neural Hidden Markov Models
        https://arxiv.org/pdf/1609.09007

        Args:
            inputs: dataset of observation sequences
            epochs: number of training epochs
            method: name of HMM method
            continue_training: whether to continue training from existing parameters
        """
        logger.info(f"Running neural HMM")
        dataloader = DataLoader(
            dataset=inputs,  # type: ignore
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self.collate_batch,
        )
        for epoch in tqdm(range(1, 1 + epochs), desc="Neural HMM Training"):
            logger.info(f"Epoch: {epoch}:")
            # total_loss = 0
            # l = 0
            for batched_samples, lengths in tqdm(dataloader):
                prev_log_likelihood = None
                for _ in range(self.num_inner_loop_updates):
                    # Calculate forward probabilities
                    start_probs, transition_probs, emission_probs = (
                        self.get_probabilities()
                    )
                    transition_probs = transition_probs.unsqueeze(0)
                    emission_probs = emission_probs.T.unsqueeze(1)
                    b, n = batched_samples.shape[:2]

                    # forward
                    alpha = torch.zeros(
                        (batched_samples.shape[0], n, self.num_states),
                        device=self.device,
                    )
                    alpha[:, 0] = (
                        start_probs.unsqueeze(0)
                        + emission_probs[batched_samples[:, 0], 0, :]
                    )
                    for t in range(1, n):
                        alpha[:, t] = torch.logsumexp(
                            alpha[:, t - 1].unsqueeze(-1)
                            + transition_probs
                            + emission_probs[batched_samples[:, t], ...],
                            dim=1,
                        )
                    # backward
                    beta = torch.zeros((b, n, self.num_states), device=self.device)
                    for t in range(n - 2, -1, -1):
                        beta[:, t] = torch.logsumexp(
                            transition_probs
                            + emission_probs[batched_samples[:, t + 1], ...]
                            + beta[:, t + 1].unsqueeze(1),
                            dim=2,
                        )
                    # log P(x)
                    log_Z = torch.logsumexp(alpha[:, n - 1], dim=1, keepdim=True) # [b, 1]

                    # E-step
                    # p(z_t | x)
                    log_gamma = alpha + beta - log_Z.unsqueeze(2)
                    gamma = torch.exp(log_gamma).detach() # [b, n, states]

                    # p(z_t, z_{t-1} | x)
                    log_xi = (
                        alpha[:, :-1].unsqueeze(-1)                                      # [b, n-1, prev_state, 1]
                        + transition_probs.unsqueeze(1)                                  # [1, 1, prev_state, curr_state]
                        + emission_probs[batched_samples[:, 1:]].squeeze(2).unsqueeze(2) # [b, n-1, 1, curr_state]
                        + beta[:, 1:].unsqueeze(2)                                       # [b, n-1, 1, curr_state]
                        - log_Z.unsqueeze(-1).unsqueeze(-1)                              # [b, 1, 1, 1]
                    )
                    xi = torch.exp(log_xi).detach() # [b, n-1, prev_state, curr_state]

                    # M-step 
                    # p(z_t | x) * ln p(x_t | z_t)
                    emission_log_probs = emission_probs[batched_samples].squeeze(2) # [b, n, states]
                    loss_emission = (gamma * emission_log_probs).sum()

                    # p(z_t, z_{t-1} | x) * ln p(z_t | z_{t-1})
                    loss_transition = (xi * transition_probs.unsqueeze(1)).sum()

                    # 初始概率梯度项
                    loss_start = (gamma[:, 0] * start_probs.unsqueeze(0)).sum()

                    loss = - (loss_start + loss_transition + loss_emission) / b

                    """
                    待添加EM版本, 且删除tag, 评测暂时为loss (类似于2020论文)
                    ------Data------
                    
                    ------GD------         
                    1. 忽略评测过程中的其他指标,在wandb记录中仅记录上方loss
                    ------EM------
                    2. 在这里(上方)添加E步的forward(本质上为从整句forward到每一个详细的都算出来)
                    3. 在下方修改M步, 获得与Backpropagation同等级的另一种方式
                    """
                    
                    # Backpropagation
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                    # [Original]:Calculate log likelihood of each sequence 
                    # origin_log_lik = torch.logsumexp(
                    #     alpha[torch.arange(b), lengths - 1, :], dim=1
                    # )
                    # origin_loss = -torch.mean(origin_log_lik)

                    if prev_log_likelihood is None:
                        prev_log_likelihood = -loss
                    else:
                        # If the log probability has converged, go to next batch
                        if (prev_log_likelihood + loss) / prev_log_likelihood < 1e-4:
                            break
                        else:
                            prev_log_likelihood = -loss
                    wandb.log({
                        "loss": prev_log_likelihood})
                    # total_loss += prev_log_likelihood
                    # l += 1
                    # # 计算完 loss 后
                    # del alpha, beta, log_xi, xi, gamma 
                    # torch.cuda.empty_cache()

            # total_loss = total_loss / l
            wandb.log({
                "loss_per_epoch": prev_log_likelihood,
                "epoch": epoch
            })       

    def get_probabilities_for_eval(self):
        (
            self.start_probs_for_eval,
            self.transition_probs_for_eval,
            self.emission_probs_for_eval,
        ) = self.get_probabilities()
        self.start_probs_for_eval = self.start_probs_for_eval.cpu()
        self.transition_probs_for_eval = self.transition_probs_for_eval.cpu()
        self.emission_probs_for_eval = self.emission_probs_for_eval.cpu()

    def inference(self, input_ids) -> list:
        return self.viterbi_log(
            input_ids,
            self.start_probs_for_eval,
            self.transition_probs_for_eval,
            self.emission_probs_for_eval,
        )

    def viterbi_log(self, input_ids, start_probs, transition_probs, emission_probs):
        """Run Viterbi algorithm with log-scale probabilities"""
        # Viterbi algorithm
        V = torch.zeros(self.num_states + 1, len(input_ids) + 1)
        path = {}
        V[1:, 1] = start_probs + emission_probs[:, input_ids[0]]

        for t in range(2, len(input_ids) + 1):
            score = (
                V[1:, t - 1].unsqueeze(1)
                + transition_probs
                + emission_probs[:, input_ids[t - 1]]
            )
            max_values, argmax = torch.max(score, dim=0)
            V[1:, t] = max_values
            for s in range(self.num_states):
                path[s, t] = argmax[s]

        optimal_path = []
        last_state = torch.argmax(V[1:, -1])
        optimal_path.append(last_state.item())

        for t in range(len(input_ids), 1, -1):
            last_state = path[last_state.item(), t]
            optimal_path.insert(0, last_state.item())

        return optimal_path
