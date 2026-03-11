import logging
import random
from typing import Callable, List, Tuple, Optional

import math
import torch
from datasets import Dataset
from torch import Tensor
from tqdm import tqdm

from pos_tagging.base import BaseUnsupervisedClassifier

logger = logging.getLogger()


class HMMClassifier(BaseUnsupervisedClassifier):
    def __init__(
        self,
        num_states: int,
        num_obs: int,
        sEM_alpha: Optional[float],
        sEM_batch_size: Optional[int],
    ) -> None:
        """
        For N hidden states and M observations,
            transition_prob: (N+1) * (N+1), with [0, :] as initial probabilities
            emission_prob: N * M

        Args:
            num_states: number of hidden states
            num_obs: number of observations
        """
        self.num_states = num_states
        self.num_obs = num_obs
        # Initialized to epsilon, so allowing unseen transition/emission to have p > 0
        self.epsilon = 1e-5
        # Initialize transition and emission probabilities
        self.reset()
        self.sEM_alpha = sEM_alpha
        self.sEM_batch_size = sEM_batch_size

    def reset(self) -> None:
        """Reset the HMM parameters to initial state."""
        # transition_prob[0, :] is the start probabilities
        # transition_prob[1:, 1:] is the state transition probabilities
        self.transition_prob = torch.full(
            [self.num_states + 1, self.num_states + 1], self.epsilon
        )
        self.transition_prob[:, 0] = 0.0  # No transitions to the start state
        # emission_prob[i, j] = P(observation j | state i)
        self.emission_prob = torch.full([self.num_states, self.num_obs], self.epsilon)
        self.log_scale = False
        self.cnt = 0  # Number of updates in sEM

    def train(
        self,
        inputs: Dataset,
        epochs: int = 5,
        method: str = "mle",
        continue_training=False,
    ) -> None:
        with torch.no_grad():
            if method == "mle":
                self.train_logmle(inputs=inputs)
            elif method == "EM":
                self.train_EM_log(
                    inputs=inputs, num_iter=epochs, continue_training=continue_training
                )
            elif method == "sEM":
                assert (
                    self.sEM_alpha is not None
                ), "sEM_alpha must be provided for sEM training"
                assert (
                    self.sEM_batch_size is not None
                ), "sEM_batch_size must be provided for sEM training"
                self.train_sEM(
                    inputs=inputs,
                    num_iter=epochs,
                    eta_fn=lambda k: (k + 2) ** (-self.sEM_alpha),
                    batch_size=self.sEM_batch_size,
                    continue_training=continue_training,
                )
            elif method == "hardEM":
                self.train_EM_hard_log(
                    inputs=inputs, num_iter=epochs, continue_training=continue_training
                )
            else:
                raise ValueError("Invalid training method name")

    def inference(self, input_ids) -> list:
        return self.viterbi_log(input_ids)

    @staticmethod
    def _normalize(mat: Tensor) -> Tensor:
        for i in range(mat.size(0)):
            if torch.sum(mat[i]) == 0:
                continue
            mat[i] = mat[i] / torch.sum(mat[i])
        return mat

    @staticmethod
    def _log_normalize(log_matrix: Tensor) -> Tensor:
        return log_matrix - torch.logsumexp(log_matrix, dim=-1, keepdim=True)

    @staticmethod
    def _normalize_log(mat: Tensor) -> Tensor:
        for i in range(mat.size(0)):
            if torch.sum(mat[i]) == 0:
                continue
            mat[i] = torch.log(mat[i]) - torch.log(torch.sum(mat[i]))
        return mat

    def _normalize_probabilities(self) -> None:
        """Normalize the transition and emission probabilities."""
        self.transition_prob[:, 1:] -= torch.logsumexp(
            self.transition_prob[:, 1:], dim=1, keepdim=True
        )
        self.emission_prob -= torch.logsumexp(self.emission_prob, dim=1, keepdim=True)

    def baum_welch(
        self, inputs: List[str], initial_value: float
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Baum-Welch algorithm to compute expected counts

        Args:
            inputs: list of observation sequences
            initial_value: initial value for accumulators
        Returns:
            start_state_counts: log probabilities of starting in each state
            transition_counts: log probabilities of transitioning between states
            emission_counts: log probabilities of emitting each observation from each state
        """
        # Initialise accumulators
        transition_counts = torch.full(
            (self.num_states, self.num_states), initial_value
        )
        start_state_counts = torch.full((self.num_states,), initial_value)
        emission_counts = torch.full((self.num_states, self.num_obs), initial_value)

        for input_ids in tqdm(inputs, desc="Running Baum-Welch"):
            n = len(input_ids)
            # Forward probabilities with shape: (seqlen, num_states)
            alpha = torch.zeros((n, self.num_states))
            # Backward probabilities with shape: (seqlen, num_states)
            beta = torch.zeros_like(alpha)

            # Calculate forward probabilities
            alpha[0] = self.transition_prob[0, 1:] + self.emission_prob[:, input_ids[0]]
            for t in range(1, n):
                alpha[t] = torch.logsumexp(
                    alpha[t - 1].unsqueeze(1)
                    + self.transition_prob[1:, 1:]
                    + self.emission_prob[:, input_ids[t]].unsqueeze(0),
                    dim=0,  # For each current state, sum over previous states
                )

            # Backward probabilities
            for t in range(n - 2, -1, -1):
                beta[t] = torch.logsumexp(
                    beta[t + 1].unsqueeze(0)
                    + self.transition_prob[1:, 1:]
                    + self.emission_prob[:, input_ids[t + 1]].unsqueeze(0),
                    dim=1,  # For each current state, sum over next states
                )

            # Given the whole sequence, the probability at each time at each state
            gamma = self._log_normalize(alpha + beta)  # Shape: (seqlen, num_states)

            # Given the whole sequence, the probability to transition at time t from state i to state j
            # Shape: (seqlen - 1, num_states, num_states)
            xi = (
                alpha[:-1, :].unsqueeze(2)
                + self.transition_prob[1:, 1:].unsqueeze(0)
                + self.emission_prob[:, input_ids[1:]].T.unsqueeze(1)
                + beta[1:, :].unsqueeze(1)
            )

            # Normalise over all possible transitions between states at this timestep
            xi = self._log_normalize(xi.view(-1, self.num_states**2)).view(
                -1, self.num_states, self.num_states
            )

            # Accumulate probabilities
            start_state_counts = torch.logaddexp(gamma[0], start_state_counts)
            transition_counts = torch.logaddexp(
                transition_counts, torch.logsumexp(xi, dim=0)
            )

            # Actual observations:
            # sum probabilities of states emitting the current observation at this time
            # Shape: (num_states, num_obs)
            for t in range(n):
                emission_counts[:, input_ids[t]] = torch.logaddexp(
                    emission_counts[:, input_ids[t]],
                    gamma[t, :],
                )

        return start_state_counts, transition_counts, emission_counts

    def train_mle(self, inputs: Dataset) -> None:
        """Supervised training by MLE

        Args:
            inputs: dataset of observation sequences with corresponding hidden states
        """
        logger.info("Running MLE")
        assert not self.log_scale
        for sentence in tqdm(inputs, "MLE training", len(inputs)):
            # Tokens should have been tokenized
            input_ids = sentence["input_ids"]
            # UPoS or XPoS should have been mapped to integers
            tags = sentence["tags"]

            # Update initial probabilities
            self.transition_prob[0, tags[0] + 1] += 1

            for i in range(len(input_ids)):
                # Update transition probabilities
                if i < len(input_ids) - 1:
                    self.transition_prob[tags[i] + 1, tags[i + 1] + 1] += 1

                # Update emission probabilities
                self.emission_prob[tags[i], input_ids[i]] += 1

        self.transition_prob = self._normalize(self.transition_prob)
        self.emission_prob = self._normalize(self.emission_prob)

    def train_logmle(self, inputs: Dataset) -> None:
        """Train with MLE algorithm using log likelihood to avoid underflow

        Args:
            inputs: dataset of observation sequences with corresponding hidden states
        """
        logger.info("Running log-scale MLE")
        assert not self.log_scale
        for sentence in tqdm(inputs, "Log-MLE training", len(inputs)):
            # Tokens should have been tokenized
            input_ids = sentence["input_ids"]
            # UPoS or XPoS should have been mapped to integers
            tags = sentence["tags"]

            # Update initial probabilities
            self.transition_prob[0, tags[0] + 1] += 1

            for i in range(len(input_ids)):
                # Update transition probabilities
                if i < len(input_ids) - 1:
                    self.transition_prob[tags[i] + 1, tags[i + 1] + 1] += 1

                # Update emission probabilities
                self.emission_prob[tags[i], input_ids[i]] += 1

        self.transition_prob = self._normalize_log(self.transition_prob)
        self.emission_prob = self._normalize_log(self.emission_prob)
        self.log_scale = True

    def train_EM_log(
        self,
        inputs: Dataset,
        num_iter: int = 5,
        continue_training: bool = False,
    ) -> None:
        """
        Train an HMM with the standard EM algorithm.
        The expected counts are calculated by Baum-Welch.

        Args:
            inputs: dataset of observation sequences
            num_iter: number of EM iterations
            continue_training: whether to continue training from existing parameters
        """
        logger.info("Running standard EM")
        self.log_scale = True
        # Accumulator initial value
        initial_value = math.log(self.epsilon)

        # Initialize transition and emission probabilities
        if not continue_training:
            self.transition_prob = torch.rand_like(self.transition_prob)
            self.emission_prob = torch.rand_like(self.emission_prob)
            self._normalize_probabilities()

        # Training loop
        input_ids_list = [sentence["input_ids"] for sentence in inputs]
        for epoch in tqdm(range(1, 1 + num_iter), desc="Standard EM Training"):
            logger.info(f"Epoch {epoch}:")
            start_state_counts, transition_counts, emission_counts = self.baum_welch(
                input_ids_list, initial_value
            )

            # Update transition and emission probabilities after an epoch
            self.transition_prob[0, 1:] = start_state_counts
            self.transition_prob[1:, 1:] = transition_counts
            self.emission_prob = emission_counts
            self._normalize_probabilities()

    def train_EM_hard_log(
        self,
        inputs: Dataset,
        num_iter: int = 10,
        continue_training: bool = False,
    ) -> None:
        """
        Train an HMM with the hard EM algorithm (also called Viterbi EM).

        Args:
            inputs: dataset of observation sequences
            num_iter: number of EM iterations
            continue_training: whether to continue training from existing parameters
        """
        logger.info("Running hard EM")
        self.log_scale = True
        # Accumulator initial value
        initial_value = 1.0

        # Initialize transition and emission probabilities
        if not continue_training:
            self.transition_prob = torch.rand_like(self.transition_prob)
            self.emission_prob = torch.rand_like(self.emission_prob)
            self._normalize_probabilities()

        # Training loop
        iters = 0
        for epoch in tqdm(range(1, 1 + num_iter), desc="Hard EM Training"):
            logger.info(f"Epoch {epoch}:")

            # Initialize the accumulators
            transition_accumulator = torch.full(
                (self.num_states, self.num_states), initial_value
            )
            emission_accumulator = torch.full(
                (self.num_states, self.num_obs), initial_value
            )
            start_state_accumulator = torch.full((self.num_states,), initial_value)

            for sentence in tqdm(inputs, desc=f"Epoch {epoch}"):
                iters += 1
                input_ids = sentence["input_ids"]

                # Run viterbi
                hidden_states = torch.tensor(self.viterbi_log(input_ids))

                # Accumulate transition and emission counts
                transition_accumulator.index_put_(
                    (hidden_states[:-1], hidden_states[1:]),
                    torch.tensor([1.0]),
                    accumulate=True,
                )
                emission_accumulator.index_put_(
                    (hidden_states, torch.tensor(input_ids)),
                    torch.tensor([1.0]),
                    accumulate=True,
                )
                start_state_accumulator[hidden_states[0]] += 1

            # Update transition and emission probabilities after an epoch
            self.transition_prob[0, 1:] = self._normalize_log(start_state_accumulator)
            self.transition_prob[1:, 1:] = self._normalize_log(transition_accumulator)
            self.emission_prob = self._normalize_log(emission_accumulator)

    def train_sEM(
        self,
        inputs: Dataset,
        num_iter: int = 30,
        eta_fn: Callable[[int], float] = lambda k: 0.8,
        batch_size: int = 256,
        continue_training: bool = False,
    ) -> None:
        """
        Train an HMM with a stepwise EM algorithm.

        Args:
            inputs: dataset of observation sequences
            num_iter: number of EM iterations
            eta_fn: the function used to calculate the stepsize eta
            batch_size: mini-batch size for each step
            continue_training: whether to continue training from existing parameters
        """
        logger.info("Running stepwise EM")
        self.log_scale = True
        # Accumulator initial value
        initial_value = math.log(self.epsilon)

        # Initialize transition and emission probabilities and accumulators
        if not continue_training:
            self.transition_prob = torch.rand_like(self.transition_prob)
            self.emission_prob = torch.rand_like(self.emission_prob)
            self._normalize_probabilities()
            self.eta_iters = 0

            self.start_counts = torch.full((self.num_states,), initial_value)
            self.transition_counts = torch.full(
                (self.num_states, self.num_states), initial_value
            )
            self.emission_counts = torch.full(
                (self.num_states, self.num_obs), initial_value
            )

        # Training loop
        input_ids_list = [sentence["input_ids"] for sentence in inputs]  # type: ignore
        for epoch in tqdm(range(1, 1 + num_iter), desc="Stepwise EM Training"):
            logger.info(f"Epoch {epoch}:")
            # Shuffle the dataset
            input_ids_list = random.sample(input_ids_list, len(input_ids_list))
            for i in tqdm(
                range(0, len(input_ids_list), batch_size), desc=f"Epoch {epoch}"
            ):
                batch = input_ids_list[i : i + batch_size]

                start_counts, transition_counts, emission_counts = self.baum_welch(
                    batch, initial_value
                )

                # Update transition and emission probabilities
                eta = torch.tensor(eta_fn(self.eta_iters))
                self.eta_iters += 1
                self.start_counts = torch.logaddexp(
                    torch.log(1 - eta) + self.start_counts,
                    torch.log(eta) + start_counts,
                )
                self.transition_counts = torch.logaddexp(
                    torch.log(1 - eta) + self.transition_counts,
                    torch.log(eta) + transition_counts,
                )
                self.emission_counts = torch.logaddexp(
                    torch.log(1 - eta) + self.emission_counts,
                    torch.log(eta) + emission_counts,
                )
                self.transition_prob[0, 1:] = self.start_counts
                self.transition_prob[1:, 1:] = self.transition_counts
                self.emission_prob = self.emission_counts
                self._normalize_probabilities()

    def viterbi_log(self, input_ids):
        """Run Viterbi algorithm with log-scale probabilities"""
        assert self.log_scale
        # Viterbi algorithm
        V = torch.zeros(self.num_states + 1, len(input_ids) + 1)
        path = {}
        V[1:, 1] = self.transition_prob[0, 1:] + self.emission_prob[:, input_ids[0]]

        for t in range(2, len(input_ids) + 1):
            score = (
                V[1:, t - 1].unsqueeze(1)
                + self.transition_prob[1:, 1:]
                + self.emission_prob[:, input_ids[t - 1]]
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
