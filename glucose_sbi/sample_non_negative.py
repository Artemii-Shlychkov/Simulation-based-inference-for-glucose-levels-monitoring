import torch
from sbi.inference import DirectPosterior
from sbi.neural_nets.estimators.shape_handling import reshape_to_batch_event
from sbi.samplers.rejection.rejection import accept_reject_sample
from sbi.utils.sbiutils import within_support


def all_nonnegative(theta: torch.Tensor) -> torch.Tensor:
    """Return a boolean tensor indicating whether each sample's parameters are all nonnegative.

    Parameters
    ----------
    theta : torch.Tensor
        The tensor of samples to check.

    Returns
    -------
    torch.Tensor
        A boolean tensor indicating whether each sample's parameters are all nonnegative.

    """
    # theta shape is (N, dim). We want to check each sample row:
    return (theta >= 0).all(dim=-1)


class DirectPosteriorNonnegative(DirectPosterior):
    """Subclass of `DirectPosterior` that samples only nonnegative parameters."""

    def sample(
        self,
        sample_shape: torch.Size | None = None,
        x: torch.Tensor | None = None,
        max_sampling_batch_size: int = 10_000,
        sample_with: str | None = None,
        *,
        show_progress_bars: bool = False,
    ) -> torch.Tensor:
        if sample_shape is None:
            sample_shape = torch.Size()
        num_samples = sample_shape.numel()
        x = self._x_else_default_x(x)
        x = reshape_to_batch_event(
            x, event_shape=self.posterior_estimator.condition_shape
        )

        # Reuse the standard checks from the parent class:
        if x is not None and x.shape[0] > 1:
            msg = (
                ".sample() supports only `batchsize == 1`. If you intend "
                "to sample multiple observations, use `.sample_batched()`. "
            )
            raise ValueError(msg)
        if sample_with is not None:
            msg = (
                f"You set `sample_with={sample_with}`. As of sbi v0.18.0, setting "
                f"`sample_with` is no longer supported."
            )
            raise ValueError(msg)

        # Our custom accept/reject function that:
        # 1) is within the original prior support
        # 2) is all nonnegative
        def accept_reject_fn(theta_batch: torch.Tensor) -> torch.Tensor:
            # 1) within prior support
            within = within_support(self.prior, theta_batch)
            # 2) each sample's parameters >= 0
            nonneg = all_nonnegative(theta_batch)
            return within & nonneg

        samples, _ = accept_reject_sample(
            proposal=self.posterior_estimator,
            accept_reject_fn=accept_reject_fn,
            num_samples=num_samples,
            show_progress_bars=show_progress_bars,
            max_sampling_batch_size=max_sampling_batch_size,
            proposal_sampling_kwargs={"condition": x},
            alternative_method="build_posterior(..., sample_with='mcmc')",
        )

        # samples shape is (num_samples, batch_dim=1, param_dim).
        # By default, we remove the batch dimension:
        return samples[:, 0]


def sample_non_negative(
    posterior: DirectPosterior,
    num_samples: int,
    true_observation: torch.Tensor | None = None,
) -> torch.Tensor:
    """Sample from a posterior, ensuring all samples are nonnegative.

    Parameters
    ----------
    posterior : DirectPosterior
        The posterior to sample from.
    num_samples : int
        The number of samples to draw.
    true_observation : torch.Tensor
        The true observation.

    Returns
    -------
    torch.Tensor
        The samples drawn from the posterior.

    """
    # Create a custom DirectPosterior subclass that samples only nonnegative parameters:
    post_nonneg = DirectPosteriorNonnegative(
        posterior_estimator=posterior.posterior_estimator,
        prior=posterior.prior,
        max_sampling_batch_size=posterior.max_sampling_batch_size,
        device=posterior._device,  # noqa: SLF001
    )
    sample_size = torch.Size([num_samples])

    # Sample from the custom DirectPosterior subclass:
    return post_nonneg.sample((sample_size), x=true_observation)
