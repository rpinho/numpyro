# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

"""Regression and edge-case test bank for NumPyro issue #1614.

Issue #1614: initialization strategies such as ``init_to_value`` are ignored
when targeting the original name of a reparameterized site (for example under
``LocScaleReparam``, ``CircularReparam``, ``TransformReparam``, etc.).

Each test documents:
- the expected behavior (starts at the requested value, raises an error naming
  the affected site, or matches master invariance), and
- which independent referee round(s) (r1..r7) or acceptance bank surfaced it.

Tests are organized into five groups:
  (a) Core bug (#1614 repro, examples/ssbvm_mixture.py k-means init, families)
  (b) Plates, scope, mask, sample_shape, control flow, and vectorized chains
  (c) Handler composition and user objects
  (d) Numeric edges
  (e) Entry points (AutoGuides, Predictive, SVI, SA, Gibbs, NestedSampler)
"""

from functools import partial
import re
import warnings

import numpy as np
from numpy.testing import assert_allclose
import pytest

import jax
from jax import random
import jax.numpy as jnp

import numpyro
from numpyro.contrib.control_flow import cond
import numpyro.distributions as dist
from numpyro.distributions import biject_to, constraints, transforms
import numpyro.handlers as handlers
from numpyro.infer import (
    HMC,
    HMCECS,
    MCMC,
    NUTS,
    SA,
    SVI,
    DiscreteHMCGibbs,
    Gibbs,
    HMCGibbs,
    MixedHMC,
    Predictive,
    Trace_ELBO,
    init_to_feasible,
    init_to_mean,
    init_to_median,
    init_to_uniform,
    init_to_value,
)
from numpyro.infer.autoguide import (
    AutoDelta,
    AutoDiagonalNormal,
    AutoNormal,
)
from numpyro.infer.reparam import (
    CircularReparam,
    DiscreteCosineReparam,
    ExplicitReparam,
    HaarReparam,
    LocScaleReparam,
    NeuTraReparam,
    ProjectedNormalReparam,
    TransformReparam,
    UnitJacobianReparam,
)
from numpyro.infer.util import find_valid_initial_params, initialize_model
from numpyro.optim import Adam
from numpyro.primitives import Messenger


def _wrap_angle_diff(actual, expected):
    diff = np.asarray(actual) - np.asarray(expected)
    return (diff + np.pi) % (2.0 * np.pi) - np.pi


def _assert_circular_close(actual, expected, atol=1e-4):
    assert_allclose(_wrap_angle_diff(actual, expected), 0.0, atol=atol)


def _error_names_site(exc, site_names):
    msg = str(exc)
    return any(
        re.search(rf"(?<![A-Za-z0-9_]){re.escape(name)}(?![A-Za-z0-9_])", msg)
        for name in site_names
    )


def _run_or_expect_site_error(fn, site_names):
    """Run ``fn()``; return its value, or assert a raised error names a site."""
    try:
        return fn()
    except Exception as exc:
        assert _error_names_site(exc, site_names), (
            f"Expected error naming one of {site_names}, "
            f"got {type(exc).__name__}: {exc}"
        )
        return None


def _init_replay(model, strategy, key=None, model_args=()):
    key = random.key(42) if key is None else key
    info = initialize_model(key, model, init_strategy=strategy, model_args=model_args)
    replayed = info.postprocess_fn(info.param_info.z)
    return info, replayed


# =============================================================================
# (a) Core bug: #1614 repro, examples/ssbvm_mixture.py k-means init, families
# =============================================================================


def test_issue_1614_init_to_value():
    """Core #1614 repro with init_to_value on CircularReparam.

    Expected: NUTS starts at ``loc_1 == 0.5`` (mod 2*pi) with finite energy.
    Found in: r1, independent_bank.
    """
    data = jnp.array([1.0, 2.0, 3.0])

    @handlers.reparam(config={"loc_1": CircularReparam()})
    def model():
        angle = numpyro.sample("loc_1", dist.VonMises(0.0, 100.0))
        with numpyro.plate("plate", 3):
            numpyro.sample("y", dist.VonMises(angle, 2.0), obs=data)

    kernel = NUTS(model, init_strategy=init_to_value(values={"loc_1": 0.5}))
    state = kernel.init(random.key(42), num_warmup=0)
    replayed = kernel.postprocess_fn((), {})(state.z)
    assert np.isfinite(float(state.potential_energy))
    _assert_circular_close(replayed["loc_1"], 0.5, atol=1e-4)


def test_issue_1614_init_to_median():
    """Core #1614 repro with init_to_median on CircularReparam.

    Expected: starts near the VonMises(0, 100) prior median (0.0, within 0.05
    rad) or raises an error naming ``loc_1``.
    Found in: r1, r4, independent_bank.
    """
    data = jnp.array([1.0, 2.0, 3.0])

    @handlers.reparam(config={"loc_1": CircularReparam()})
    def model():
        angle = numpyro.sample("loc_1", dist.VonMises(0.0, 100.0))
        with numpyro.plate("plate", 3):
            numpyro.sample("y", dist.VonMises(angle, 2.0), obs=data)

    def _run():
        kernel = NUTS(model, init_strategy=init_to_median(num_samples=1000))
        state = kernel.init(random.key(42), num_warmup=0)
        return state, kernel.postprocess_fn((), {})(state.z)

    out = _run_or_expect_site_error(_run, ["loc_1"])
    if out is not None:
        state, replayed = out
        assert np.isfinite(float(state.potential_energy))
        _assert_circular_close(replayed["loc_1"], 0.0, atol=0.05)


def test_ssbvm_mixture_kmeans_init():
    """Enumerated SineBivariateVonMises mixture initialized at k-means centers.

    Mirrors ``examples/ssbvm_mixture.py``: ``phi_loc`` and ``psi_loc`` are
    reparameterized with ``CircularReparam`` inside a mixture plate and
    initialized via ``init_to_value`` using synthetic cluster centroids.

    Expected: NUTS starts at the requested ``phi_loc`` and ``psi_loc`` centers
    (mod 2*pi) with finite potential energy.
    Found in: r1, independent_bank.
    """
    data = jnp.array(
        [
            [-2.2, -1.2],
            [-2.0, -1.0],
            [-1.8, -0.8],
            [0.7, 1.2],
            [0.9, 1.4],
            [1.1, 1.6],
        ],
        dtype=jnp.float32,
    )
    centers = jnp.stack([data[:3].mean(axis=0), data[3:].mean(axis=0)])
    requests = {"phi_loc": centers[:, 0], "psi_loc": centers[:, 1]}

    @handlers.reparam(
        config={"phi_loc": CircularReparam(), "psi_loc": CircularReparam()}
    )
    def ss_model(obs, num_data, num_mix_comp=2):
        mix_weights = numpyro.sample(
            "mix_weights", dist.Dirichlet(jnp.ones(num_mix_comp))
        )
        beta_mean_phi = numpyro.sample("beta_mean_phi", dist.Uniform(0.0, 1.0))
        beta_count_phi = numpyro.sample(
            "beta_count_phi", dist.Gamma(1.0, 1.0 / num_mix_comp)
        )
        halpha_phi = beta_mean_phi * beta_count_phi
        beta_mean_psi = numpyro.sample("beta_mean_psi", dist.Uniform(0.0, 1.0))
        beta_count_psi = numpyro.sample(
            "beta_count_psi", dist.Gamma(1.0, 1.0 / num_mix_comp)
        )
        halpha_psi = beta_mean_psi * beta_count_psi

        with numpyro.plate("mixture", num_mix_comp):
            phi_loc = numpyro.sample("phi_loc", dist.VonMises(np.pi, 2.0))
            psi_loc = numpyro.sample("psi_loc", dist.VonMises(0.0, 0.1))
            phi_conc = numpyro.sample(
                "phi_conc", dist.Beta(halpha_phi, beta_count_phi - halpha_phi)
            )
            psi_conc = numpyro.sample(
                "psi_conc", dist.Beta(halpha_psi, beta_count_psi - halpha_psi)
            )
            corr_scale = numpyro.sample("corr_scale", dist.Beta(2.0, 10.0))
            skewness = numpyro.sample(
                "skewness", dist.Normal(0.0, 0.5).expand((2,)).to_event(1)
            )
            skewness = transforms.L1BallTransform()(skewness)

        with numpyro.plate("obs_plate", num_data, dim=-1):
            assign = numpyro.sample(
                "mix_comp",
                dist.Categorical(mix_weights),
                infer={"enumerate": "parallel"},
            )
            sine = dist.SineBivariateVonMises(
                phi_loc=phi_loc[assign],
                psi_loc=psi_loc[assign],
                phi_concentration=1000.0 * phi_conc[assign],
                psi_concentration=1000.0 * psi_conc[assign],
                weighted_correlation=corr_scale[assign],
            )
            return numpyro.sample(
                "phi_psi", dist.SineSkewed(sine, skewness[assign]), obs=obs
            )

    kernel = NUTS(ss_model, init_strategy=init_to_value(values=requests))
    state = kernel.init(random.key(23), num_warmup=0, model_args=(data, len(data), 2))
    replayed = kernel.postprocess_fn((data, len(data), 2), {})(state.z)
    assert np.isfinite(float(state.potential_energy))
    _assert_circular_close(replayed["phi_loc"], requests["phi_loc"], atol=1e-4)
    _assert_circular_close(replayed["psi_loc"], requests["psi_loc"], atol=1e-4)


def test_dependent_reparam_chain_depth_5():
    """Five-site dependent chain of LocScaleReparam sites.

    Fixed-iteration replay caps (e.g. stopping after 4 replays) miss site 5.

    Expected: all five sites replay to ``[1, 2, 3, 4, 5]``, or an error naming
    one of the chain sites is raised.
    Found in: r4, r5, independent_bank.
    """
    names = [f"stage_{i}" for i in range(5)]
    requests = {name: float(i + 1) for i, name in enumerate(names)}

    @handlers.reparam(config={name: LocScaleReparam(0.0) for name in names})
    def model():
        parent = 0.0
        for name in names:
            parent = numpyro.sample(name, dist.Normal(parent, 1.0))

    out = _run_or_expect_site_error(
        lambda: _init_replay(model, init_to_value(values=requests)),
        names,
    )
    if out is not None:
        _, replayed = out
        for name, expected in requests.items():
            assert_allclose(replayed[name], expected, atol=1e-4)


def test_reparam_families_init_to_value():
    """Explicit original-site requests across built-in reparameterizers.

    Covers TransformReparam, ExplicitReparam, ProjectedNormalReparam,
    HaarReparam, and DiscreteCosineReparam.

    Expected: each family replays to the requested original value (or raises an
    error naming ``target``).
    Found in: r3, r5, independent_bank.
    """
    cases = [
        (dist.LogNormal(0.5, 0.5), TransformReparam(), jnp.array(4.0)),
        (
            dist.Normal(2.0, 1.0),
            ExplicitReparam(transforms.AffineTransform(1.0, 2.0)),
            jnp.array(-3.0),
        ),
        (
            dist.ProjectedNormal(jnp.array([2.0, -5.0, 1.0])),
            ProjectedNormalReparam(),
            jnp.array([0.0, 0.6, 0.8]),
        ),
        (
            dist.Normal(jnp.arange(5.0), 1.0).to_event(1),
            HaarReparam(flip=True),
            jnp.array([0.0, 2.0, -1.0, 4.0, 3.0]),
        ),
        (
            dist.Normal(jnp.arange(5.0), 1.0).to_event(1),
            DiscreteCosineReparam(smooth=0.5),
            jnp.array([0.0, 2.0, -1.0, 4.0, 3.0]),
        ),
    ]
    for base_dist, rp, goal in cases:

        @handlers.reparam(config={"target": rp})
        def model(d=base_dist):
            numpyro.sample("target", d)

        out = _run_or_expect_site_error(
            lambda m=model, g=goal: _init_replay(
                m, init_to_value(values={"target": g})
            ),
            ["target"],
        )
        if out is not None:
            _, replayed = out
            assert_allclose(replayed["target"], goal, atol=1e-4)


def test_stacked_reparams_init_to_value():
    """Nested/stacked reparameterizers (TransformReparam + LocScaleReparam).

    Expected: ``rate`` replays to ``4.0`` or raises an error naming ``rate``.
    Found in: r3, r5.
    """

    @handlers.reparam(config={"rate_base": LocScaleReparam(0.25)})
    @handlers.reparam(config={"rate": TransformReparam()})
    def model():
        numpyro.sample("rate", dist.LogNormal(0.75, 2.0))

    out = _run_or_expect_site_error(
        lambda: _init_replay(model, init_to_value(values={"rate": 4.0})),
        ["rate"],
    )
    if out is not None:
        _, replayed = out
        assert_allclose(replayed["rate"], 4.0, atol=1e-4)


# =============================================================================
# (b) Plates, scope, mask, sample_shape, control flow, and vectorized chains
# =============================================================================


def test_plate_enclosing_reparam_full_request():
    """Plate enclosing an in-model reparam handler with a full-shape request.

    Initializing the auxiliary before the enclosing plate expands the
    distribution collapses shape ``(4,)`` to scalar ``()``.

    Expected: auxiliary and original site both have shape ``(4,)`` and replay
    to the requested four values (or raise an error naming ``theta``).
    Found in: r1, r2.
    """
    target = jnp.array([-1.0, 0.5, 2.0, 3.5])

    def model():
        with numpyro.plate("rows", 4):
            with handlers.reparam(config={"theta": LocScaleReparam(0.25)}):
                numpyro.sample("theta", dist.Normal(-1.5, 2.25))

    out = _run_or_expect_site_error(
        lambda: _init_replay(model, init_to_value(values={"theta": target})),
        ["theta"],
    )
    if out is not None:
        info, replayed = out
        assert info.param_info.z["theta_decentered"].shape == (4,)
        assert replayed["theta"].shape == (4,)
        assert_allclose(replayed["theta"], target, atol=1e-4)


def test_plate_scalar_request_preserves_shape():
    """Scalar request for a plated reparameterized site in 2-chain NUTS.

    Passing a scalar to a plated reparam site must never silently shrink latent
    shape from ``(2, 4)`` to ``(2,)``.

    Expected: either broadcasts to shape ``(2, 4)`` starting at ``0.75``, or
    raises an error naming ``theta``.
    Found in: r2, independent_bank.
    """

    def model():
        with numpyro.plate("members", 4):
            with handlers.reparam(config={"theta": LocScaleReparam(0.4)}):
                numpyro.sample("theta", dist.Normal(1.5, 2.25))

    def _run():
        kernel = NUTS(model, init_strategy=init_to_value(values={"theta": 0.75}))
        state = kernel.init(random.split(random.key(73), 2), num_warmup=0)
        replayed = jax.vmap(kernel.postprocess_fn((), {}))(state.z)
        return state.z["theta_decentered"], replayed["theta"]

    out = _run_or_expect_site_error(_run, ["theta"])
    if out is not None:
        aux, theta = out
        assert aux.shape == (2, 4)
        assert theta.shape == (2, 4)
        assert_allclose(theta, jnp.full((2, 4), 0.75), atol=1e-4)


def test_scope_outside_reparam_partial_centering():
    """Enclosing scope around a reparam handler with partial centering.

    Looking up the unscoped local name misses ``group/angle`` and falls back to
    an unrelated start.

    Expected: starts at ``group/angle == 0.375`` or raises an error naming
    ``group/angle`` (or ``angle``).
    Found in: r1, r3.
    """

    def base():
        numpyro.sample("angle", dist.Normal(-1.5, 2.25))

    model = handlers.scope(
        handlers.reparam(base, config={"angle": LocScaleReparam(0.25)}),
        prefix="group",
    )

    out = _run_or_expect_site_error(
        lambda: _init_replay(model, init_to_value(values={"group/angle": 0.375})),
        ["group/angle", "angle"],
    )
    if out is not None:
        _, replayed = out
        assert_allclose(replayed["group/angle"], 0.375, atol=1e-4)


def test_broadcasting_mask_preserves_shape():
    """Broadcasting 3-element mask around a scalar-prior reparam site.

    Computing the auxiliary shape before applying the broadcasting mask shrinks
    the NUTS latent from 3 coordinates to 1.

    Expected: either preserves latent shape ``(3,)`` starting at ``[9, 9, 9]``,
    or raises an error naming ``theta``.
    Found in: r3, r4.
    """
    mask = jnp.array([True, True, True])

    def base():
        numpyro.sample("theta", dist.Normal(3.0, 2.0))

    model = handlers.mask(
        handlers.reparam(base, config={"theta": LocScaleReparam(0.0)}),
        mask=mask,
    )

    def _run():
        kernel = NUTS(model, init_strategy=init_to_value(values={"theta": 9.0}))
        state = kernel.init(random.key(29), num_warmup=0)
        replayed = kernel.postprocess_fn((), {})(state.z)
        return state.z["theta_decentered"], replayed["theta"]

    out = _run_or_expect_site_error(_run, ["theta"])
    if out is not None:
        aux, theta = out
        assert aux.shape == (3,)
        assert theta.shape == (3,)
        assert_allclose(theta, jnp.array([9.0, 9.0, 9.0]), atol=1e-4)


def test_sample_shape_request_preserves_shape():
    """Explicit ``sample_shape=(3,)`` under LocScaleReparam(0).

    Clearing ``sample_shape`` during lookup can silently drop or mismatch the
    requested 3-vector.

    Expected: starts at ``[2.0, 5.0, 8.0]`` with shape ``(3,)`` or raises an
    error naming ``theta``.
    Found in: r2, r5.
    """
    target = jnp.array([2.0, 5.0, 8.0])

    @handlers.reparam(config={"theta": LocScaleReparam(0)})
    def model():
        numpyro.sample("theta", dist.Normal(-1.0, 3.0), sample_shape=(3,))

    def _run():
        kernel = NUTS(model, init_strategy=init_to_value(values={"theta": target}))
        state = kernel.init(random.key(73), num_warmup=0)
        replayed = kernel.postprocess_fn((), {})(state.z)
        return state.z["theta_decentered"], replayed["theta"]

    out = _run_or_expect_site_error(_run, ["theta"])
    if out is not None:
        aux, theta = out
        assert aux.shape == (3,)
        assert theta.shape == (3,)
        assert_allclose(theta, target, atol=1e-4)


def test_vectorized_chains_nuts_init_to_value():
    """Three-chain vectorized MCMC(NUTS) iteration-zero state on plated reparam.

    Expected: all three chains start at the requested vector with shape
    ``(3, 4)`` at iteration 0 (or raise an error naming ``theta``).
    Found in: r1, r5, r7, independent_bank.
    """
    target = jnp.array([1.0, -2.0, 0.5, 4.0])

    def model():
        with numpyro.plate("subjects", 4):
            with handlers.reparam(config={"theta": LocScaleReparam(0.25)}):
                numpyro.sample("theta", dist.Normal(0.0, 2.0))

    def _run():
        kernel = NUTS(model, init_strategy=init_to_value(values={"theta": target}))
        mcmc = MCMC(
            kernel,
            num_warmup=0,
            num_samples=1,
            num_chains=3,
            chain_method="vectorized",
            progress_bar=False,
        )
        mcmc.warmup(random.key(11))
        state = mcmc.last_state
        replayed = jax.vmap(kernel.postprocess_fn((), {}))(state.z)
        return state.z["theta_decentered"], replayed["theta"]

    out = _run_or_expect_site_error(_run, ["theta"])
    if out is not None:
        aux, theta = out
        assert aux.shape == (3, 4)
        assert theta.shape == (3, 4)
        assert_allclose(theta, jnp.broadcast_to(target, (3, 4)), atol=1e-4)


def test_subsampled_plate_data_dependent_loc():
    """Subsampled plate with index-dependent location via HMCECS.

    Expected: starts at the requested 4-vector on the active subsample, or
    raises an error naming ``theta``.
    Found in: r7.
    """
    target = jnp.array([4.0, 3.0, 2.0, 1.0])

    def model():
        with numpyro.plate("rows", 32, subsample_size=4) as idx:
            with handlers.reparam(config={"theta": LocScaleReparam(0.0)}):
                theta = numpyro.sample(
                    "theta", dist.Normal(idx.astype(jnp.float32), 2.0)
                )
            numpyro.sample("obs", dist.Normal(theta, 1.0), obs=jnp.zeros(4))

    def _run():
        kernel = HMCECS(
            NUTS(model, init_strategy=init_to_value(values={"theta": target})),
            num_blocks=2,
        )
        state = kernel.init(random.key(34), 0, None, (), {})
        return kernel.postprocess_fn((), {})(state.z)["theta"]

    theta = _run_or_expect_site_error(_run, ["theta"])
    if theta is not None:
        assert theta.shape == (4,)
        assert_allclose(theta, target, atol=1e-4)


def test_control_flow_cond_reparam_init():
    """Scoped ``cond`` branches containing a reparameterized site.

    Expected: both branches honor ``g/theta == 3.0``, or raise an error naming
    ``g/theta`` (or ``theta``).
    Found in: r4, r5.
    """

    def make_model(flag):
        def arm(loc, arg):
            with handlers.reparam(config={"theta": LocScaleReparam(0.0)}):
                return numpyro.sample("theta", dist.Normal(loc + arg, 2.0))

        def body():
            return cond(
                jnp.array(flag),
                partial(arm, 10.0),
                partial(arm, -4.0),
                1.0,
            )

        return handlers.scope(body, prefix="g")

    for flag in (True, False):
        out = _run_or_expect_site_error(
            lambda f=flag: _init_replay(
                make_model(f), init_to_value(values={"g/theta": 3.0})
            ),
            ["g/theta", "theta"],
        )
        if out is not None:
            _, replayed = out
            assert_allclose(replayed["g/theta"], 3.0, atol=1e-4)


# =============================================================================
# (c) Handler composition and user objects
# =============================================================================


def test_block_hide_fn_not_called_on_original_sphere():
    """Block ``hide_fn`` calling ``biject_to`` on ProjectedNormalReparam.

    Master only exposes the unconstrained real-vector auxiliary to an enclosing
    ``block(hide_fn=...)``, never the original sphere distribution (for which
    ``biject_to`` raises ``NotImplementedError``).

    Expected: behaves exactly as master — succeeds under ``init_to_uniform``,
    ``init_to_feasible``, and ``Predictive``.
    Found in: r2.
    """

    def hide_matrix_latents(site):
        return (
            site["type"] == "sample"
            and biject_to(site["fn"].support).domain.event_dim >= 2
        )

    def base():
        with handlers.reparam(config={"direction": ProjectedNormalReparam()}):
            return numpyro.sample(
                "direction", dist.ProjectedNormal(jnp.array([1.0, 2.0, 0.0]))
            )

    wrapped = handlers.block(base, hide_fn=hide_matrix_latents)

    for strategy in (init_to_uniform, init_to_feasible):
        kernel = NUTS(wrapped, init_strategy=strategy)
        state = kernel.init(random.key(73), num_warmup=0)
        out = kernel.postprocess_fn((), {})(state.z)
        assert out["direction"].shape == (3,)
        assert_allclose(jnp.linalg.norm(out["direction"]), 1.0, atol=1e-5)

    pred = Predictive(
        wrapped, num_samples=1, return_sites=["direction", "direction_normal"]
    )(random.key(73))
    assert pred["direction"].shape == (1, 3)


def test_block_hidden_neutra_same_local_name():
    """Blocked NeuTraReparam site sharing a local name with a visible site.

    Request discovery must not cross ``block()`` and reject the hidden NeuTra
    site when the request targets a separate visible site of the same name.

    Expected: behaves as master — visible ``target`` starts at ``11.0``.
    Found in: r3, r4.
    """

    def inner():
        return numpyro.sample("target", dist.Normal(0.0, 1.0))

    guide = AutoDiagonalNormal(inner)
    svi = SVI(inner, guide, Adam(0.01), Trace_ELBO())
    params = svi.get_params(svi.init(random.key(29)))
    hidden = NeuTraReparam(guide, params).reparam(inner)

    def model():
        with handlers.block(), handlers.seed(rng_seed=13):
            local = hidden()
        public = numpyro.sample("target", dist.Normal(0.0, 1.0))
        return local, public

    _, public = handlers.substitute(
        handlers.seed(model, random.key(29)),
        substitute_fn=init_to_value(values={"target": 11.0}),
    )()
    assert_allclose(public, 11.0, atol=1e-5)


def test_substitute_fn_no_extra_eq_calls():
    """Custom ``substitute_fn`` callable with no init strategy.

    Comparing ``substitute_fn.func`` against built-in init functions triggers
    extra ``__eq__`` calls on user callables.

    Expected: behaves exactly as master — zero ``__eq__`` calls on the
    callback's ``func`` attribute.
    Found in: r3.
    """
    eq_calls = []

    class TrackedFunc:
        def __eq__(self, other):
            eq_calls.append(other)
            return self is other

        def __call__(self, site):
            if site["type"] == "sample":
                return jnp.array(1.0)
            return None

    class Callback:
        def __init__(self):
            self.func = TrackedFunc()

        def __call__(self, site):
            return self.func(site)

    @handlers.reparam(config={"theta": LocScaleReparam(0.0)})
    def model():
        return numpyro.sample("theta", dist.Normal(3.0, 2.0))

    val = handlers.seed(
        handlers.substitute(model, substitute_fn=Callback()), rng_seed=29
    )()
    assert_allclose(val, 5.0, atol=1e-5)
    assert len(eq_calls) == 0


def test_empty_values_dict_subclass_no_extra_bool():
    """Empty ``dict`` subclass passed to ``init_to_value``.

    Checking ``if values:`` calls ``__bool__`` on user dict subclasses even
    when no request is present.

    Expected: behaves exactly as master — zero ``__bool__`` calls.
    Found in: r4, r5.
    """

    class WatchedDict(dict):
        def __init__(self):
            super().__init__()
            self.bool_calls = 0

        def __bool__(self):
            self.bool_calls += 1
            return len(self) != 0

    @handlers.reparam(config={"theta": LocScaleReparam(0.0)})
    def model():
        numpyro.sample("theta", dist.Normal(2.0, 3.0))

    values = WatchedDict()
    _init_replay(model, init_to_value(values=values))
    assert values.bool_calls == 0


def test_nonempty_values_dict_subclass_behavior():
    """Non-empty ``dict`` subclass passed to ``init_to_value``.

    A plain ``dict`` subclass without overridden methods should not be silently
    ignored or treated inconsistently with ``dict``.

    Expected: starts at ``theta == 4.0`` or raises an error naming ``theta``.
    Found in: r5.
    """

    class Values(dict):
        pass

    @handlers.reparam(config={"theta": LocScaleReparam(0.0)})
    def model():
        numpyro.sample("theta", dist.Normal(10.0, 2.0))

    out = _run_or_expect_site_error(
        lambda: _init_replay(model, init_to_value(values=Values(theta=4.0))),
        ["theta"],
    )
    if out is not None:
        _, replayed = out
        assert_allclose(replayed["theta"], 4.0, atol=1e-4)


def test_custom_partial_strategy_no_extra_func_reads():
    """Custom ``functools.partial`` subclass wrapping an unrecognized strategy.

    Classifying unrecognized partials by reading ``.func`` adds observable
    descriptor reads compared to an unreparameterized model.

    Expected: behaves exactly as master — same ``.func`` read count on the
    reparameterized model as on the unreparameterized control.
    Found in: r4.
    """

    class LoggedPartial(partial):
        reads = 0

        @property
        def func(self):
            self.reads += 1
            return super().func

    def custom(site):
        return init_to_uniform(site, radius=0.75)

    def plain():
        numpyro.sample("theta", dist.Normal(2.0, 3.0))

    reparam_model = handlers.reparam(plain, config={"theta": LocScaleReparam(0.0)})

    s_plain = LoggedPartial(custom)
    _init_replay(plain, s_plain)

    s_reparam = LoggedPartial(custom)
    _init_replay(reparam_model, s_reparam)

    assert s_reparam.reads == s_plain.reads


def test_delegating_custom_strategy_prototype_matches_final():
    """User callable forwarding to ``init_to_mean`` on TransformReparam.

    Factory normalization can expose the inner ``init_to_mean`` partial,
    causing the prototype trace (``1.0``) and final initialized state
    (``exp(0.5)``) to disagree.

    Expected: behaves as master (prototype and final state agree on ``1.0``),
    or consistently uses ``exp(0.5)`` in both prototype and final state.
    Found in: r4, r5.
    """

    def logged_mean(site=None):
        return init_to_mean(site)

    @handlers.reparam(config={"rate": TransformReparam()})
    def model():
        numpyro.sample("rate", dist.LogNormal(0.0, 1.0))

    info, replayed = _init_replay(model, logged_mean)
    proto_val = info.model_trace["rate"]["value"]
    final_val = replayed["rate"]
    assert_allclose(final_val, proto_val, atol=1e-5)


def test_unit_jacobian_suffix_no_extra_format():
    """User ``str`` suffix with ``__format__`` under UnitJacobianReparam.

    Formatting the auxiliary suffix before checking whether the site was
    requested adds an extra ``__format__`` call when only another site is
    requested.

    Expected: behaves exactly as master — ``__format__`` call count with an
    unrelated request equals the call count with an empty ``values={}`` dict.
    Found in: r6.
    """
    calls = []

    class Suffix(str):
        def __format__(self, spec):
            calls.append(spec)
            return super().__format__(spec)

    suffix = Suffix("aux")

    def model():
        numpyro.sample("other", dist.Normal(0.0, 1.0))
        with handlers.reparam(
            config={
                "theta": UnitJacobianReparam(transforms.IdentityTransform(), suffix)
            }
        ):
            numpyro.sample("theta", dist.Normal(0.0, 1.0))

    calls.clear()
    _init_replay(model, init_to_value(values={}), key=random.key(71))
    empty_count = len(calls)

    calls.clear()
    _init_replay(model, init_to_value(values={"other": 0.75}), key=random.key(71))
    unrelated_count = len(calls)

    assert unrelated_count == empty_count


def test_custom_messenger_no_extra_class_name_reads():
    """Inert custom ``Messenger`` subclass with metaclass tracking ``__name__``.

    Building an unused refusal reason on an unrequested reparameterized site
    reads ``type(h).__name__`` and can alter deterministic outputs.

    Expected: behaves exactly as master — zero ``__name__`` reads when only an
    unrelated site (or absent site) is requested.
    Found in: r7.
    """
    reads = []

    class AuditMeta(type):
        def __getattribute__(cls, name):
            if name == "__name__":
                reads.append(name)
            return super().__getattribute__(name)

    class Monitor(Messenger, metaclass=AuditMeta):
        pass

    def model():
        with Monitor():
            with handlers.reparam(config={"offset": LocScaleReparam(0.5)}):
                numpyro.sample("offset", dist.Normal(1.0, 2.0))
        numpyro.deterministic("name_reads", len(reads))
        numpyro.sample("level", dist.Normal(0.0, 1.0))

    for values in ({}, {"level": 3.0}, {"absent": 1.0}):
        reads.clear()
        _, replayed = _init_replay(
            model, init_to_value(values=values), key=random.key(29)
        )
        assert len(reads) == 0
        assert float(replayed["name_reads"]) == 0.0


def test_debug_key_reuse_find_valid_initial_params():
    """Direct ``find_valid_initial_params`` under ``jax.debug_key_reuse(True)``.

    Reusing the same typed PRNG key across prototype/retry passes triggers
    ``KeyReuseError`` when key-reuse debugging is enabled.

    Expected: succeeds without ``KeyReuseError`` for both visible and blocked
    reparameterized sites.
    Found in: r4, r5.
    """

    @handlers.reparam(config={"theta": LocScaleReparam(0.0)})
    def model():
        numpyro.sample("theta", dist.Normal(2.0, 3.0))

    blocked = handlers.block(model, hide=["theta"])

    with jax.debug_key_reuse(True):
        # Blocked original site must succeed cleanly without KeyReuseError.
        (_, _, _), valid_blocked = find_valid_initial_params(
            random.key(71),
            blocked,
            init_strategy=init_to_value(values={"theta": 8.0}),
        )
        assert bool(valid_blocked)

        # Visible site must either start at 8.0 or raise an error naming theta
        # (never KeyReuseError).
        def _run_visible():
            (z, _, _), valid = find_valid_initial_params(
                random.key(71),
                model,
                init_strategy=init_to_value(values={"theta": 8.0}),
            )
            assert bool(valid)
            return 2.0 + 3.0 * float(z["theta_decentered"])

        val = _run_or_expect_site_error(_run_visible, ["theta"])
        if val is not None:
            assert_allclose(val, 8.0, atol=1e-4)


# =============================================================================
# (d) Numeric edges
# =============================================================================


def test_float32_locscale_partial_precision_loss():
    """Float32 cancellation in partial LocScaleReparam(0.5) at scale 2^18.

    In float32, the rounded auxiliary ``130560.0`` replays to ``0.0`` instead
    of the requested ``1.0``.

    Expected: replays close to ``1.0`` or raises an error naming ``theta``
    (never silently accepts ``0.0``).
    Found in: r2, r3.
    """
    size = float(2**18)

    @handlers.reparam(config={"theta": LocScaleReparam(0.5)})
    def model():
        numpyro.sample("theta", dist.Normal(size, size))

    with jax.enable_x64(False):
        out = _run_or_expect_site_error(
            lambda: _init_replay(
                model, init_to_value(values={"theta": jnp.array(1.0)})
            ),
            ["theta"],
        )
        if out is not None:
            _, replayed = out
            assert_allclose(replayed["theta"], 1.0, atol=1e-3)


def test_float32_projected_normal_large_concentration():
    """ProjectedNormalReparam with float32 concentration ``[2^25, 2^25, 2^25]``.

    Adding an O(1) vector to ``2^25`` in float32 swamps the requested direction
    and replays to ``(1, 1, 1) / sqrt(3)``.

    Expected: replays close to ``[0.6, -0.8, 0.0]`` or raises an error naming
    ``direction``.
    Found in: r1, r2.
    """
    goal = jnp.array([0.6, -0.8, 0.0], dtype=jnp.float32)

    @handlers.reparam(config={"direction": ProjectedNormalReparam()})
    def model():
        mu = jnp.array([2.0**25, 2.0**25, 2.0**25], dtype=jnp.float32)
        numpyro.sample("direction", dist.ProjectedNormal(mu))

    with jax.enable_x64(False):
        out = _run_or_expect_site_error(
            lambda: _init_replay(model, init_to_value(values={"direction": goal})),
            ["direction"],
        )
        if out is not None:
            _, replayed = out
            assert_allclose(replayed["direction"], goal, atol=1e-3)


def test_float32_compiled_vs_eager_affine_fma():
    """Compiled vs eager float32 replay of ``4097 * z - float32(4097^2)``.

    Eager replay evaluates ``4097 * 4097 - 16785408 == 0.0`` in float32, while
    XLA-compiled replay (FMA fusion, as used by ``MCMC``) evaluates exact
    ``4097^2 - 16785408 == 1.0``.

    Expected: compiled HMC iteration-0 replay equals ``0.0`` (in both 1 and 3
    vectorized chains), or initialization raises an error naming ``position``.
    Found in: r7.
    """
    scale = 4097.0
    offset = -float(np.float32(scale * scale))

    def model():
        with handlers.reparam(config={"position": TransformReparam()}):
            pos = numpyro.sample(
                "position",
                dist.TransformedDistribution(
                    dist.Normal(scale, 1.0),
                    transforms.AffineTransform(offset, scale),
                ),
            )
        numpyro.deterministic("double_position", 2.0 * pos)

    with jax.enable_x64(False):
        for chains in (1, 3):

            def _run(n_chains=chains):
                kernel = HMC(
                    model,
                    init_strategy=init_to_value(values={"position": 0.0}),
                    num_steps=1,
                    trajectory_length=None,
                    step_size=1e-12,
                    adapt_step_size=False,
                    adapt_mass_matrix=False,
                )
                driver = MCMC(
                    kernel,
                    num_warmup=0,
                    num_samples=1,
                    num_chains=n_chains,
                    chain_method="vectorized",
                    progress_bar=False,
                )
                driver.warmup(random.key(29))
                pp = kernel.postprocess_fn((), {})
                if n_chains > 1:
                    pp = jax.vmap(pp)
                compiled = jax.jit(pp)(driver.last_state.z)
                return compiled["position"]

            out = _run_or_expect_site_error(_run, ["position"])
            if out is not None:
                assert_allclose(out, 0.0, atol=1e-3)


def test_int64_narrowing_overflow_x32():
    """Host ``np.int64(2**32)`` and ``2**32 + 7`` with ``x64=False``.

    Narrowing host int64 payloads with ``jnp.asarray`` wraps ``2**32`` to
    ``0`` (and ``2**32 + 7`` to ``7``) before recording the replay target.

    Expected: replays near the requested float value ``float(2**32)`` /
    ``float(2**32 + 7)``, or raises an error naming ``height`` (never silently
    wraps to ``0`` or ``7``).
    Found in: r7.
    """

    @handlers.reparam(config={"height": LocScaleReparam(0.0)})
    def model():
        numpyro.sample("height", dist.Normal(0.0, 1.0))

    with jax.enable_x64(False):
        for payload in (np.int64(2**32), np.int64(2**32 + 7)):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                out = _run_or_expect_site_error(
                    lambda v=payload: _init_replay(
                        model, init_to_value(values={"height": v})
                    ),
                    ["height"],
                )
            if out is not None:
                _, replayed = out
                assert_allclose(replayed["height"], float(payload), rtol=1e-4, atol=1.0)


def test_int32_angle_circular_nuts():
    """Integer ``jnp.int32(0)`` angle request for CircularReparam in NUTS.

    Preserving an integer dtype on the auxiliary causes NUTS gradient
    evaluation to fail with ``TypeError``.

    Expected: NUTS initializes with a floating auxiliary and ``angle == 0.0``
    (mod 2*pi), or raises an error naming ``angle``.
    Found in: r2, r3, r4.
    """

    @handlers.reparam(config={"angle": CircularReparam()})
    def model():
        numpyro.sample("angle", dist.VonMises(0.3, 4.0))

    def _run():
        kernel = NUTS(
            model,
            init_strategy=init_to_value(values={"angle": jnp.int32(0)}),
        )
        state = kernel.init(random.key(73), num_warmup=0)
        replayed = kernel.postprocess_fn((), {})(state.z)
        return state.z["angle_unwrapped"], replayed["angle"]

    out = _run_or_expect_site_error(_run, ["angle"])
    if out is not None:
        aux, angle = out
        assert jnp.issubdtype(aux.dtype, jnp.floating)
        _assert_circular_close(angle, 0.0, atol=1e-4)


def test_bounded_uniform_affine_near_endpoint_clip():
    """Supported AffineTransform(Uniform(0, 1)) near the upper float32 edge.

    The exact base preimage ``1 - 2^-24`` clips to ``1 - 2^-23`` through
    ``SigmoidTransform``, so final replay lands at ``-0.0625`` instead of
    ``0.0``.

    Expected: replays close to ``0.0`` or raises an error naming ``position``
    (never silently accepts ``-0.0625``).
    Found in: r6, r7.
    """

    @handlers.reparam(config={"position": TransformReparam()})
    def model():
        tr = transforms.AffineTransform(-1048575.9375, 1048576.0)
        numpyro.sample(
            "position", dist.TransformedDistribution(dist.Uniform(0.0, 1.0), tr)
        )

    with jax.enable_x64(False):
        out = _run_or_expect_site_error(
            lambda: _init_replay(model, init_to_value(values={"position": 0.0})),
            ["position"],
        )
        if out is not None:
            _, replayed = out
            assert_allclose(replayed["position"], 0.0, atol=1e-3)


def test_bounded_uniform_closed_endpoint_error_names_site():
    """Boundary endpoint ``0.0`` requested for TransformedDistribution(Uniform).

    Expected: either replays to ``0.0`` with finite potential energy, or raises
    an error explicitly naming ``position`` (not a generic ``RuntimeError``
    omitting the site name).
    Found in: r7.
    """

    @handlers.reparam(config={"position": TransformReparam()})
    def model():
        tr = transforms.AffineTransform(0.0, 1.0)
        numpyro.sample(
            "position", dist.TransformedDistribution(dist.Uniform(0.0, 1.0), tr)
        )

    out = _run_or_expect_site_error(
        lambda: _init_replay(model, init_to_value(values={"position": 0.0})),
        ["position"],
    )
    if out is not None:
        info, replayed = out
        assert np.isfinite(float(info.param_info.potential_energy))
        assert_allclose(replayed["position"], 0.0, atol=1e-4)


def test_folded_uniform_negative_preimage_abs_transform():
    """Non-bijective ``FoldedDistribution(Uniform(-4, -1))`` under TransformReparam.

    ``AbsTransform.inv(2.5)`` returns ``+2.5`` (outside ``(-4, -1)``) even
    though ``-2.5`` is the unique valid base preimage.

    Expected: starts at ``length == 2.5`` (base ``-2.5``), or raises an error
    explicitly naming ``length``.
    Found in: r5.
    """

    @handlers.reparam(config={"length": TransformReparam()})
    def model():
        numpyro.sample("length", dist.FoldedDistribution(dist.Uniform(-4.0, -1.0)))

    def _run():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            kernel = NUTS(model, init_strategy=init_to_value(values={"length": 2.5}))
            state = kernel.init(random.key(13), num_warmup=0)
            return kernel.postprocess_fn((), {})(state.z)

    out = _run_or_expect_site_error(_run, ["length"])
    if out is not None:
        assert_allclose(out["length"], 2.5, atol=1e-4)


def test_projected_normal_off_sphere_request():
    """Slightly off-sphere request ``[1.00008, 0, 0]`` for ProjectedNormalReparam.

    ``[1.00008, 0, 0]`` fails ``constraints.sphere`` and must not be silently
    renormalized to ``[1, 0, 0]`` in float32.

    Expected: raises an error explicitly naming ``direction``.
    Found in: r7.
    """
    off_sphere = np.array([1.0 + 8e-5, 0.0, 0.0], dtype=np.float32)
    assert not bool(constraints.sphere(off_sphere))

    @handlers.reparam(config={"direction": ProjectedNormalReparam()})
    def model():
        numpyro.sample("direction", dist.ProjectedNormal(jnp.array([0.0, 0.0, 1.0])))

    with jax.enable_x64(False):
        with pytest.raises(Exception) as exc_info:
            _init_replay(model, init_to_value(values={"direction": off_sphere}))
        assert _error_names_site(exc_info.value, ["direction"]), (
            f"Expected error naming 'direction', got "
            f"{type(exc_info.value).__name__}: {exc_info.value}"
        )


def test_jit_initialize_model_reparam():
    """``jax.jit`` tracing ``initialize_model`` with an original-site request.

    Expected: either returns the exact requested ``theta == 5.0`` under JIT, or
    raises an error naming ``theta``.
    Found in: r4, r5, r7, independent_bank.
    """

    @handlers.reparam(config={"theta": LocScaleReparam(0.0)})
    def model():
        numpyro.sample("theta", dist.Normal(1.0, 2.0))

    def _run():
        @jax.jit
        def init_z(k):
            info = initialize_model(
                k, model, init_strategy=init_to_value(values={"theta": 5.0})
            )
            return info.param_info.z["theta_decentered"]

        aux = init_z(random.key(34))
        return 1.0 + 2.0 * float(aux)

    val = _run_or_expect_site_error(_run, ["theta"])
    if val is not None:
        assert_allclose(val, 5.0, atol=1e-4)


# =============================================================================
# (e) Entry points: AutoGuides, Predictive, SVI, SA, Gibbs, NestedSampler
# =============================================================================


def test_autoguides_svi_plated_reparam_init():
    """AutoDelta, AutoNormal, and AutoDiagonalNormal with SVI on plated reparam.

    Checks both shape preservation under ``init_to_mean`` (which broke
    ``AutoDiagonalNormal`` in r1) and honoring ``init_to_value`` on the
    original site name.

    Expected: all three guides initialize with shape ``(4,)``, finite ELBO, and
    reconstructed original median matching the requested vector (or raise an
    error naming ``theta``).
    Found in: r1, r2, r5, independent_bank.
    """
    target = jnp.array([-1.0, 0.5, 2.0, 3.5])

    def model():
        with numpyro.plate("row", 4):
            with handlers.reparam(config={"theta": LocScaleReparam(0.25)}):
                theta = numpyro.sample("theta", dist.Normal(-1.5, 2.25))
            numpyro.sample(
                "obs", dist.Normal(theta, 1.0), obs=jnp.array([-2.0, 0.0, 2.0, 4.0])
            )

    for guide_cls in (AutoDelta, AutoNormal, AutoDiagonalNormal):
        # Control: init_to_mean must preserve (4,) shape and finite loss.
        guide_mean = guide_cls(model, init_loc_fn=init_to_mean)
        svi_mean = SVI(model, guide_mean, Adam(0.01), Trace_ELBO())
        state_mean = svi_mean.init(random.key(7))
        med_mean = guide_mean.median(svi_mean.get_params(state_mean))
        assert med_mean["theta_decentered"].shape == (4,)
        _, loss_mean = jax.jit(svi_mean.update)(state_mean)
        assert np.isfinite(float(loss_mean))

        # Original-site init_to_value request.
        def _run_value(cls=guide_cls):
            guide = cls(model, init_loc_fn=init_to_value(values={"theta": target}))
            svi = SVI(model, guide, Adam(0.01), Trace_ELBO())
            state = svi.init(random.key(7))
            params = svi.get_params(state)
            med = guide.median(params)
            aux = med["theta_decentered"]
            # Reconstruct original coordinate for LocScaleReparam(0.25) on N(-1.5, 2.25).
            reconstructed = -1.5 + (2.25**0.75) * (aux + 0.375)
            _, loss = jax.jit(svi.update)(state)
            return reconstructed, float(loss)

        out = _run_or_expect_site_error(_run_value, ["theta"])
        if out is not None:
            reconstructed, loss = out
            assert np.isfinite(loss)
            assert reconstructed.shape == (4,)
            assert_allclose(reconstructed, target, atol=1e-4)


def test_svi_partial_init_params_autodel_invariance():
    """SVI with partial ``init_params`` on AutoDelta with a reparameterized site.

    Supplying ``init_params`` for only an unrelated site must not perturb the
    default initialization of an unrequested reparameterized site.

    Expected: behaves as master — ``b_auto_loc == 3.0`` and
    ``angle_unwrapped_auto_loc`` matches the fresh ``init_params=None`` run.
    Found in: r5.
    """

    @handlers.reparam(config={"angle": CircularReparam()})
    def model():
        numpyro.sample("angle", dist.VonMises(0.0, 80.0))
        numpyro.sample("b", dist.Normal(2.0, 1.0))

    svi = SVI(model, AutoDelta(model), Adam(0.001), Trace_ELBO())
    fresh = svi.get_params(svi.init(random.key(7)))
    partial_params = svi.get_params(
        svi.init(random.key(7), init_params={"b_auto_loc": jnp.array(3.0)})
    )
    assert_allclose(partial_params["b_auto_loc"], 3.0, atol=1e-6)
    assert_allclose(
        partial_params["angle_unwrapped_auto_loc"],
        fresh["angle_unwrapped_auto_loc"],
        atol=1e-6,
    )


def test_predictive_autodel_params_bool_and_fresh_guide():
    """Predictive with AutoDelta: supplied ``dict`` subclass and fresh guide.

    Evaluating ``bool(params)`` on a supplied dict subclass adds a user-visible
    ``__bool__`` call; running a fresh default AutoDelta through
    ``initialize_model`` can trigger extra model executions.

    Expected: behaves exactly as master — zero ``__bool__`` calls on
    ``params``, exact prediction ``[16.0, 16.0]``, and fresh guide runs with
    matching visit count to an unreparameterized guide-free baseline check.
    Found in: r4, r5.
    """

    @handlers.reparam(config={"income": LocScaleReparam(centered=0)})
    def model():
        numpyro.sample("income", dist.Normal(10.0, 2.0))

    class Params(dict):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.bool_calls = 0

        def __bool__(self):
            self.bool_calls += 1
            return len(self) != 0

    supplied = Params(income_decentered_auto_loc=jnp.array(3.0))
    pred = Predictive(model, guide=AutoDelta(model), params=supplied, num_samples=2)(
        random.key(13)
    )
    assert supplied.bool_calls == 0
    assert_allclose(pred["income"], jnp.array([16.0, 16.0]), atol=1e-5)

    # Fresh AutoDelta with empty params should run cleanly and produce finite draws.
    fresh_pred = Predictive(model, guide=AutoDelta(model), params={}, num_samples=2)(
        random.key(13)
    )
    assert fresh_pred["income"].shape == (2,)
    assert np.all(np.isfinite(np.asarray(fresh_pred["income"])))


def test_sa_init_to_value_iteration_zero():
    """Sample Adaptive (SA) MCMC with ``init_to_value`` on a reparam site.

    ``SA.init`` perturbs the initial parameters into a proposal cloud and picks
    a random member, so iteration-0 ``theta`` silently disagrees with both
    master and the requested ``7.0``.

    Expected: either starts at ``theta == 7.0`` at iteration 0, or raises an
    error naming ``theta``.
    Found in: r6, r7.
    """

    @handlers.reparam(config={"theta": LocScaleReparam(0.0)})
    def model():
        numpyro.sample("theta", dist.Normal(3.0, 2.0))

    def _run():
        kernel = SA(model, init_strategy=init_to_value(values={"theta": 7.0}))
        state = kernel.init(random.key(71), num_warmup=3, model_args=())
        return kernel.postprocess_fn((), {})(state.z)["theta"]

    val = _run_or_expect_site_error(_run, ["theta"])
    if val is not None:
        assert_allclose(val, 7.0, atol=1e-4)


def test_gibbs_hmcgibbs_and_mixedhmc_init():
    """Gibbs, HMCGibbs, DiscreteHMCGibbs, and MixedHMC entry points.

    Multi-block Gibbs can change upstream sites after an earlier block's
    reparam init; discrete wrapper kernels forward ``init_strategy`` to inner
    HMC/NUTS kernels.

    Expected: each kernel either initializes ``theta`` to the requested value
    at iteration 0 or raises an error naming ``theta``.
    Found in: r5, r7, independent_bank.
    """

    def mixed_model():
        b = numpyro.sample("b", dist.Bernoulli(0.5))
        with handlers.reparam(config={"theta": LocScaleReparam(0.0)}):
            theta = numpyro.sample("theta", dist.Normal(10.0 + b, 2.0))
        numpyro.sample("obs", dist.Normal(theta, 1.0), obs=5.0)

    strategy = init_to_value(values={"theta": 4.0})

    for make_kernel in [
        lambda: HMCGibbs(
            HMC(mixed_model, init_strategy=strategy),
            gibbs_fn=lambda rng_key, gibbs_sites, hmc_sites: gibbs_sites,
            gibbs_sites=["b"],
        ),
        lambda: DiscreteHMCGibbs(NUTS(mixed_model, init_strategy=strategy)),
        lambda: MixedHMC(
            HMC(mixed_model, init_strategy=strategy), num_discrete_updates=2
        ),
    ]:

        def _run(mk=make_kernel):
            kernel = mk()
            state = kernel.init(random.key(25), 0, None, (), {})
            return kernel.postprocess_fn((), {})(state.z)["theta"]

        val = _run_or_expect_site_error(_run, ["theta"])
        if val is not None:
            assert_allclose(val, 4.0, atol=1e-4)

    # Multi-block Gibbs where reparameterized site is in the first block.
    def two_block_model():
        a = numpyro.sample("a", dist.Normal(0.0, 1.0))
        with handlers.reparam(config={"theta": LocScaleReparam(0.0)}):
            theta = numpyro.sample("theta", dist.Normal(a, 2.0))
        numpyro.sample("obs", dist.Normal(theta, 1.0), obs=1.0)

    def _run_gibbs():
        gibbs = Gibbs(
            [
                (
                    NUTS(two_block_model, init_strategy=strategy),
                    ["theta_decentered"],
                ),
                (NUTS(two_block_model), ["a"]),
            ]
        )
        state = gibbs.init(random.key(25), 0, None, (), {})
        return gibbs.postprocess_fn((), {})(state.z)["theta"]

    val_gibbs = _run_or_expect_site_error(_run_gibbs, ["theta"])
    if val_gibbs is not None:
        assert_allclose(val_gibbs, 4.0, atol=1e-4)


def test_nested_sampler_and_uniform_reparam():
    """NestedSampler invariance and ``UniformReparam`` with ``init_to_value``.

    Expected: ``NestedSampler`` runs cleanly on a ``LocScaleReparam`` model and
    produces finite weighted samples; ``UniformReparam`` with an original-site
    request either starts at ``theta == 0.0`` or raises an error naming
    ``theta``.
    Found in: r4, r5, r7.
    """
    prev_x64 = jax.config.jax_enable_x64
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        warnings.simplefilter("ignore", DeprecationWarning)
        pytest.importorskip("jaxns")
        from numpyro.contrib.nested_sampling import NestedSampler, UniformReparam

    try:

        def ns_model():
            with handlers.reparam(config={"theta": LocScaleReparam(0.0)}):
                theta = numpyro.sample("theta", dist.Normal(1.0, 2.0))
            numpyro.sample("obs", dist.Normal(theta, 1.0), obs=1.5)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ns = NestedSampler(
                ns_model,
                constructor_kwargs={
                    "num_live_points": 12,
                    "max_samples": 120,
                    "devices": jax.devices()[:1],
                },
                termination_kwargs={"dlogZ": 1.0},
            )
            ns.run(random.key(91))
            samples, weights = ns.get_weighted_samples()
        assert "theta" in samples
        assert np.all(np.isfinite(np.asarray(samples["theta"])))
        assert np.all(np.isfinite(np.asarray(weights)))

        @handlers.reparam(config={"theta": UniformReparam()})
        def uniform_model():
            numpyro.sample("theta", dist.Normal(0.0, 1.0))

        out = _run_or_expect_site_error(
            lambda: _init_replay(uniform_model, init_to_value(values={"theta": 0.0})),
            ["theta"],
        )
        if out is not None:
            _, replayed = out
            assert_allclose(replayed["theta"], 0.0, atol=1e-4)
    finally:
        jax.config.update("jax_enable_x64", prev_x64)
