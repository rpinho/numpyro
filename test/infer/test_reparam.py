# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

from functools import partial

import numpy as np
from numpy.testing import assert_allclose
import pytest

import jax
from jax import jacobian, jit, lax, random
import jax.numpy as jnp

import numpyro
import numpyro.distributions as dist
from numpyro.distributions import constraints
from numpyro.distributions.transforms import AffineTransform, ExpTransform
import numpyro.handlers as handlers
from numpyro.infer import MCMC, NUTS, SVI, Trace_ELBO, init_to_value
from numpyro.infer.autoguide import AutoDiagonalNormal, AutoIAFNormal
from numpyro.infer.reparam import (
    CircularReparam,
    DiscreteCosineReparam,
    ExplicitReparam,
    HaarReparam,
    LocScaleReparam,
    NeuTraReparam,
    ProjectedNormalReparam,
    TransformReparam,
)
from numpyro.infer.util import initialize_model, log_density
from numpyro.optim import Adam


# Test helper to extract a few central moments from samples.
def get_moments(x):
    m1 = jnp.mean(x, axis=0)
    x = x - m1
    xx = x * x
    xxx = x * xx
    xxxx = xx * xx
    m2 = jnp.mean(xx, axis=0)
    m3 = jnp.mean(xxx, axis=0) / m2**1.5
    m4 = jnp.mean(xxxx, axis=0) / m2**2
    return jnp.stack([m1, m2, m3, m4])


# Helper functions to get central circular moments
def mean_vector(x, m, n):
    s = jnp.mean(jnp.sin(n * (x - m)), axis=0)
    c = jnp.mean(jnp.cos(n * (x - m)), axis=0)
    return s, c


def circular_moment(x, n):
    m = jnp.arctan2(*mean_vector(x, 0.0, n))  # circular mean
    s, c = mean_vector(x, m, n)
    # direction = jnp.arctan2(s, c)
    length = jnp.hypot(s, c)
    return length
    # return jnp.array([direction, length])


def get_circular_moments(x):
    return jnp.stack([circular_moment(x, i) for i in range(1, 3)])


def test_syntax():
    loc = np.random.uniform(-1.0, 1.0, ())
    scale = np.random.uniform(0.5, 1.5, ())
    data = np.random.uniform(-1.0, 1.0, ())
    config = {"x": LocScaleReparam(), "y": LocScaleReparam()}

    def model():
        x = numpyro.sample("x", dist.Normal(loc, scale))
        y = numpyro.sample("y", dist.Normal(x, scale))
        numpyro.sample("z", dist.Normal(y, scale), obs=data)

    # Context manager syntax.
    seed = numpyro.handlers.seed(rng_seed=0)
    trace = numpyro.handlers.trace()
    reparam = numpyro.handlers.reparam(config=config)
    with reparam, trace, seed:
        model()
    tr1 = trace.trace

    # Eager function syntax.
    seed = numpyro.handlers.seed(model, rng_seed=0)
    trace = numpyro.handlers.trace(seed)
    reparam = numpyro.handlers.reparam(trace, config=config)
    reparam()
    tr2 = trace.trace

    # Partial function syntax.
    seed = numpyro.handlers.seed(rng_seed=0)
    trace = numpyro.handlers.trace()
    reparam = numpyro.handlers.reparam(config=config)
    m = model
    m = seed(m)
    m = trace(m)
    m = reparam(m)
    m()
    tr3 = trace.trace

    # Decorator syntax.
    seed = numpyro.handlers.seed(rng_seed=0)
    trace = numpyro.handlers.trace()
    reparam = numpyro.handlers.reparam(config=config)

    @reparam
    @trace
    @seed
    def m():
        return model()

    m()
    tr4 = trace.trace

    assert tr1.keys() == tr2.keys() == tr3.keys() == tr4.keys()


@pytest.mark.parametrize(
    "batch_shape,base_batch_shape",
    [((), ()), ((4,), (4,)), ((2, 3), (2, 3)), ((2, 3), ())],
    ids=str,
)
@pytest.mark.parametrize("event_shape", [(), (5,)], ids=str)
def test_log_normal(batch_shape, base_batch_shape, event_shape):
    shape = batch_shape + event_shape
    base_shape = base_batch_shape + event_shape
    loc = np.random.rand(*base_shape) * 2 - 1
    scale = np.random.rand(*base_shape) + 0.5

    def model():
        fn = dist.TransformedDistribution(
            dist.Normal(jnp.zeros_like(loc), jnp.ones_like(scale)),
            [AffineTransform(loc, scale), ExpTransform()],
        ).expand(shape)
        if event_shape:
            fn = fn.to_event(len(event_shape)).expand_by([100000])
        with numpyro.plate_stack("plates", batch_shape):
            with numpyro.plate("particles", 100000):
                return numpyro.sample("x", fn)

    with handlers.trace() as tr:
        value = handlers.seed(model, 0)()
    expected_moments = get_moments(jnp.log(value))

    with numpyro.handlers.reparam(config={"x": TransformReparam()}):
        with handlers.trace() as tr:
            value = handlers.seed(model, 0)()
    assert tr["x"]["type"] == "deterministic"
    actual_moments = get_moments(jnp.log(value))
    assert_allclose(actual_moments, expected_moments, atol=0.05, rtol=0.01)


def neals_funnel(dim):
    y = numpyro.sample("y", dist.Normal(0, 3))
    with numpyro.plate("D", dim):
        numpyro.sample("x", dist.Normal(0, jnp.exp(y / 2)))


def dirichlet_categorical(data):
    concentration = jnp.array([1.0, 1.0, 1.0])
    p_latent = numpyro.sample("p", dist.Dirichlet(concentration))
    with numpyro.plate("N", data.shape[0] if data is not None else 10):
        numpyro.sample("obs", dist.Categorical(p_latent), obs=data)
    return p_latent


def test_neals_funnel_smoke():
    dim = 10

    guide = AutoIAFNormal(neals_funnel)
    svi = SVI(neals_funnel, guide, Adam(1e-10), Trace_ELBO())
    svi_state = svi.init(random.key(0), dim)

    def body_fn(i, val):
        svi_state, loss = svi.update(val, dim)
        return svi_state

    svi_state = lax.fori_loop(0, 1000, body_fn, svi_state)
    params = svi.get_params(svi_state)

    neutra = NeuTraReparam(guide, params)
    model = neutra.reparam(neals_funnel)
    nuts = NUTS(model)
    mcmc = MCMC(nuts, num_warmup=50, num_samples=50)
    mcmc.run(random.key(1), dim)
    samples = mcmc.get_samples()
    transformed_samples = neutra.transform_sample(samples["auto_shared_latent"])
    assert "x" in transformed_samples
    assert "y" in transformed_samples


@pytest.mark.parametrize(
    "model, kwargs",
    [
        (neals_funnel, {"dim": 10}),
        (dirichlet_categorical, {"data": np.ones(10, dtype=np.int32)}),
    ],
)
def test_reparam_log_joint(model, kwargs):
    guide = AutoIAFNormal(model)
    svi = SVI(model, guide, Adam(1e-10), Trace_ELBO(), **kwargs)
    svi_state = svi.init(random.key(0))
    params = svi.get_params(svi_state)
    neutra = NeuTraReparam(guide, params)
    reparam_model = neutra.reparam(model)
    _, pe_fn, _, _ = initialize_model(random.key(1), model, model_kwargs=kwargs)
    init_params, pe_fn_neutra, _, _ = initialize_model(
        random.key(2), reparam_model, model_kwargs=kwargs
    )
    latent_x = list(init_params[0].values())[0]
    pe_transformed = pe_fn_neutra(init_params[0])
    latent_y = neutra.transform(latent_x)
    log_det_jacobian = neutra.transform.log_abs_det_jacobian(latent_x, latent_y)
    pe = pe_fn(guide._unpack_latent(latent_y))
    assert_allclose(pe_transformed, pe - log_det_jacobian, rtol=2e-7)


def test_neutra_reparam_unobserved_model():
    model = dirichlet_categorical
    data = jnp.ones(10, dtype=jnp.int32)
    guide = AutoIAFNormal(model)
    svi = SVI(model, guide, Adam(1e-3), Trace_ELBO())
    svi_state = svi.init(random.key(0), data)
    params = svi.get_params(svi_state)
    neutra = NeuTraReparam(guide, params)
    reparam_model = neutra.reparam(model)
    with handlers.seed(rng_seed=0):
        reparam_model(data=None)


def test_neutra_reparam_with_plate():
    def model():
        with numpyro.plate("N", 3, dim=-1):
            x = numpyro.sample("x", dist.Normal(0, 1))
        assert x.shape == (3,)

    guide = AutoDiagonalNormal(model)
    svi = SVI(model, guide, Adam(1e-3), Trace_ELBO())
    svi_state = svi.init(random.key(0))
    params = svi.get_params(svi_state)
    neutra = NeuTraReparam(guide, params)
    reparam_model = neutra.reparam(model)
    with handlers.seed(rng_seed=0):
        reparam_model()


@pytest.mark.parametrize("shape", [(), (4,), (3, 2)], ids=str)
@pytest.mark.parametrize("centered", [0.0, 0.6, 1.0, None])
@pytest.mark.parametrize("dist_type", ["Normal", "StudentT"])
@pytest.mark.parametrize("event_dim", [0, 1])
def test_loc_scale(dist_type, centered, shape, event_dim):
    loc = np.random.uniform(-1.0, 1.0, shape)
    scale = np.random.uniform(0.5, 1.5, shape)
    event_dim = min(event_dim, len(shape))
    if event_dim == 1 and centered is not None:
        centered = jnp.broadcast_to(centered, shape[-event_dim:])

    def model(loc, scale):
        with numpyro.plate_stack("plates", shape[: len(shape) - event_dim]):
            with numpyro.plate("particles", 10000):
                if "dist_type" == "Normal":
                    numpyro.sample("x", dist.Normal(loc, scale).to_event(event_dim))
                else:
                    numpyro.sample(
                        "x", dist.StudentT(10.0, loc, scale).to_event(event_dim)
                    )

    def get_expected_probe(loc, scale):
        with numpyro.handlers.trace() as trace:
            with numpyro.handlers.seed(rng_seed=0):
                model(loc, scale)
        return get_moments(trace["x"]["value"])

    if "dist_type" == "Normal":
        reparam = LocScaleReparam(centered)
    else:
        reparam = LocScaleReparam(centered, shape_params=["df"])

    def get_actual_probe(loc, scale):
        with numpyro.handlers.trace() as trace:
            with numpyro.handlers.seed(rng_seed=0):
                with numpyro.handlers.reparam(config={"x": reparam}):
                    model(loc, scale)
        return get_moments(trace["x"]["value"])

    expected_probe = get_expected_probe(loc, scale)
    actual_probe = get_actual_probe(loc, scale)
    assert_allclose(actual_probe, expected_probe, atol=0.1)

    expected_grad = jacobian(get_expected_probe, argnums=(0, 1))(loc, scale)
    actual_grad = jacobian(get_actual_probe, argnums=(0, 1))(loc, scale)
    assert_allclose(actual_grad[0], expected_grad[0], atol=0.05)  # loc grad
    assert_allclose(actual_grad[1], expected_grad[1], atol=0.05)  # scale grad


@pytest.mark.parametrize("centered", [-0.3, 10.0, np.array([0.1, -2.0])])
def test_loc_scale_centered_invalid(centered):
    N = 10
    loc = np.random.uniform(-1.0, 1.0, size=(N, 2))
    scale = np.random.uniform(0.5, 1.5, size=(N, 2))

    def model(loc, scale):
        with numpyro.plate("particles", N):
            numpyro.sample("x", dist.Normal(loc, scale).to_event(1))

    with pytest.raises(ValueError, match=".*does not satisfy.*"):
        with handlers.reparam(config=LocScaleReparam(centered)):
            with handlers.seed(rng_seed=10):
                model(loc, scale)


@pytest.mark.parametrize("shape", [(), (4,), (3, 2)], ids=str)
@pytest.mark.parametrize("dim", [2, 3, 4])
def test_projected_normal(shape, dim):
    def model(concentration):
        with numpyro.plate_stack("plates", shape):
            with numpyro.plate("particles", 10000):
                numpyro.sample("x", dist.ProjectedNormal(concentration))

    def get_expected_probe(concentration):
        with numpyro.handlers.trace() as trace:
            with numpyro.handlers.seed(rng_seed=0):
                model(concentration)
        return get_moments(trace["x"]["value"])

    def get_actual_probe(concentration):
        with numpyro.handlers.trace() as trace:
            with numpyro.handlers.seed(rng_seed=0):
                reparam = ProjectedNormalReparam()
                with numpyro.handlers.reparam(config={"x": reparam}):
                    model(concentration)
        return get_moments(trace["x"]["value"])

    concentration = np.random.normal(shape + (dim,))
    expected_probe = get_expected_probe(concentration)
    actual_probe = get_actual_probe(concentration)
    assert_allclose(actual_probe, expected_probe, atol=0.1)

    expected_grad = jacobian(get_expected_probe)(concentration)
    actual_grad = jacobian(get_actual_probe)(concentration)
    assert_allclose(actual_grad, expected_grad, atol=0.05)


@pytest.mark.parametrize("shape", [(), (4,), (3, 2)], ids=str)
def test_circular(shape):
    # Define two models which should return the same distributions
    # This model is the expected distribution
    def model_exp(loc, concentration):
        with numpyro.plate_stack("plates", shape):
            with numpyro.plate("particles", 10000):
                numpyro.sample("x", dist.VonMises(loc, concentration))

    # This model is for inference
    reparam = CircularReparam()

    @numpyro.handlers.reparam(config={"x": reparam})
    def model_act(loc, concentration):
        numpyro.sample("x", dist.VonMises(loc, concentration))

    def get_expected_probe(loc, concentration):
        with numpyro.handlers.trace() as trace:
            with numpyro.handlers.seed(rng_seed=0):
                model_exp(loc, concentration)
        return get_circular_moments(trace["x"]["value"])

    def get_actual_probe(loc, concentration):
        kernel = NUTS(model_act, dense_mass=True)
        mcmc = MCMC(kernel, num_warmup=1000, num_samples=10000, num_chains=1)
        mcmc.run(random.key(0), loc, concentration)
        samples = mcmc.get_samples()
        return get_circular_moments(samples["x"])

    loc = np.random.uniform(-np.pi, np.pi, shape)
    concentration = np.random.lognormal(1.0, 1.0, shape)
    expected_probe = get_expected_probe(loc, concentration)
    actual_probe = get_actual_probe(loc, concentration)

    assert_allclose(actual_probe, expected_probe, atol=0.1)


def test_circular_reparam_no_observe():
    def model():
        numpyro.sample("x", dist.VonMises(0, 1), obs=0.5)

    with numpyro.handlers.seed(rng_seed=0):
        with numpyro.handlers.reparam(config={"x": CircularReparam()}):
            with pytest.raises(AssertionError, match="not support observe"):
                model()


_unconstrain_reparam = numpyro.infer.util._unconstrain_reparam


def test_unconstrained_reparam_subsample():
    def model():
        with numpyro.plate("N", 10, 2):
            numpyro.sample("x", dist.HalfNormal(1))

    params = {"x": 3.0}
    substituted_model = handlers.substitute(
        handlers.seed(model, rng_seed=0),
        substitute_fn=partial(_unconstrain_reparam, params),
    )
    log_det_site = handlers.trace(substituted_model).get_trace()["_x_log_det"]
    assert log_det_site["scale"] == 5.0


def test_unconstrained_reparam_scale():
    def model():
        with handlers.scale(scale=5.0):
            numpyro.sample("x", dist.HalfNormal(1))

    params = {"x": 3.0}
    substituted_model = handlers.substitute(
        handlers.seed(model, rng_seed=0),
        substitute_fn=partial(_unconstrain_reparam, params),
    )
    log_det_site = handlers.trace(substituted_model).get_trace()["_x_log_det"]
    assert log_det_site["scale"] == 5.0


def test_loc_scale_reparam_raise_for_log_normal():
    def model():
        numpyro.sample("x", dist.LogNormal(0, 1))

    reparam_model = handlers.reparam(model, config={"x": LocScaleReparam(0)})
    with pytest.raises(ValueError, match="LocScaleReparam.*"):
        handlers.seed(reparam_model, rng_seed=0)()


def test_explicit_reparam():
    def model():
        numpyro.sample("x", dist.Gamma(4, 4))

    reparam = ExplicitReparam(dist.transforms.SoftplusTransform().inv)
    reparametrized = handlers.reparam(model, {"x": reparam})
    kernel = NUTS(model=reparametrized)
    mcmc = MCMC(kernel, num_warmup=1000, num_samples=1000, num_chains=1)
    mcmc.run(random.key(2))
    samples = mcmc.get_samples()
    assert abs(samples["x"].mean() - 1) < 0.1


# (reparam class, kwargs, auxiliary-site suffix) for the time reparameterizers.
TIME_REPARAMS = [
    (HaarReparam, {}, "haar"),
    (HaarReparam, {"flip": True}, "haar"),
    (DiscreteCosineReparam, {}, "dct"),
    (DiscreteCosineReparam, {"smooth": 1.0}, "dct"),
]


@pytest.mark.parametrize("T", [6, 7], ids=["even", "odd"])
@pytest.mark.parametrize("reparam_cls, kwargs, suffix", TIME_REPARAMS)
def test_time_reparam_moments(reparam_cls, kwargs, suffix, T):
    loc = np.linspace(-1.0, 1.0, T)
    scale = np.linspace(0.5, 1.5, T)

    def model():
        with numpyro.plate("particles", 100000):
            return numpyro.sample("x", dist.Normal(loc, scale).to_event(1))

    value = handlers.seed(model, 0)()
    expected_moments = get_moments(value)

    reparam = reparam_cls(**kwargs)
    with numpyro.handlers.reparam(config={"x": reparam}):
        with handlers.trace() as tr:
            value = handlers.seed(model, 0)()
    # The original site becomes deterministic and an auxiliary site is added.
    assert tr["x"]["type"] == "deterministic"
    assert tr[f"x_{suffix}"]["type"] == "sample"
    actual_moments = get_moments(value)
    assert_allclose(actual_moments, expected_moments, atol=0.05, rtol=0.05)


@pytest.mark.parametrize("reparam_cls, kwargs, suffix", TIME_REPARAMS)
def test_time_reparam_log_joint(reparam_cls, kwargs, suffix):
    T = 8

    def model():
        numpyro.sample("x", dist.Normal(0.0, 1.0).expand([T]).to_event(1))

    reparam = reparam_cls(**kwargs)
    reparam_model = handlers.reparam(model, config={"x": reparam})
    aux = random.normal(random.key(1), (T,))

    # Log density of the reparameterized model at the auxiliary value.
    log_prob_reparam, _ = log_density(reparam_model, (), {}, {f"x_{suffix}": aux})

    # Recover the latent value and evaluate the original model density.
    with handlers.trace() as tr, handlers.substitute(data={f"x_{suffix}": aux}):
        handlers.seed(reparam_model, 0)()
    x_value = tr["x"]["value"]
    log_prob_original, _ = log_density(model, (), {}, {"x": x_value})

    # The transform has unit Jacobian, so the log density is unchanged.
    assert_allclose(log_prob_reparam, log_prob_original, atol=1e-4)


@pytest.mark.parametrize("reparam_cls", [HaarReparam, DiscreteCosineReparam])
def test_time_reparam_nuts_smoke(reparam_cls):
    T = 10
    data = np.linspace(-1.0, 1.0, T)

    def model(data):
        x = numpyro.sample("x", dist.Normal(0.0, 1.0).expand([T]).to_event(1))
        numpyro.sample("obs", dist.Normal(x, 0.5).to_event(1), obs=data)

    reparam_model = handlers.reparam(model, config={"x": reparam_cls()})
    mcmc = MCMC(
        NUTS(reparam_model),
        num_warmup=200,
        num_samples=200,
        num_chains=1,
        progress_bar=False,
    )
    mcmc.run(random.key(0), data=data)
    samples = mcmc.get_samples()["x"]
    assert samples.shape == (200, T)
    # Conjugate posterior mean is data * (1 / 0.25) / (1 + 1 / 0.25) = 0.8 * data.
    assert_allclose(samples.mean(0), 0.8 * data, atol=0.3)


def test_time_reparam_does_not_support_obs():
    def model():
        numpyro.sample(
            "x", dist.Normal(0.0, 1.0).expand([4]).to_event(1), obs=jnp.zeros(4)
        )

    reparam_model = handlers.reparam(model, config={"x": HaarReparam()})
    with pytest.raises(AssertionError):
        handlers.seed(reparam_model, 0)()


@pytest.mark.parametrize("reparam_cls", [HaarReparam, DiscreteCosineReparam])
def test_time_reparam_event_dim_2(reparam_cls):
    # Reparameterize along the time axis (dim=-2) of a (T, K) latent.
    T, K = 8, 3
    data = np.random.RandomState(0).randn(T, K)

    def model(data):
        x = numpyro.sample("x", dist.Normal(0.0, 1.0).expand([T, K]).to_event(2))
        numpyro.sample("obs", dist.Normal(x, 0.5).to_event(2), obs=data)

    reparam_model = handlers.reparam(model, config={"x": reparam_cls(dim=-2)})
    mcmc = MCMC(
        NUTS(reparam_model),
        num_warmup=200,
        num_samples=200,
        num_chains=1,
        progress_bar=False,
    )
    mcmc.run(random.key(0), data=data)
    samples = mcmc.get_samples()["x"]
    assert samples.shape == (200, T, K)
    assert_allclose(samples.mean(0), 0.8 * data, atol=0.4)


@pytest.mark.parametrize("reparam_cls, kwargs, suffix", TIME_REPARAMS)
def test_time_reparam_positive_support(reparam_cls, kwargs, suffix):
    # A positive-support latent is reparameterized in the unconstrained (log)
    # space via biject_to(support).inv; the recovered value stays positive and
    # the posterior matches the un-reparameterized model.
    T = 8
    data = np.abs(np.random.RandomState(1).randn(T)) + 0.5

    def model(data):
        x = numpyro.sample("x", dist.LogNormal(0.0, 1.0).expand([T]).to_event(1))
        numpyro.sample("obs", dist.Normal(x, 0.3).to_event(1), obs=data)

    # The auxiliary site lives in unconstrained space and x stays positive.
    reparam = reparam_cls(**kwargs)
    with handlers.trace() as tr:
        handlers.seed(handlers.reparam(model, config={"x": reparam}), 0)(data)
    assert tr["x"]["type"] == "deterministic"
    assert (tr["x"]["value"] > 0).all()
    assert tr[f"x_{suffix}"]["fn"].support == constraints.independent(
        constraints.real, 1
    )

    def run(config):
        m = handlers.reparam(model, config=config) if config else model
        mcmc = MCMC(
            NUTS(m), num_warmup=400, num_samples=400, num_chains=1, progress_bar=False
        )
        mcmc.run(random.key(0), data=data)
        return mcmc.get_samples()["x"].mean(0)

    assert_allclose(run({"x": reparam}), run({}), atol=0.15)


def test_loc_scale_traced_centered():
    @jit
    def f(centered):
        reparam = LocScaleReparam(centered)
        return reparam.centered

    assert f(jnp.array(0.5)) == 0.5


# init_to_value requests on the original name of a reparameterized site (#1614).
def _init_model_locscale():
    mu = numpyro.sample("mu", dist.Normal(1.0, 2.0))
    with numpyro.plate("p", 3):
        numpyro.sample("x", dist.Normal(mu, 3.0))


def _init_model_circular():
    phi = numpyro.sample("phi", dist.VonMises(0.5, 2.0))
    numpyro.sample("obs", dist.Normal(phi, 1.0), obs=0.3)


def _init_model_transform():
    numpyro.sample(
        "x",
        dist.TransformedDistribution(
            dist.Normal(jnp.zeros(3), 1.0).to_event(1),
            [AffineTransform(1.0, 2.0), ExpTransform()],
        ),
    )


def _init_model_projected():
    numpyro.sample("x", dist.ProjectedNormal(jnp.array([1.0, 2.0, 0.5])))


def _init_model_vector():
    numpyro.sample("x", dist.Normal(jnp.zeros(4), 1.0).to_event(1))


def _init_model_gamma():
    numpyro.sample("x", dist.Gamma(4.0, 4.0))


INIT_CASES = [
    (_init_model_locscale, LocScaleReparam(0.0), np.array([0.1, -2.0, 5.0])),
    (_init_model_locscale, LocScaleReparam(0.4), np.array([0.1, -2.0, 5.0])),
    (_init_model_circular, CircularReparam(), np.array(0.5)),
    (_init_model_transform, TransformReparam(), np.array([0.5, 1.0, 4.0])),
    (
        _init_model_projected,
        ProjectedNormalReparam(),
        np.array([0.6, 0.0, 0.8]),
    ),
    (_init_model_vector, HaarReparam(), np.array([0.1, -0.2, 0.3, 0.4])),
    (_init_model_vector, DiscreteCosineReparam(), np.array([0.1, -0.2, 0.3, 0.4])),
    (
        _init_model_gamma,
        ExplicitReparam(dist.transforms.SoftplusTransform().inv),
        np.array(1.25),
    ),
]


@pytest.mark.parametrize("model, reparam, value", INIT_CASES)
def test_init_to_value_original_name(model, reparam, value):
    name = "phi" if model is _init_model_circular else "x"
    reparam_model = handlers.reparam(model, config={name: reparam})
    info = initialize_model(
        random.PRNGKey(0),
        reparam_model,
        init_strategy=init_to_value(values={name: value}),
    )
    actual = info.postprocess_fn(info.param_info.z)[name]
    assert_allclose(actual, value, rtol=1e-5, atol=1e-5)
    assert_allclose(info.model_trace[name]["value"], value, rtol=1e-5, atol=1e-5)


def test_init_to_value_original_name_issue_1614():
    def model():
        angle = numpyro.sample("loc_1", dist.VonMises(0, 100))
        with numpyro.plate("plate", 3):
            numpyro.sample("y", dist.VonMises(angle, 2), obs=jnp.arange(1, 4))

    reparam_model = handlers.reparam(model, config={"loc_1": CircularReparam()})
    kernel = NUTS(reparam_model, init_strategy=init_to_value(values={"loc_1": 0.5}))
    for chain_method, num_chains in [("sequential", 1), ("vectorized", 2)]:
        mcmc = MCMC(
            kernel,
            num_warmup=0,
            num_samples=1,
            num_chains=num_chains,
            chain_method=chain_method,
            progress_bar=False,
        )
        mcmc.warmup(random.PRNGKey(1))
        z = mcmc.post_warmup_state.z
        if num_chains > 1:
            z = {k: v[0] for k, v in z.items()}
        assert_allclose(kernel.postprocess_fn((), {})(z)["loc_1"], 0.5, atol=1e-5)


def test_init_to_value_original_name_untouched():
    reparam_model = handlers.reparam(
        _init_model_locscale, config={"x": LocScaleReparam(0.0)}
    )

    def init(strategy):
        info = initialize_model(
            random.PRNGKey(0), reparam_model, init_strategy=strategy
        )
        return info.param_info.z

    class Values(dict):
        pass

    expected = init(init_to_value(values={"other": 0.0}))
    value = jnp.array([0.1, -2.0, 5.0])
    # out of scope: identical to a request that names no site
    for strategy in [
        init_to_value(values=Values(x=value)),
        lambda site=None: init_to_value(site, values={"x": value}),
    ]:
        actual = init(strategy)
        for k in expected:
            assert_allclose(actual[k], expected[k], rtol=0, atol=0)
    # an explicit auxiliary request wins over the original
    aux = jnp.array([1.0, 2.0, 3.0])
    z = init(init_to_value(values={"x": value, "x_decentered": aux}))
    assert_allclose(z["x_decentered"], aux)


@pytest.mark.parametrize(
    "model, reparam, value, error",
    [
        (
            lambda: numpyro.sample(
                "x",
                dist.TransformedDistribution(
                    dist.Uniform(-4.0, -1.0), dist.transforms.AbsTransform()
                ),
            ),
            TransformReparam(),
            2.5,
            NotImplementedError,
        ),
        (_init_model_locscale, LocScaleReparam(0.0), 1.0, ValueError),
        (_init_model_circular, CircularReparam(), 4.0, ValueError),
        (_init_model_projected, ProjectedNormalReparam(), np.ones(3), ValueError),
    ],
)
def test_init_to_value_original_name_raises(model, reparam, value, error):
    name = "phi" if model is _init_model_circular else "x"
    reparam_model = handlers.reparam(model, config={name: reparam})
    with pytest.raises(error, match=f"'{name}'"):
        initialize_model(
            random.PRNGKey(0),
            reparam_model,
            init_strategy=init_to_value(values={name: value}),
        )


def test_init_to_value_original_name_jit_raises():
    reparam_model = handlers.reparam(
        _init_model_locscale, config={"x": LocScaleReparam(0.0)}
    )

    @jit
    def f(key):
        info = initialize_model(
            key,
            reparam_model,
            init_strategy=init_to_value(values={"x": jnp.zeros(3)}),
        )
        return info.param_info.z

    with pytest.raises(NotImplementedError, match="'x'"):
        f(random.PRNGKey(0))


def test_init_to_value_original_name_autoguide_unchanged():
    # AutoGuides are out of scope: the original-name request is ignored as before.
    reparam_model = handlers.reparam(
        _init_model_locscale, config={"x": LocScaleReparam(0.0)}
    )
    value = jnp.array([0.1, -2.0, 5.0])
    guide = AutoDiagonalNormal(
        reparam_model, init_loc_fn=init_to_value(values={"x": value})
    )
    ref = AutoDiagonalNormal(
        reparam_model, init_loc_fn=init_to_value(values={"other": 0.0})
    )
    svi = SVI(reparam_model, guide, Adam(0.1), Trace_ELBO())
    svi_ref = SVI(reparam_model, ref, Adam(0.1), Trace_ELBO())
    loc = svi.get_params(svi.init(random.PRNGKey(0)))["auto_loc"]
    loc_ref = svi_ref.get_params(svi_ref.init(random.PRNGKey(0)))["auto_loc"]
    assert_allclose(loc, loc_ref, rtol=0, atol=0)


def test_init_to_value_original_name_hidden_raises():
    # the original site is hidden from initialization by `block`: refuse loudly
    reparam_model = handlers.block(
        handlers.reparam(_init_model_locscale, config={"x": LocScaleReparam(0.0)}),
        hide=["x"],
    )
    with pytest.raises(NotImplementedError, match="'x'"):
        initialize_model(
            random.PRNGKey(0),
            reparam_model,
            init_strategy=init_to_value(values={"x": np.zeros(3)}),
        )


def test_init_to_value_original_name_circular_pi():
    # +pi is in the closed support of VonMises but never produced by CircularReparam;
    # the site starts within tolerance below it
    reparam_model = handlers.reparam(
        _init_model_circular, config={"phi": CircularReparam()}
    )
    info = initialize_model(
        random.PRNGKey(0),
        reparam_model,
        init_strategy=init_to_value(values={"phi": np.pi}),
    )
    actual = info.postprocess_fn(info.param_info.z)["phi"]
    assert_allclose(actual, np.pi, rtol=1e-4)


def test_init_to_value_original_name_scope_prefix():
    # a scope (with a non-str prefix) between the reparam and the initialization
    reparam_model = handlers.scope(
        handlers.reparam(_init_model_locscale, config={"x": LocScaleReparam(0.0)}),
        prefix=0,
    )
    value = np.array([0.1, -2.0, 5.0])
    info = initialize_model(
        random.PRNGKey(0),
        reparam_model,
        init_strategy=init_to_value(values={"0/x": value}),
    )
    assert_allclose(info.postprocess_fn(info.param_info.z)["0/x"], value, rtol=1e-5)


def test_init_to_value_original_name_gibbs():
    from numpyro.infer import Gibbs

    def model():
        mu = numpyro.sample("mu", dist.Normal(0.0, 1.0))
        numpyro.sample("x", dist.Normal(mu, 2.0))

    reparam_model = handlers.reparam(model, config={"x": LocScaleReparam(0.0)})

    def run(first, second):
        kernel = Gibbs(
            [
                (
                    NUTS(reparam_model, init_strategy=init_to_value(values=first)),
                    ["mu"],
                ),
                (NUTS(reparam_model, init_strategy=init_to_value(values=second)), None),
            ]
        )
        mcmc = MCMC(kernel, num_warmup=0, num_samples=1, progress_bar=False)
        mcmc.warmup(random.PRNGKey(0))
        return mcmc.post_warmup_state.z

    # the last block is initialized against the final values of the other blocks
    z = run({"mu": 3.0}, {"x": 1.5})
    assert_allclose(z["mu"] + 2.0 * z["x_decentered"], 1.5, rtol=1e-5)
    # an earlier block could be invalidated by later blocks: refuse
    kernel = Gibbs(
        [
            (NUTS(reparam_model, init_strategy=init_to_value(values={"x": 1.5})), None),
            (
                NUTS(reparam_model, init_strategy=init_to_value(values={"mu": 3.0})),
                ["mu"],
            ),
        ]
    )
    mcmc = MCMC(kernel, num_warmup=0, num_samples=1, progress_bar=False)
    with pytest.raises(NotImplementedError, match="'x'"):
        mcmc.warmup(random.PRNGKey(0))


def test_init_to_value_original_name_sa_raises():
    from numpyro.infer import SA

    def model():
        with handlers.reparam(config={"x": LocScaleReparam(0.0)}):
            numpyro.sample("x", dist.Normal(3.0, 2.0))

    kernel = SA(model, init_strategy=init_to_value(values={"x": 7.0}))
    with pytest.raises(NotImplementedError, match="'x'"):
        kernel.init(random.PRNGKey(0), 0, None, (), {})
    # explicit auxiliary request still works identically to master
    aux_kernel = SA(model, init_strategy=init_to_value(values={"x_decentered": 2.0}))
    state = aux_kernel.init(random.PRNGKey(0), 0, None, (), {})
    assert jnp.isfinite(state.z["x_decentered"])


def test_init_to_value_original_name_replay_mismatch_raises():
    # SigmoidTransform clips expit at 1 - finfo.eps, so 1 - eps/2 cannot be reproduced on replay.
    scale = 2.0 ** (49 if jax.config.x64_enabled else 20)

    def near_end():
        with handlers.reparam(config={"x": TransformReparam()}):
            numpyro.sample(
                "x",
                dist.TransformedDistribution(
                    dist.Uniform(0.0, 1.0),
                    AffineTransform(-(scale - 0.0625), scale),
                ),
            )

    with pytest.raises(ValueError, match=r"'x'.*coordinate \(\).*0\.0.*-0\.0625"):
        initialize_model(
            random.PRNGKey(0), near_end, init_strategy=init_to_value(values={"x": 0.0})
        )

    # Interior 31-coordinate vector whose unconstrained round trip rounds past tolerance.
    centers = jnp.arange(1, 32, dtype=jnp.float32) / 32

    def bounded():
        with handlers.reparam(config={"x": TransformReparam()}):
            numpyro.sample(
                "x",
                dist.TransformedDistribution(
                    dist.Uniform(jnp.zeros_like(centers), jnp.ones_like(centers)),
                    AffineTransform(-(2**20) * centers, float(2**20)),
                ).to_event(1),
            )

    with pytest.raises(ValueError, match=r"'x'.*coordinates differ.*\(0,\)|\(2,\)"):
        initialize_model(
            random.PRNGKey(0),
            bounded,
            init_strategy=init_to_value(values={"x": jnp.zeros(31)}),
        )


def test_init_to_value_original_name_complex_raises():
    import warnings

    def model():
        with handlers.reparam(config={"x": LocScaleReparam(0.0)}):
            numpyro.sample("x", dist.Normal(3.0, 2.0))

    with warnings.catch_warnings(record=True) as ws:
        warnings.simplefilter("always")
        with pytest.raises(ValueError, match="'x'"):
            initialize_model(
                random.PRNGKey(0),
                model,
                init_strategy=init_to_value(values={"x": 7.0 + 12.0j}),
            )
    assert len(ws) == 0


def test_init_to_value_original_name_suffix_format_once():
    from numpyro.distributions.transforms import IdentityTransform
    from numpyro.infer.reparam import UnitJacobianReparam

    calls = []

    class Suffix(str):
        def __format__(self, spec):
            calls.append(spec)
            return super().__format__(spec)

    suffix = Suffix("aux")

    def model():
        numpyro.sample("other", dist.Normal(0.0, 1.0))
        with handlers.reparam(
            config={"x": UnitJacobianReparam(IdentityTransform(), suffix)}
        ):
            numpyro.sample("x", dist.Normal(0.0, 1.0))

    initialize_model(
        random.PRNGKey(0), model, init_strategy=init_to_value(values={"other": 0.75})
    )
    unreq_calls = len(calls)
    calls.clear()
    initialize_model(random.PRNGKey(0), model, init_strategy=init_to_value(values={}))
    assert unreq_calls == len(calls)


def test_init_to_value_original_name_barker_and_ensemble():
    from numpyro.infer import AIES, ESS, BarkerMH

    def model():
        mu = numpyro.sample("mu", dist.Normal(1.0, 1.0))
        with handlers.reparam(config={"x": LocScaleReparam(0.0)}):
            numpyro.sample("x", dist.Normal(mu, 2.0))

    strategy = init_to_value(values={"x": 7.0})
    barker = BarkerMH(model, init_strategy=strategy)
    bstate = barker.init(random.PRNGKey(0), 0, None, (), {})
    assert_allclose(barker.postprocess_fn((), {})(bstate.z)["x"], 7.0, rtol=1e-5)

    for cls in (AIES, ESS):
        ens = cls(model, init_strategy=strategy)
        estate = ens.init(random.split(random.PRNGKey(0), 4), 0, None, (), {})
        got = jax.vmap(ens.postprocess_fn((), {}))(estate.z)["x"]
        assert_allclose(got, jnp.full((4,), 7.0), rtol=1e-5)
