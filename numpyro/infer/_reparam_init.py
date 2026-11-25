# Copyright Contributors to the Pyro project.
# SPDX-License-Identifier: Apache-2.0

"""
Private support for honouring ``init_to_value`` requests made on the *original* name of a
site rewritten by :class:`~numpyro.handlers.reparam` (issue #1614).

Scope (anything else behaves exactly as before):

* only :func:`~numpyro.infer.util.initialize_model` (and therefore the MCMC kernels built on
  it), never AutoGuides;
* only an exact ``functools.partial`` of :func:`~numpyro.infer.init_to_value` whose
  ``values`` is a non-empty plain ``dict`` with ``str`` keys;
* only sites whose reparametrizer has an exact inverse (see :func:`set_inverse` callers in
  :mod:`numpyro.infer.reparam`), under a handler stack made only of the handlers listed in
  ``_ALLOWED_HANDLERS``.

A request on a reparameterized site outside that scope raises a ``NotImplementedError`` or
``ValueError`` naming the site instead of being silently ignored.

Whether a site is requested is decided from handler types, object identity and lookups of
exact-``str`` names in the plain ``dict`` only: no user callback, truth test, comparison
or attribute read on a user object is added on any path where nothing is requested.
"""

from contextlib import contextmanager
from functools import partial

import numpy as np

import jax
from jax import device_get
import jax.numpy as jnp

from numpyro import handlers
from numpyro.distributions import transforms as T
from numpyro.infer.initialization import init_to_value
from numpyro.primitives import _PYRO_STACK, plate
from numpyro.util import is_prng_key, not_jax_tracer

# >0 while an AutoGuide sets up its prototype (AutoGuides are out of scope).
_SUPPRESSED = [0]

_ALLOWED_HANDLERS = (
    handlers.condition,
    handlers.infer_config,
    handlers.mask,
    handlers.reparam,
    handlers.scale,
    handlers.scope,
    handlers.seed,
    handlers.substitute,
    handlers.trace,
    plate,
)

_BIJECTIVE_TRANSFORMS = (
    T.AffineTransform,
    T.CorrCholeskyTransform,
    T.DiscreteCosineTransform,
    T.ExpTransform,
    T.HaarTransform,
    T.IdentityTransform,
    T.LowerCholeskyAffine,
    T.LowerCholeskyTransform,
    T.OrderedTransform,
    T.PermuteTransform,
    T.PowerTransform,
    T.SigmoidTransform,
    T.SoftplusTransform,
    T.StickBreakingTransform,
)

# float32: 2**-13, float64: 2**-42 (1024 units in the last place of max(|x|, 1)).
_ULPS = 1024


class _Unsupported(Exception):
    pass


def _is_one_of(cls, classes):
    """``cls in classes`` by identity only (never calls a metaclass ``__eq__``)."""
    for c in classes:
        if cls is c:
            return True
    return False


@contextmanager
def suppressed():
    """Disable the translation (used by AutoGuides around ``initialize_model``)."""
    _SUPPRESSED[0] += 1
    try:
        yield
    finally:
        _SUPPRESSED[0] -= 1


# Non-empty while a kernel is initialized in a position where a translated value could
# be invalidated afterwards (a `Gibbs` block followed by other blocks); requests on
# reparameterized sites then raise instead of being translated.
_REFUSING = []


@contextmanager
def refusing(reason):
    """Refuse (raise) requests on reparameterized sites, giving ``reason``."""
    _REFUSING.append(reason)
    try:
        yield
    finally:
        _REFUSING.pop()


def check_bijective(t):
    """Raise ``_Unsupported`` unless ``t`` is built only from allowlisted bijections."""
    tt = type(t)
    if tt is T.ComposeTransform:
        for part in t.parts:
            check_bijective(part)
    elif tt is T._InverseTransform:
        check_bijective(t._inv)
    elif tt is T.IndependentTransform:
        check_bijective(t.base_transform)
    elif not _is_one_of(tt, _BIJECTIVE_TRANSFORMS):
        raise _Unsupported(
            f"its transform {tt.__name__} is not one of the bijective transforms "
            "supported for initialization"
        )


def _inverse_chain(transforms, x):
    for t in transforms:
        check_bijective(t)
    for t in reversed(transforms):
        x = t.inv(x)
    return x


def _forward_chain(transforms, x):
    for t in transforms:
        x = t(x)
    return x


def set_inverse(reparam, name, aux_name, inverse, forward):
    """
    Called by the supported reparametrizers right before they sample their auxiliary site.

    :param reparam: the reparametrizer instance.
    :param str name: the ``name`` argument it was called with.
    :param str aux_name: local name of the auxiliary site.
    :param inverse: maps a requested original value to the auxiliary value.
    :param forward: the reparametrizer's own auxiliary -> original computation.
    """
    recorder = handlers._REPARAM_INIT
    if recorder is None or not recorder.stack:
        return
    rec = recorder.stack[-1]
    if rec.reparam is not reparam or rec.name is not name:
        return
    if not _is_one_of(type(reparam), _supported_reparams()):
        return
    rec.hooked = True
    rec.aux_local = aux_name
    rec.inverse = inverse
    rec.forward = forward


def _supported_reparams():
    from numpyro.infer import reparam as R

    return (
        R.CircularReparam,
        R.DiscreteCosineReparam,
        R.ExplicitReparam,
        R.HaarReparam,
        R.LocScaleReparam,
        R.ProjectedNormalReparam,
        R.TransformReparam,
        R.UnitJacobianReparam,
    )


def _not_supported(name, reason):
    return NotImplementedError(
        f"init_to_value cannot initialize the reparameterized site '{name}' from the value"
        f" requested for it: {reason}. Remove '{name}' from `values`, or request the"
        " auxiliary site(s) created by the reparametrizer instead."
    )


def _bad_value(name, problem):
    return ValueError(
        f"init_to_value: the value requested for the reparameterized site '{name}'"
        f" {problem}."
    )


def _replay_mismatch(name, want, got, bad, tol, batched):
    coords = (
        [tuple(int(i) for i in idx) for idx in zip(*np.nonzero(bad))]
        if bad.ndim > 0
        else [()]
    )
    parts = []
    for idx in coords[:5]:
        if batched:
            loc = f"chain {idx[0]} coordinate {idx[1:]}"
        else:
            loc = f"coordinate {idx}"
        parts.append(f"{loc}: requested {float(want[idx])!r}, got {float(got[idx])!r}")
    summary = "; ".join(parts)
    if len(coords) > 5:
        rest = coords[5:32]
        more = f" (+ {len(coords) - 5} more: {rest}"
        if len(coords) > 32:
            more += ", ..."
        more += ")"
        summary = f"{len(coords)} coordinates differ: {summary}{more}"
    elif len(coords) > 1:
        summary = f"{len(coords)} coordinates differ: {summary}"
    return _bad_value(
        name,
        f"was not reproduced when replaying the initial state within {_ULPS} ulps"
        f" (tol={tol:.4g}): {summary}",
    )


# formatting these exact builtin types runs no user code
_FORMATTABLE = (str, int, float, bool, type(None))
_HIDE_CONTAINERS = (list, tuple, set, frozenset)


def _scoped_name(h, name):
    """Apply an exact ``scope`` handler to ``name`` as it would for a deterministic site."""
    prefix, divider, hide = h.prefix, h.divider, h.hide_types
    if (
        not _is_one_of(type(prefix), _FORMATTABLE)
        or not _is_one_of(type(divider), _FORMATTABLE)
        or not _is_one_of(type(hide), _HIDE_CONTAINERS)
        or any(type(t) is not str for t in hide)
    ):
        return None
    if name and "deterministic" not in hide:
        return f"{prefix}{divider}{name}"
    return name


def _outer_name(handler, name):
    """
    Name of the original site after every exact ``scope`` below its reparam handler
    (used only to recognise requests on originals hidden from initialization).
    """
    if type(name) is not str:
        return None
    below = None
    for i, h in enumerate(_PYRO_STACK):
        if h is handler:
            below = _PYRO_STACK[:i]
            break
    if below is None:
        return None
    for h in reversed(below):
        if type(h) is handlers.scope:
            name = _scoped_name(h, name)
            if name is None:
                return None
    return name


class _Record:
    __slots__ = (
        "msg",
        "handler",
        "reparam",
        "name",
        "hooked",
        "aux_local",
        "inverse",
        "forward",
        "aux_seen",
        "aux_explicit",
        "extra_latent",
        "nested",
        "reason",
        "translated",
        "request_name",
        "request",
        "value",
        "seen_at_init",
        "outer_name",
    )

    def __init__(self, handler, msg, reparam):
        self.msg = msg
        self.handler = handler
        self.reparam = reparam
        self.name = msg["name"]
        self.hooked = False
        self.aux_local = None
        self.inverse = None
        self.forward = None
        self.aux_seen = False
        self.aux_explicit = False
        self.extra_latent = False
        self.nested = False
        self.reason = None
        self.translated = False
        self.request_name = None
        self.request = None
        self.value = None
        self.seen_at_init = False
        self.outer_name = _outer_name(handler, self.name)

    def why_not(self):
        if self.nested:
            return "nested reparametrizers (a reparameterized auxiliary site) are not supported"
        if not self.hooked:
            return (
                f"its reparametrizer {type(self.reparam).__name__} has no exact inverse"
                " supported for initialization"
            )
        if self.reason is not None:
            return self.reason
        return "its auxiliary site was not visible to the initialization strategy"


class _State:
    """Per-``initialize_model`` call state (the user's strategy and its ``values``)."""

    def __init__(self, strategy, values):
        self.strategy = strategy
        # a snapshot: its keys are exact `str`, so lookups never run user code even if
        # the caller mutates its dict while the model runs
        self.values = dict(values)
        self.translated = False
        self.translated_requests = {}
        self.refuse = _REFUSING[-1] if _REFUSING else None

    def strategy_for(self):
        return partial(_init_to_value_reparam, state=self)

    @contextmanager
    def recording(self, w, prototype):
        recorder = _Recorder(self, w, prototype)
        prev = handlers._REPARAM_INIT
        handlers._REPARAM_INIT = recorder
        try:
            yield recorder
        finally:
            handlers._REPARAM_INIT = prev

    def verify_replay(
        self,
        init_params,
        is_valid,
        postprocess_fn,
        rng_key,
        dynamic_args,
        model_args,
        model_kwargs,
    ):
        """
        Safety net: replay the final unconstrained ``init_params`` through the model
        and verify every translated original site coordinate by coordinate against the
        request. Never return a state that silently misses an in-scope request.
        """
        if not self.translated_requests:
            return
        first_name = next(iter(self.translated_requests))
        if not not_jax_tracer(is_valid) or any(
            not not_jax_tracer(leaf) for leaf in jax.tree.leaves(init_params)
        ):
            raise _not_supported(
                first_name,
                "initialize_model is traced by a JAX transformation (jit/vmap/pmap,"
                " e.g. MCMC chain_method='parallel'), so the requested value cannot"
                " be verified",
            )
        pp = (
            postprocess_fn(*model_args, **model_kwargs)
            if dynamic_args
            else postprocess_fn
        )
        batched = not is_prng_key(rng_key)
        replayed = jax.vmap(pp)(init_params) if batched else pp(init_params)
        for name, want in self.translated_requests.items():
            if name not in replayed:
                raise _bad_value(
                    name,
                    "was not present when replaying the initial state through the model",
                )
            got_arr = replayed[name]
            if not not_jax_tracer(got_arr):
                raise _not_supported(
                    name,
                    "initialize_model is traced by a JAX transformation (jit/vmap/pmap,"
                    " e.g. MCMC chain_method='parallel'), so the requested value cannot"
                    " be verified",
                )
            got_raw = np.asarray(device_get(got_arr))
            got_dtype = (
                got_raw.dtype
                if np.issubdtype(got_raw.dtype, np.floating)
                else np.dtype(jnp.result_type(float))
            )
            tol = float(_ULPS * np.finfo(got_dtype).eps)
            got_np = got_raw.astype(np.float64)
            if batched:
                if got_np.ndim < 1 or got_np.shape[1:] != want.shape:
                    raise _bad_value(
                        name,
                        f"has shape {want.shape} but replaying the initial state"
                        f" produced shape {got_np.shape[1:] if got_np.ndim >= 1 else got_np.shape}",
                    )
                want_cmp = np.broadcast_to(want, got_np.shape)
            else:
                if got_np.shape != want.shape:
                    raise _bad_value(
                        name,
                        f"has shape {want.shape} but replaying the initial state"
                        f" produced shape {got_np.shape}",
                    )
                want_cmp = want
            bound = tol * np.maximum(np.abs(want_cmp), 1.0)
            bad = (
                (~np.isfinite(got_np))
                | (~np.isfinite(want_cmp))
                | (np.abs(got_np - want_cmp) > bound)
            )
            if np.any(bad):
                raise _replay_mismatch(name, want_cmp, got_np, bad, tol, batched)


def make_state(init_strategy):
    """
    Return a :class:`_State` if ``init_strategy`` is in scope, else ``None``.
    Only types, identity and exact-``str`` keys are inspected.
    """
    if _SUPPRESSED[0]:
        return None
    if type(init_strategy) is not partial or init_strategy.func is not init_to_value:
        return None
    if len(init_strategy.args) != 0:
        return None
    keywords = init_strategy.keywords
    if len(keywords) != 1:
        return None
    for key in keywords:
        if type(key) is not str or key != "values":
            return None
    values = keywords["values"]
    if type(values) is not dict or len(values) == 0:
        return None
    for key in values:
        if type(key) is not str:
            return None
    return _State(init_strategy, values)


def _init_to_value_reparam(site=None, *, state):
    if site is None:
        return partial(_init_to_value_reparam, state=state)
    recorder = handlers._REPARAM_INIT
    if recorder is not None and recorder.state is state:
        handled, value = recorder.on_site(site)
        if handled:
            return value
    return state.strategy(site)


class _Recorder:
    def __init__(self, state, w, prototype):
        self.state = state
        self.w = w
        self.prototype = prototype
        self.stack = []
        self.pending = {}
        self.records = []
        self.seen_names = set()

    # --- called by handlers.reparam.process_message -----------------------------------
    def open(self, handler, msg, reparam):
        rec = _Record(handler, msg, reparam)
        if self.stack:
            rec.nested = True
            for other in self.stack:
                other.nested = True
        self.stack.append(rec)
        return rec

    def close(self, rec, new_fn, value):
        top = self.stack.pop()
        assert top is rec
        if new_fn is None and value is not None:
            rec.value = value
            # keep a reference to the message so that its id cannot be reused
            self.pending[id(rec.msg)] = rec
            self.records.append(rec)

    # --- called by the init strategy for every message reaching the init substitute -----
    def on_site(self, site):
        kind = site["type"]
        if kind == "sample":
            if not self.stack or site["is_observed"]:
                return False, None
            rec = self.stack[-1]
            if rec.nested:
                return False, None
            if not rec.hooked:
                if rec.reason is None:
                    rec.reason = "a latent site was sampled before its auxiliary site"
                return False, None
            if rec.aux_seen:
                rec.extra_latent = True
                return False, None
            rec.aux_seen = True
            return self._translate(rec, site)
        if kind == "deterministic":
            rec = self.pending.get(id(site))
            if rec is not None and rec.msg is site and self._real_arrival():
                self._check_original(rec, site)
        return False, None

    def _real_arrival(self):
        """
        True when the message is being processed by the init substitute itself, i.e. it is
        unique on the stack and no ``block`` sits above it (``contrib.control_flow``
        scan/cond replay the strategy inside a ``block``).
        """
        w = self.w
        found = False
        for h in _PYRO_STACK:
            if type(h) is handlers.substitute and h.substitute_fn is w:
                if found:
                    return False
                found = True
            elif found and issubclass(type(h), handlers.block):
                return False
        return found

    def _walk(self, rec):
        """Exact handler-stack analysis: returns (ok, predicted original name, reason)."""
        w = self.w
        s_idx = r_idx = None
        for i, h in enumerate(_PYRO_STACK):
            if type(h) is handlers.substitute and h.substitute_fn is w:
                if s_idx is not None:
                    return False, None, "the initialization handler appears twice"
                s_idx = i
            if h is rec.handler:
                if r_idx is not None:
                    return False, None, "its reparam handler appears twice on the stack"
                r_idx = i
        if s_idx is None:
            return (
                False,
                None,
                "it is not visible to the initialization strategy"
                " (e.g. inside contrib.control_flow scan/cond)",
            )
        if r_idx is None or r_idx < s_idx:
            return False, None, "its reparam handler is outside initialize_model"
        for h in _PYRO_STACK[:s_idx]:
            if type(h) is not handlers.trace:
                return (
                    False,
                    None,
                    f"initialize_model is called under a {type(h).__name__} handler",
                )
        from numpyro.infer.util import _substitute_default_key

        for h in _PYRO_STACK[s_idx + 1 :]:
            th = type(h)
            if (
                not _is_one_of(th, _ALLOWED_HANDLERS)
                and th is not _substitute_default_key
            ):
                return (
                    False,
                    None,
                    f"it is reparameterized under a {type(h).__name__} handler, which is"
                    " not supported for initialization",
                )
        name = rec.name
        if type(name) is not str:
            return False, None, "its name is not a str"
        for i in range(r_idx - 1, s_idx, -1):
            h = _PYRO_STACK[i]
            if type(h) is handlers.scope:
                name = _scoped_name(h, name)
                if name is None:
                    return (
                        False,
                        None,
                        "it is under a scope whose prefix, divider or hide_types are not"
                        " plain str/int/float/bool/None values and a list/tuple/set of str",
                    )
        return True, name, None

    def _translate(self, rec, site):
        values = self.state.values
        aux_name = site["name"]
        if type(aux_name) is not str:
            rec.reason = "its auxiliary site name is not a str"
            return False, None
        if aux_name in values:
            # an explicit auxiliary request wins, exactly as before
            rec.aux_explicit = True
            return False, None
        ok, name, reason = self._walk(rec)
        if not ok:
            rec.reason = reason
            return False, None
        if name not in values:
            return False, None
        if self.state.refuse is not None:
            raise _not_supported(name, self.state.refuse)
        rec.request_name = name
        aux = self._aux_value(rec, site, name, values[name])
        rec.translated = True
        self.state.translated = True
        return True, aux

    def _aux_value(self, rec, site, name, requested):
        try:
            x = jnp.asarray(requested)
        except Exception as e:
            raise _bad_value(name, "cannot be converted to an array") from e
        if jnp.issubdtype(x.dtype, jnp.complexfloating):
            raise _bad_value(
                name, f"is complex (dtype {x.dtype}); expected a real value"
            )
        float_dtype = jnp.result_type(float)
        if not jnp.issubdtype(x.dtype, jnp.floating):
            x = x.astype(float_dtype)
        try:
            aux = rec.inverse(x)
        except _Unsupported as e:
            raise _not_supported(name, str(e)) from None
        aux = jnp.asarray(aux, dtype=float_dtype)
        fn = site["fn"]
        expected = tuple(site["kwargs"].get("sample_shape", ())) + tuple(fn.shape())
        if jnp.shape(aux) != expected:
            raise _bad_value(
                name,
                f"has shape {jnp.shape(x)}; it maps to shape {jnp.shape(aux)} for the"
                f" auxiliary site '{site['name']}', which has shape {expected}",
            )
        x0 = rec.forward(aux)
        if jnp.shape(x0) != jnp.shape(x):
            raise _bad_value(
                name,
                f"has shape {jnp.shape(x)} but the site has shape {jnp.shape(x0)};"
                " pass a value of the full site shape",
            )
        in_support = jnp.all(fn.support(aux))
        tol = _ULPS * jnp.finfo(jnp.result_type(x0)).eps
        close = jnp.all(jnp.abs(x0 - x) <= tol * jnp.maximum(jnp.abs(x), 1))
        ok = in_support & close & jnp.all(jnp.isfinite(aux))
        if not not_jax_tracer(ok):
            if self.prototype:
                raise _not_supported(
                    name,
                    "initialize_model is traced by a JAX transformation (jit/vmap/pmap,"
                    " e.g. MCMC chain_method='parallel'), so the requested value cannot"
                    " be verified",
                )
            return jnp.where(ok, aux, jnp.nan)
        if not bool(ok):
            if not self.prototype:
                # e.g. depends on a randomly initialized parent: reject this draw
                return jnp.full_like(aux, jnp.nan)
            if not bool(jnp.all(jnp.isfinite(aux))):
                raise _bad_value(
                    name,
                    "maps to a non-finite value of the auxiliary site"
                    f" '{site['name']}' (it is outside the support of the site or on its"
                    " boundary, where the log density is not finite)",
                )
            if not bool(in_support):
                raise _bad_value(
                    name,
                    f"maps to {aux}, which is outside the support of the auxiliary site"
                    f" '{site['name']}'",
                )
            raise _bad_value(
                name,
                f"is not reproduced by the reparametrizer within {_ULPS} ulps of"
                f" max(|value|, 1) (got {x0}); the reparametrizer cannot produce it in"
                f" {jnp.result_type(x0)}",
            )
        rec.request = x
        if self.prototype:
            self.state.translated_requests[name] = np.asarray(
                device_get(x), dtype=np.float64
            )
        return aux

    def _check_original(self, rec, site):
        rec.seen_at_init = True
        name = site["name"]
        if type(name) is str:
            self.seen_names.add(name)
        if rec.translated:
            if (
                type(name) is not str
                or name != rec.request_name
                or site["value"] is not rec.value
                or rec.extra_latent
            ):
                raise _not_supported(
                    rec.request_name,
                    "a handler between the reparametrizer and the initialization renamed"
                    " or replaced the site",
                )
            if self.prototype:
                x0 = site["value"]
                x = rec.request
                tol = _ULPS * jnp.finfo(jnp.result_type(x0)).eps
                close = jnp.all(jnp.abs(x0 - x) <= tol * jnp.maximum(jnp.abs(x), 1))
                if jnp.shape(x0) != jnp.shape(x) or not bool(close):
                    raise _bad_value(
                        rec.request_name,
                        f"is not reproduced by the reparametrizer (got {x0})",
                    )
            return
        if type(name) is not str:
            return
        values = self.state.values
        if name in values and not rec.aux_explicit and not self._aux_local_given(rec):
            raise _not_supported(name, rec.why_not())

    def _aux_local_given(self, rec):
        aux = rec.aux_local
        return type(aux) is str and aux in self.state.values

    def finish(self, model_trace):
        """
        After the prototype trace: refuse requests on reparameterized sites hidden from
        initialization, and translated sites that follow a discrete latent site (which
        `initialize_model` enumerates, so the initial value would not be determined).
        """
        values = self.state.values
        translated = {
            rec.request_name
            for rec in self.records
            if rec.translated and rec.seen_at_init
        }
        if translated:
            discrete = None
            for key, site in model_trace.items():
                if (
                    site["type"] == "sample"
                    and not site["is_observed"]
                    and site["fn"].support.is_discrete
                ):
                    discrete = key
                elif (
                    discrete is not None
                    and site["type"] == "deterministic"
                    and type(key) is str
                    and key in translated
                ):
                    raise _not_supported(
                        key,
                        f"it follows the discrete latent site '{discrete}', which is"
                        " enumerated, so the site's initial value is not determined",
                    )
        for rec in self.records:
            if rec.seen_at_init:
                continue
            if rec.translated:
                raise _not_supported(
                    rec.request_name, "the site was hidden from initialization"
                )
            name = rec.outer_name
            if type(name) is not str or name not in values:
                continue
            if self._aux_local_given(rec) or name in self.seen_names:
                # a visible reparameterized site of that name took the request
                continue
            visible = None
            for key, site in model_trace.items():
                if type(key) is str and key == name:
                    visible = site
                    break
            if visible is None or visible["type"] == "deterministic":
                raise _not_supported(
                    name,
                    rec.reason
                    if rec.reason is not None and not rec.nested and rec.hooked
                    else (
                        rec.why_not()
                        if rec.nested or not rec.hooked
                        else "the reparameterized site is hidden from initialization"
                        " (e.g. by block, scan or cond)"
                    ),
                )


def _checked(t):
    """``[t]`` after checking that ``t`` is an allowlisted bijection."""
    check_bijective(t)
    return [t]
