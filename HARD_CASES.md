# Hard Cases for NumPyro Issue #1614

Pinned baseline: `pyro-ppl/numpyro` at `fc7691c7c39f72d45b20374dfc1f2cabc68df0a7` (CPU, JAX/jaxlib `0.12.0.1`, default float32).
Suite: `test_reparam_init_1614.py` (40 tests: master 10 passed / 30 failed; `v7.patch` 31 passed / 9 failed).

| Test name | Group | What it checks | Master | v7 | Found in |
| --- | --- | --- | --- | --- | --- |
| `test_issue_1614_init_to_value` | (a) core bug | `#1614` `CircularReparam` + `init_to_value({"loc_1": 0.5})` starts at `0.5` | FAIL | PASS | r1, bank |
| `test_issue_1614_init_to_median` | (a) core bug | `#1614` `CircularReparam` + `init_to_median` starts near `0.0` or raises naming `loc_1` | FAIL | FAIL | r1, r4, bank |
| `test_ssbvm_mixture_kmeans_init` | (a) core bug | `examples/ssbvm_mixture.py` enumerated `SineBivariateVonMises` starts at k-means centers | FAIL | PASS | r1, bank |
| `test_dependent_reparam_chain_depth_5` | (a) core bug | 5-site dependent `LocScaleReparam` chain replays to `[1, 2, 3, 4, 5]` | FAIL | PASS | r4, r5, bank |
| `test_reparam_families_init_to_value` | (a) core bug | `Transform`, `Explicit`, `ProjectedNormal`, `Haar`, `DiscreteCosine` honor explicit values | FAIL | PASS | r3, r5, bank |
| `test_stacked_reparams_init_to_value` | (a) core bug | Stacked `TransformReparam` + `LocScaleReparam` honors `rate=4.0` or names `rate` | FAIL | PASS | r3, r5 |
| `test_plate_enclosing_reparam_full_request` | (b) plates/scope/mask/chains | Enclosing `plate(4)` + full-shape request preserves `(4,)` shape and values | FAIL | PASS | r1, r2 |
| `test_plate_scalar_request_preserves_shape` | (b) plates/scope/mask/chains | Scalar request in plated 2-chain NUTS broadcasts to `(2, 4)` or raises naming `theta` | FAIL | PASS | r2, bank |
| `test_scope_outside_reparam_partial_centering` | (b) plates/scope/mask/chains | Enclosing `scope` around `LocScaleReparam(0.25)` honors `group/angle=0.375` | FAIL | PASS | r1, r3 |
| `test_broadcasting_mask_preserves_shape` | (b) plates/scope/mask/chains | Broadcasting 3-element `mask` around scalar reparam preserves `(3,)` or names `theta` | FAIL | PASS | r3, r4 |
| `test_sample_shape_request_preserves_shape` | (b) plates/scope/mask/chains | Explicit `sample_shape=(3,)` honors 3-vector or raises naming `theta` | FAIL | PASS | r2, r5 |
| `test_vectorized_chains_nuts_init_to_value` | (b) plates/scope/mask/chains | 3-chain vectorized `MCMC(NUTS)` starts all chains at requested `(3, 4)` vector | FAIL | PASS | r1, r5, r7, bank |
| `test_subsampled_plate_data_dependent_loc` | (b) plates/scope/mask/chains | Subsampled plate with index-dependent loc via `HMCECS` honors 4-vector or names `theta` | FAIL | PASS | r7 |
| `test_control_flow_cond_reparam_init` | (b) plates/scope/mask/chains | Scoped `cond` branches with `LocScaleReparam` honor `g/theta=3.0` or name site | FAIL | PASS | r4, r5 |
| `test_block_hide_fn_not_called_on_original_sphere` | (c) handlers/objects | `block(hide_fn=...)` calling `biject_to` never sees original `ProjectedNormal` sphere | PASS | PASS | r2 |
| `test_block_hidden_neutra_same_local_name` | (c) handlers/objects | Blocked `NeuTraReparam` named `target` does not break visible `target=11.0` | PASS | PASS | r3, r4 |
| `test_substitute_fn_no_extra_eq_calls` | (c) handlers/objects | Custom `substitute_fn` callable gets zero extra `func.__eq__` comparisons | PASS | PASS | r3 |
| `test_empty_values_dict_subclass_no_extra_bool` | (c) handlers/objects | Empty `dict` subclass passed to `init_to_value` gets zero `__bool__` calls | PASS | PASS | r4, r5 |
| `test_nonempty_values_dict_subclass_behavior` | (c) handlers/objects | Non-empty `dict` subclass `Values(theta=4.0)` honors `4.0` or raises naming `theta` | FAIL | FAIL | r5 |
| `test_custom_partial_strategy_no_extra_func_reads` | (c) handlers/objects | Custom `partial` subclass sees no extra `.func` descriptor reads vs plain model | PASS | PASS | r4 |
| `test_delegating_custom_strategy_prototype_matches_final` | (c) handlers/objects | User callable forwarding to `init_to_mean` agrees between prototype and final replay | PASS | PASS | r4, r5 |
| `test_unit_jacobian_suffix_no_extra_format` | (c) handlers/objects | `UnitJacobianReparam` `str` suffix gets no extra `__format__` call for unrelated request | PASS | PASS | r6 |
| `test_custom_messenger_no_extra_class_name_reads` | (c) handlers/objects | Inert custom `Messenger` gets zero `__name__` reads when reparam site is unrequested | PASS | FAIL | r7 |
| `test_debug_key_reuse_find_valid_initial_params` | (c) handlers/objects | Direct `find_valid_initial_params` under `debug_key_reuse(True)` avoids `KeyReuseError` and honors/names `theta` | FAIL | FAIL | r4, r5 |
| `test_float32_locscale_partial_precision_loss` | (d) numeric edges | `Normal(2^18, 2^18)` + `LocScaleReparam(0.5)` requesting `1.0` does not silently return `0.0` | FAIL | PASS | r2, r3 |
| `test_float32_projected_normal_large_concentration` | (d) numeric edges | `ProjectedNormal([2^25]*3)` requesting `[0.6, -0.8, 0]` does not silently collapse to `(1,1,1)/sqrt(3)` | FAIL | PASS | r1, r2 |
| `test_float32_compiled_vs_eager_affine_fma` | (d) numeric edges | `4097*z - float32(4097^2)` compiled HMC replay matches `0.0` (not FMA `1.0`) or names `position` | FAIL | FAIL | r7 |
| `test_int64_narrowing_overflow_x32` | (d) numeric edges | Host `int64(2^32)` / `int64(2^32+7)` in x32 does not silently wrap to `0` / `7` | FAIL | FAIL | r7 |
| `test_int32_angle_circular_nuts` | (d) numeric edges | `jnp.int32(0)` angle in `CircularReparam` NUTS produces float auxiliary at `0.0` or names `angle` | FAIL | PASS | r2, r3, r4 |
| `test_bounded_uniform_affine_near_endpoint_clip` | (d) numeric edges | `AffineTransform(Uniform)` near float32 upper edge does not silently clip `0.0` to `-0.0625` | FAIL | PASS | r6, r7 |
| `test_bounded_uniform_closed_endpoint_error_names_site` | (d) numeric edges | Boundary `0.0` on `TransformedDistribution(Uniform)` does not raise generic `RuntimeError` omitting `position` | FAIL | FAIL | r7 |
| `test_folded_uniform_negative_preimage_abs_transform` | (d) numeric edges | `FoldedDistribution(Uniform(-4, -1))` requesting `2.5` finds base `-2.5` or names `length` | FAIL | PASS | r5 |
| `test_projected_normal_off_sphere_request` | (d) numeric edges | Off-sphere `[1.00008, 0, 0]` in float32 raises naming `direction` instead of silently normalizing | FAIL | FAIL | r7 |
| `test_jit_initialize_model_reparam` | (d) numeric edges | `jax.jit(initialize_model)` honors `theta=5.0` or raises naming `theta` | FAIL | PASS | r4, r5, r7, bank |
| `test_autoguides_svi_plated_reparam_init` | (e) entry points | `AutoDelta`/`AutoNormal`/`AutoDiagonalNormal` + `SVI` preserve `(4,)` shape and honor `init_to_value` or name `theta` | FAIL | FAIL | r1, r2, r5, bank |
| `test_svi_partial_init_params_autodel_invariance` | (e) entry points | Partial `SVI.init(init_params=...)` leaves unrequested `CircularReparam` `AutoDelta` loc unchanged | PASS | PASS | r5 |
| `test_predictive_autodel_params_bool_and_fresh_guide` | (e) entry points | `Predictive(guide=AutoDelta)` makes zero `params.__bool__` calls and runs fresh guide cleanly | PASS | PASS | r4, r5 |
| `test_sa_init_to_value_iteration_zero` | (e) entry points | `SA.init` either starts at `theta=7.0` at iteration 0 or raises naming `theta` (not random cloud draw) | FAIL | PASS | r6, r7 |
| `test_gibbs_hmcgibbs_and_mixedhmc_init` | (e) entry points | `Gibbs`, `HMCGibbs`, `DiscreteHMCGibbs`, `MixedHMC` honor `theta=4.0` at iter 0 or name `theta` | FAIL | PASS | r5, r7, bank |
| `test_nested_sampler_and_uniform_reparam` | (e) entry points | `NestedSampler` runs cleanly with `LocScaleReparam`; `UniformReparam` honors `0.0` or names `theta` | FAIL | PASS | r4, r5, r7 |

Why fixing #1614 cleanly is hard:
1. Init values are looked up before enclosing `plate`, `scope`, and `mask` handlers have expanded shapes or prefixed site names, so translating at `reparam` time mis-resolves names and collapses batch shapes.
2. A reparameterizer's inverse is not generally available or unique at init time (non-bijective transforms, custom/stacked reparams, dependent parent values, and float32/XLA-FMA/int64 rounding mean eager inverses can disagree with compiled replay).
3. Effect-handler composition (`block`, `condition`, `substitute`, `AutoGuide`, `Predictive`, `SA`, `Gibbs`, custom `Messenger`/dict/str objects) makes any global init hook leak observable side effects or break out-of-scope invariance.
