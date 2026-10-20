.. _api_ref:

.. currentmodule:: neighbayes

API reference
=============

Models
------

A model class names a process and a family: ``SAR``, ``SARLogit``,
``SARNegBinFlow``. Panel structure, effects, dynamics, and the flow restriction
are arguments to the same class:

- ``time_col`` (formula mode) or ``T`` (matrix mode) makes the model a panel;
- ``unit_effects`` (``pair_effects`` for flows) is ``None``, ``"fixed"``, or
  ``"random"``, and ``time_effects=True`` adds fixed time effects;
- ``dynamic=True`` adds the time lag :math:`y_{t-1}`, and ``dynamic="restricted"``
  gives the restricted dynamic SDM;
- ``separable=True`` pins :math:`\rho_w = -\rho_o \rho_d` in a flow model.

A combination the package does not implement raises when the model is
constructed.

.. currentmodule:: neighbayes.models

Gaussian
^^^^^^^^

.. autosummary::
   :toctree: generated/

   OLS
   SLX
   SAR
   SEM
   SDM
   SDEM

Censored and binary
^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: generated/

   SARTobit
   SEMTobit
   SDMTobit
   Logit
   SARLogit
   SEMLogit
   SDEMLogit
   SEMProbit

Counts
^^^^^^

.. autosummary::
   :toctree: generated/

   NegBin
   SARNegBin
   ZINB
   SARZINB
   HurdleNegBin
   SARHurdleNegBin

Flows
^^^^^

.. autosummary::
   :toctree: generated/

   OLSFlow
   SARFlow
   SEMFlow
   NegBinFlow
   SARNegBinFlow
   SARPoissonFlow
   SARZINBFlow
   SARHurdleNegBinFlow

Multilevel
^^^^^^^^^^

.. autosummary::
   :toctree: generated/

   Multilevel
   Level

Default Gibbs Priors
---------------------

.. currentmodule:: neighbayes.models.priors

.. autosummary::
   :toctree: generated/

   GibbsBasePriors
   GaussianGibbsPriors
   ReducedGibbsPriors
   FlowReducedGibbsPriors
   ZINBGibbsPriors
   LogitGibbsPriors
   REGibbsPriors

Default NUTS Priors
---------------------

.. currentmodule:: neighbayes.models.priors

.. autosummary::
   :toctree: generated/

   OLSPriors
   SLXPriors
   SARPriors
   SEMPriors
   SDMPriors
   SDEMPriors
   NegBinPriors
   SARNegBinPriors
   SARTobitPriors
   SEMTobitPriors
   SDMTobitPriors
   SARProbitPriors
   SARLogitPriors
   SEMLogitPriors
   PanelBasePriors
   PanelOLSPriors
   PanelSARPriors
   PanelSEMPriors
   PanelSDEMPriors
   PanelOLSREPriors
   PanelSARREPriors
   PanelSEMREPriors
   PanelSDEMREPriors
   PanelCountPriors
   PanelZINBPriors
   SARZINBPriors
   PanelHurdlePriors
   FlowPanelHurdlePriors
   SARHurdlePriors
   MultilevelPriors
   PanelDynamicBasePriors
   PanelOLSDynamicPriors
   PanelSLXDynamicPriors
   PanelSARDynamicPriors
   PanelSEMDynamicPriors
   PanelSDMRDynamicPriors
   PanelSDMUDynamicPriors
   PanelSDEMDynamicPriors


Bayesian Diagnostics
--------------------

.. currentmodule:: neighbayes.diagnostics.lmtests

.. autosummary::
   :toctree: generated/

   BayesianLMTestResult
   bayesian_lm_lag_test
   bayesian_lm_error_test
   bayesian_lm_wx_test
   bayesian_lm_sdm_joint_test
   bayesian_lm_slx_error_joint_test
   bayesian_lm_error_from_sar_test
   bayesian_lm_error_sdm_test
   bayesian_lm_lag_sdem_test
   bayesian_lm_wx_sem_test
   bayesian_robust_lm_lag_test
   bayesian_robust_lm_error_test
   bayesian_robust_lm_lag_sdm_test
   bayesian_robust_lm_wx_test
   bayesian_robust_lm_error_sdem_test
   bayesian_robust_lm_error_sar_test
   bayesian_robust_lm_error_sdm_test
   bayesian_robust_lm_lag_sdem_test
   bayesian_robust_lm_lag_sem_test
   bayesian_robust_lm_wx_sem_test

GLM Bayesian LM Tests
^^^^^^^^^^^^^^^^^^^^^

LM tests for the logit and NB families: the latent SAR and SEM score tests at the GLM null (Logit, NegBin), the error and WX tests at a reduced-form SAR null (SARLogit, SARNegBin), and a Moran diagnostic on Pólya–Gamma working residuals.

.. autosummary::
   :toctree: generated/

   bayesian_glm_lm_lag_test
   bayesian_glm_lm_error_test
   bayesian_glm_robust_lm_lag_test
   bayesian_glm_robust_lm_error_test
   bayesian_glm_lm_sarma_test
   bayesian_glm_lm_wx_test
   bayesian_glm_moran_working_test


Panel Bayesian LM Tests
^^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: generated/

   bayesian_panel_lm_lag_test
   bayesian_panel_lm_error_test
   bayesian_panel_robust_lm_lag_test
   bayesian_panel_robust_lm_error_test
   bayesian_panel_lm_wx_test
   bayesian_panel_lm_sdm_joint_test
   bayesian_panel_lm_slx_error_joint_test
   bayesian_panel_lm_error_sdm_test
   bayesian_panel_lm_lag_sdem_test
   bayesian_panel_lm_wx_sem_test
   bayesian_panel_robust_lm_lag_sdm_test
   bayesian_panel_robust_lm_wx_test
   bayesian_panel_robust_lm_error_sdem_test


Flow Bayesian LM Tests
^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: generated/

   bayesian_lm_flow_destination_test
   bayesian_lm_flow_origin_test
   bayesian_lm_flow_od_test
   bayesian_lm_flow_intraunit_test
   bayesian_lm_flow_joint_test
   bayesian_robust_lm_flow_destination_test
   bayesian_robust_lm_flow_origin_test
   bayesian_robust_lm_flow_od_test
   bayesian_panel_lm_flow_destination_test
   bayesian_panel_lm_flow_origin_test
   bayesian_panel_lm_flow_od_test
   bayesian_panel_lm_flow_intraunit_test
   bayesian_panel_lm_flow_joint_test


Diagnostic Test Suites
^^^^^^^^^^^^^^^^^^^^^^

Pre-bundled collections of LM tests used by ``model.spatial_diagnostics()``
and the decision-tree renderers.

.. autosummary::
   :toctree: generated/

   DiagnosticSuite
   get_diagnostic_suite
   OLS_SUITE
   OLS_PANEL_SUITE
   SAR_SUITE
   SAR_PANEL_SUITE
   SAR_TOBIT_SUITE
   SEM_SUITE
   SEM_PANEL_SUITE
   SEM_PANEL_DYNAMIC_SUITE
   SLX_SUITE
   SLX_PANEL_SUITE
   SLX_PANEL_DYNAMIC_SUITE
   SDM_SUITE
   SDM_PANEL_SUITE
   SDM_TOBIT_SUITE
   SDEM_SUITE
   SDEM_PANEL_SUITE
   FLOW_SUITE
   FLOW_INTRA_SUITE
   FLOW_PANEL_SUITE
   LOGIT_SUITE
   NEGBIN_SUITE
   SAR_LOGIT_SUITE
   SAR_NEGBIN_SUITE


Bayesian Model Comparison
^^^^^^^^^^^^^^^^^^^^^^^^^

.. currentmodule:: neighbayes.diagnostics

.. autosummary::
   :toctree: generated/

   ModelComparison
   bayes_factor_compare_models


MCMC Efficiency
^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: generated/

   SpatialMCMCReport
   spatial_mcmc_diagnostic


Spatial Cross-Validation
^^^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: generated/

   SpatialCVResult
   spatial_kfold


Data Generating Processes
-------------------------

.. note::

   The cross-sectional and (scalar) panel DGP simulators accept ``W``
   (Graph/sparse/dense) and ``gdf`` inputs.  You may provide both
   together; in that case ``W`` is used for simulation and is checked
   against ``gdf`` for dimensional compatibility (a ``ValueError`` is
   raised when they do not describe the same number of spatial units).

   The flow DGPs below take ``G`` (libpysal Graph), ``gdf``, ``n``, and
   ``knn_k`` instead.  All four are optional: when none is supplied the
   DGP synthesises a point grid via
   :func:`~neighbayes.dgp.utils.synth_point_geodataframe` and builds a
   row-standardised KNN graph automatically.

   Each simulator draws from the model class it is named for and takes the
   same design arguments: ``T`` (with ``N``) for a panel, ``unit_effects`` and
   ``time_effects`` (scaled by ``unit_effect_sd`` and ``time_effect_sd``,
   default 0.5), and ``dynamic``.  Fixed and random unit effects draw the same
   data; the distinction belongs to the model.

.. currentmodule:: neighbayes.dgp

.. autosummary::
   :toctree: generated/

   simulate_ols
   simulate_slx
   simulate_sar
   simulate_sem
   simulate_sdm
   simulate_sdem
   simulate_sar_tobit
   simulate_sem_tobit
   simulate_sdm_tobit
   simulate_logit
   simulate_sar_logit
   simulate_sem_logit
   simulate_sdem_logit
   simulate_sem_probit
   simulate_negbin
   simulate_sar_negbin
   simulate_zinb
   simulate_sar_zinb
   simulate_hurdle_negbin
   simulate_sar_hurdle_negbin
   simulate_multilevel


Flow Data Generating Processes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Each flow DGP accepts ``n``, ``G``, ``gdf``, and ``knn_k`` as optional
arguments and (unless ``gamma_dist=0.0``) appends a ``log_distance``
column ``log(1 + d_{ij})`` with default coefficient ``gamma_dist=-0.5``.

Each flow simulator takes ``separable``, ``T``, ``pair_effects``, and
``time_effects`` as its model class does.  The Gaussian flow DGPs
(``simulate_sar_flow`` and ``simulate_sem_flow``) default to
``distribution="lognormal"``, returning strictly-positive flows
``y = exp(eta)`` where ``eta`` is the latent SAR-filtered linear
predictor (also exposed in the result dict as ``"eta_vec"`` /
``"eta"``).  Pass ``distribution="normal"`` to recover the legacy
Gaussian-on-y behaviour.  The Gaussian-likelihood flow models in
``neighbayes.models.flow`` operate on the latent scale, so fit on
``np.log(y)`` to recover the SAR parameters.  The Negative
Binomial DGPs are unchanged.

.. autosummary::
   :toctree: generated/

   simulate_ols_flow
   simulate_sar_flow
   simulate_sem_flow
   simulate_negbin_flow
   simulate_sar_negbin_flow
   simulate_sar_poisson_flow
   simulate_sar_zinb_flow
   simulate_sar_hurdle_negbin_flow



Graph Utilities
---------------

.. currentmodule:: neighbayes.graph

.. autosummary::
   :toctree: generated/

   FlowDesignMatrix
   flow_design_matrix
   flow_design_matrix_asymmetric
   flow_design_matrix_with_orig
   flow_weight_matrices
   destination_weights
   origin_weights
   od_weights


