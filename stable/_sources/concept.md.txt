# Why

Apart from the general need for current Bayesian spatial models in Python, the package is motivated by a handful of specific gaps

## Gibbs beats NUTS for spatial autoregressive models

Python has a flourishing Bayesian ecosystem, but its organized around NUTS, which struggles relative to Gibbs for spatial models (Wolf et al.) 

## The Jacobian term needs special care

computing the spatial log determinant has huge performance implications, especially for MCMC (bivand); there's no solid open implementation for fast and accurate lodget estimation (`bsreg` 2023 in R is the most recent and it's not performant). This dramatically limits scalability

## Spatial non-linear models

probit and tobit are old hat. Polya-gamma augmentation has been introduced in the spatial econometric literature (Krisztin) but only the logit model, and with no open implementation. These models have performance/scalability concerns even beyond the logdet

## Spatial Flow Models

similarly, flow models are well-known in the spatial econometrics literature (lesage), but no open implementation exists (i *think* matlab is the only one); performance again

## Bayesian diagnostics

the Bayesian workflow needs spatial contamination diagnostics; this work was started in dogan but (1) not fleshed out
for the array of spatial models and (2) no implementation (performance again)

## Model suite

The package organizes models along three column dimensions — **likelihood** (linear / non-linear), **temporal structure** (cross-section / panel), and **outcome structure** (single / flow). Each cell lists the spatial structures implemented for that combination.

<table style="border-collapse: collapse; text-align: center; margin: 1em auto;">
  <thead>
    <tr>
      <th rowspan="2" style="border: 1px solid #999; padding: 8px 12px; background: #f5f5f5;"></th>
      <th colspan="2" style="border: 1px solid #999; padding: 8px 12px; background: #e8e8e8;"><strong>Linear</strong></th>
      <th colspan="2" style="border: 1px solid #999; padding: 8px 12px; background: #e8e8e8;"><strong>Non-linear</strong></th>
    </tr>
    <tr>
      <th style="border: 1px solid #999; padding: 6px 12px;">Cross-section</th>
      <th style="border: 1px solid #999; padding: 6px 12px;">Panel</th>
      <th style="border: 1px solid #999; padding: 6px 12px;">Cross-section</th>
      <th style="border: 1px solid #999; padding: 6px 12px;">Panel</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th style="border: 1px solid #999; padding: 8px 12px; background: #f5f5f5;"><strong>Single</strong></th>
      <td style="border: 1px solid #999; padding: 10px 14px; vertical-align: top; text-align: left; line-height: 1.6;">Aspatial, SLX,<br>SAR, SEM,<br>SDM, SDEM</td>
      <td style="border: 1px solid #999; padding: 10px 14px; vertical-align: top; text-align: left; line-height: 1.6;">Aspatial, SLX,<br>SAR, SEM,<br>SDM, SDEM</td>
      <td style="border: 1px solid #999; padding: 10px 14px; vertical-align: top; text-align: left; line-height: 1.6;">Aspatial, SAR,<br>SEM, SDM</td>
      <td style="border: 1px solid #999; padding: 10px 14px; vertical-align: top; text-align: left; line-height: 1.6;">SAR, SEM</td>
    </tr>
    <tr>
      <th style="border: 1px solid #999; padding: 8px 12px; background: #f5f5f5;"><strong>Flow</strong></th>
      <td style="border: 1px solid #999; padding: 10px 14px; vertical-align: top; text-align: left; line-height: 1.6;">Aspatial, SAR,<br>SEM, SDEM</td>
      <td style="border: 1px solid #999; padding: 10px 14px; vertical-align: top; text-align: left; line-height: 1.6;">Aspatial, SAR,<br>SEM</td>
      <td style="border: 1px solid #999; padding: 10px 14px; vertical-align: top; text-align: left; line-height: 1.6;">Aspatial, SAR</td>
      <td style="border: 1px solid #999; padding: 10px 14px; vertical-align: top; text-align: left; line-height: 1.6;">Aspatial, SAR</td>
    </tr>
  </tbody>
</table>
