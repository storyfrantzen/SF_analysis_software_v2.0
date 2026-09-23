# Published reference data

`clas6_bedlinskiy_2014_structure_functions.csv` transcribes Table VIII of
I. Bedlinskiy et al. (CLAS Collaboration), *Exclusive pi0 electroproduction at
W > 2 GeV with CLAS*, Physical Review C **90**, 025205 (2014),
DOI: 10.1103/PhysRevC.90.025205, arXiv:1405.0988.

The table reports the unseparated `sigma_U = sigma_T + epsilon sigma_L`,
`sigma_LT`, and `sigma_TT` structure functions in nb/GeV2 at a 5.75 GeV beam
energy. The `_stat` and `_sys` columns are the first and second published
uncertainties, respectively. `Q2` and `minus_t` are in GeV2. The 96 rows were
checked against the rendered published table on pages 24 and 25 of the article.

`igor_pass2_v1_structure_functions.csv` transcribes the 174 structure-function
rows printed in `IGOR_Pi0_Analysis_Pass2_version1.pdf` (October 2025), supplied
by the analysis author for an internal RGA Fall 2018 comparison.  The note
reports statistical uncertainties only and states that the tabulated values
already include its global factor of 1.3.  The `_sys` columns are therefore
zero placeholders required by the common reference-table schema; comparisons
must not interpret them as a claim of zero systematic uncertainty.  The `row`
column preserves the table row number and `reported_global_scale` records the
stated factor.
