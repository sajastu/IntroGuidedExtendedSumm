import json

from utils.rouge_score import evaluate_rouge

intro_sents = ["The objective of the work presented here is to study the mechanism of radiative line driving and the corresponding properties of the winds of possible generations of very massive stars at extremely low metallicities and to investigate the principal influence of these winds on ionizing fluxes and observable ultraviolet spectra.", "The basic new element of this approach, needed in the domain of extremely low metallicity, is the introduction of depth dependent force multipliers representing the radiative line acceleration.", "Because of the depth dependent force multipliers a new formulation of the critical point equations is developed and a new iterative solution algorithm for the complete stellar wind problem is introduced (section 4)."]

non_intro_sents = ["In this section we develop a fast algorithm to calculate stellar wind structures and mass - loss rates from the equation of motion (eq.[eom1]) using a radiative line acceleration parametrized in the form of eq.[fmp3].", "After the new concept to calculate stellar wind structures with variable force multipliers has been introduced and tested by comparing with the observed wind properties.", "The purpose of this first study is to provide an estimate about the strengths of stellar winds at very low metallicity for very massive hot stars in a mass range roughly between 100 to 300 m@xmath3.", "With our new approach to describe line driven stellar winds at extremely low metallicity we were able to make first predictions of stellar wind properties, ionizing fluxes and synthetic spectra of a possible population of very massive stars in this range of metallicity.", "We have demonstrated that the normal scaling laws , which predict stellar - mass loss rates and wind momenta to decrease as a power law with @xmath1 break down at a certain threshold and we have replaced the power - law by a different fit – formula.", "We also calculated synthetic spectra and were able to present for the first time predictions of uv spectra of very massive stars at extremely low metallicities.", "We learned that the presence of stellar winds leads to observable broad spectral line features, which might be used for spectral diagnostics, should such an extreme stellar population be detected at high redshift."]

rg_scores = dict()
for i, s in enumerate(intro_sents):
    for j, sn in enumerate(non_intro_sents):
        if 's' + str(i+1) not in rg_scores.keys():
            rg_scores['s' + str(i+1)] = dict()
        rg_scores['s' + str(i+1)]['s' + str(j+4)] = (evaluate_rouge([s], [sn])[-1] + evaluate_rouge([s], [sn])[-2])

json.dump(rg_scores, open('anal_rg.json', mode='w'))