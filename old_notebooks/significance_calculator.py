import functools
import numpy as np
from matplotlib import pyplot as plt
import uproot
import mpmath
from likelihoods import LEff, LEff_v
import operator
import scipy as sp


class significanceCalculator(object):

    def __init__(self, expected_bin_contents, bin_edges=None, pot=1):
        if expected_bin_contents is not None:
            assert 'signal' in expected_bin_contents.keys()
            if type(list(expected_bin_contents.values())[0]) == np.ndarray:
                self.expected_bin_contents = expected_bin_contents
                self.expected_bin_sigmas = None
                self.n_bin = len(expected_bin_contents['signal'])
            elif isinstance(list(expected_bin_contents.values())[0], tuple) or isinstance(list(expected_bin_contents.values())[0], list):
                bin_means = [value[0] for value in expected_bin_contents.values()]
                bin_sigmas = [value[2] for value in expected_bin_contents.values()]
                self.expected_bin_contents = dict(zip(expected_bin_contents.keys(), bin_means))
                self.expected_bin_sigmas = dict(zip(expected_bin_contents.keys(), bin_sigmas))
                self.n_bin = len(expected_bin_contents['signal'][0])
            else:
                print('something went wrong')
                return

            self.bin_edges = bin_edges

            self.total_bg_prediction = np.zeros(self.n_bin)
            for name, prediction in self.expected_bin_contents.items():
                if name == 'signal':
                    continue
                self.total_bg_prediction += prediction

            if self.expected_bin_sigmas is not None:
                self.total_bg_sigma2 = np.zeros(self.n_bin)
                self.signal_sigma2 = self.expected_bin_sigmas['signal']**2
                if self.expected_bin_sigmas is not None:
                    for name, sigma in self.expected_bin_sigmas.items():
                        if name == 'signal':
                            continue
                        self.total_bg_sigma2 += sigma**2

        self.pot = pot

        self.ts_dict = {'pois_llr': self.neymanPearsonLikelihoodRatio,
                        'gaus_llr': self.gaussianLikelihoodRatio,
                        'delta_chi2': self.deltaChi2,
                        'delta_chi2_cov': self.deltaChi2WithCov,
                        'chi2_cov': self.Chi2WithCov,
                        'chi2_cnp': self.Chi2CNP,
                        'chi2_mu0': self.Chi2_mu0,
                        'chi2_mu1': self.Chi2_mu1,
                        'eff_llr': self.logRatioEffectiveLikelihood,
                        }

        self.ts_labels = {'pois_llr': 'Poisson log likelihood ratio',
                          'gaus_llr': 'Gaussian log likelihood ratio',
                          'delta_chi2': r'$\Delta X^2$ - stat only',
                          'delta_chi2_cov': r'$\Delta X^2$ - with systematics',
                          'chi2_cov': r'$X^2$ - with systematics',
                          'chi2_cnp': r'$X^2$ CNP',
                          'chi2_mu0': r'$X^2(\mu=0)$',
                          'chi2_mu1': r'$X^2(\mu=1)$',
                          'eff_llr': 'Effective log likelihood ratio',
                          }

    def setDataFromTTrees(self, dict_tfile_names, dict_pot_scaling, weight_variable='weightSplineTimesTune'):
        self.dict_dataset = {}
        for sample_name, tfile_name in dict_tfile_names.items():
            aux_file = uproot.open(tfile_name)
            aux_array = aux_file['NeutrinoSelectionFilter'].arrays(namedecode="utf-8")

            if 'weightTune' in aux_array.keys():
                aux_array['weightTune'][aux_array['weightTune'] <= 0] = 1.
                aux_array['weightTune'][np.isinf(aux_array['weightTune'])] = 1.
                aux_array['weightTune'][aux_array['weightTune'] > 100] = 1.
                aux_array['weightTune'][np.isnan(aux_array['weightTune']) == True] = 1.
            if 'weightSplineTimesTune' in aux_array.keys():
                aux_array['weightSplineTimesTune'][aux_array['weightSplineTimesTune'] <= 0] = 1.
                aux_array['weightSplineTimesTune'][np.isinf(aux_array['weightSplineTimesTune'])] = 1.
                aux_array['weightSplineTimesTune'][aux_array['weightSplineTimesTune'] > 100] = 1.
                aux_array['weightSplineTimesTune'][np.isnan(aux_array['weightSplineTimesTune']) == True] = 1.

            if weight_variable in aux_array.keys():
                aux_array['this_weight'] = aux_array[weight_variable] * dict_pot_scaling[sample_name]
            else:
                aux_array['this_weight'] = (aux_array['evt'] == aux_array['evt']) * dict_pot_scaling[sample_name]
            self.dict_dataset[sample_name] = aux_array

    def setVariableOfInterest(self, bin_edges, reco_energy, true_energy, true_pdg):
        self.bin_edges = bin_edges
        self.reco_energy = reco_energy
        self.n_bin = len(bin_edges) - 1
        self.true_energy = true_energy
        self.true_pdg = true_pdg

        for array in self.dict_dataset.values():
            if true_energy not in array.keys():
                array[true_energy] = 0*(array['evt'] == array['evt'])
            if true_pdg not in array.keys():
                array[true_pdg] = 0*(array['evt'] == array['evt'])

    def setCovarianceMatrix(self, cov, is_cov='cov'):
        self.cov_matrix = cov
        self.is_cov = is_cov

    def setNuOscillator(self, nu_oscillator, deltam2, sin2theta2):
        self.nu_oscillator = nu_oscillator
        self.deltam2 = deltam2
        self.sin2theta2 = sin2theta2
        self.label = (r'Sensitivity to $\Delta m^2_{14}$ = ' +
                  '{:.3g}'.format(deltam2) +
                  r' $eV^2, \sin^2(2\theta_{e\mu})$ = ' +
                  '{:.3g}'.format(sin2theta2))

    def setPOT(self, pot):
        self.pot = pot

    def setSelectionLabel(self, label):
        self.selection_label = label

    def fillHistogramPlain(self):
        self.expected_bin_contents = {}
        for name, array in self.dict_dataset.items():
            print(name)
            aux_mean, _ = np.histogram(array[self.reco_energy],
                                          weights=array['this_weight'],
                                          bins=self.bin_edges)
            self.expected_bin_contents[name] = aux_mean
        self.total_bg_prediction = functools.reduce(operator.add, self.expected_bin_contents.values())
        
    def fillHistogramOsc(self):
        self.expected_bin_contents = {'bg': np.zeros(self.n_bin),
                  'nue': np.zeros(self.n_bin),
                  'signal': np.zeros(self.n_bin)}
        for array in self.dict_dataset.values():
            array['signal_weight'] = array['this_weight'] * self.nu_oscillator.oscillationWeightAtEnergy(array['nu_e'], self.deltam2, self.sin2theta2)

            nu_e = (np.abs(array[self.true_pdg]) == 12)
            non_nu_e = (np.abs(array[self.true_pdg]) != 12)

            aux_mean_bg, _ = np.histogram(array[self.reco_energy][non_nu_e],
                                          weights=array['this_weight'][non_nu_e],
                                          bins=self.bin_edges)
            aux_mean_nue, _ = np.histogram(array[self.reco_energy][nu_e],
                                          weights=array['this_weight'][nu_e],
                                          bins=self.bin_edges)
            aux_signal, _ = np.histogram(array[self.reco_energy][nu_e],
                                          weights=array['signal_weight'][nu_e],
                                          bins=self.bin_edges)

            self.expected_bin_contents['bg'] += aux_mean_bg
            self.expected_bin_contents['nue'] += aux_mean_nue
            self.expected_bin_contents['signal'] += aux_signal
        self.total_bg_prediction = self.expected_bin_contents['bg'] + self.expected_bin_contents['nue']

    def produceEntriesForOsc(self):
        self.expected_bin_contents = {}
        self.total_bg_prediction = np.zeros(self.n_bin)
        aux_nue_true_energy = []
        aux_nue_reco_energy = []
        aux_nue_weight = []

        for array in self.dict_dataset.values():
            nu_e = (np.abs(array[self.true_pdg]) == 12)
            self.total_bg_prediction += np.histogram(array[self.reco_energy],
                                          weights=array['this_weight'],
                                          bins=self.bin_edges)[0]
            
            aux_nue_reco_energy.append(array[self.reco_energy][nu_e])
            aux_nue_true_energy.append(array[self.true_energy][nu_e])
            aux_nue_weight.append(array['this_weight'][nu_e])
        
        self.nue_reco_energy = np.concatenate(aux_nue_reco_energy)
        self.nue_true_energy = np.concatenate(aux_nue_true_energy)
        self.nue_weight = np.concatenate(aux_nue_weight)

    def fillHistogramNueBeam(self):
        self.expected_bin_contents = {'bg': np.zeros(self.n_bin),
                  'signal': np.zeros(self.n_bin)}
        for array in self.dict_dataset.values():
            inf_mask = ~np.isinf(array['this_weight'])
            nu_e = (np.abs(array[self.true_pdg]) == 12) & inf_mask
            non_nu_e = (np.abs(array[self.true_pdg]) != 12) & inf_mask

            aux_mean_bg, _ = np.histogram(array[self.reco_energy][non_nu_e],
                                          weights=array['this_weight'][non_nu_e],
                                          bins=self.bin_edges)
            aux_mean_nue, _ = np.histogram(array[self.reco_energy][nu_e],
                                          weights=array['this_weight'][nu_e],
                                          bins=self.bin_edges)

            self.expected_bin_contents['bg'] += aux_mean_bg
            self.expected_bin_contents['signal'] += aux_mean_nue
        self.total_bg_prediction = self.expected_bin_contents['bg']

    def fillHistogramLEE(self):
        self.label = (r'Sensitivity to discover the MiniBooNE unfolded LEE')

        self.expected_bin_contents = {'bg': np.zeros(self.n_bin),
                  'signal': np.zeros(self.n_bin)}
        for array in self.dict_dataset.values():
            nu_e = (np.abs(array[self.true_pdg]) == 12)

            aux_mean_bg, _ = np.histogram(array[self.reco_energy],
                                          weights=array['this_weight'],
                                          bins=self.bin_edges)
            aux_mean_signal, _ = np.histogram(array[self.reco_energy][nu_e],
                                          weights=array['this_weight'][nu_e]*array['leeweight'][nu_e],
                                          bins=self.bin_edges)

            self.expected_bin_contents['bg'] += aux_mean_bg
            self.expected_bin_contents['signal'] += aux_mean_signal
        self.total_bg_prediction = self.expected_bin_contents['bg']

    def poissonLogLikelihood(self, mu, obs_bin_contents, pot_scale_factor):
        prediction = pot_scale_factor*(self.total_bg_prediction + mu*self.expected_bin_contents['signal'])

        likelihood_bin_by_bin = -prediction + np.where(prediction!=0, obs_bin_contents*np.log(prediction), 0)
        return likelihood_bin_by_bin.sum(axis=-1)

    def gaussianLogLikelihood(self, mu, obs_bin_contents, pot_scale_factor):
        prediction = pot_scale_factor*(self.total_bg_prediction + mu*self.expected_bin_contents['signal'])

        likelihood_bin_by_bin = np.where(prediction!=0, -0.5*(obs_bin_contents-prediction)**2/prediction - np.log(np.sqrt(prediction)), 0)
        return likelihood_bin_by_bin.sum(axis=-1)

    def gaussianLogLikelihoodApprox(self, mu, obs_bin_contents, pot_scale_factor):
        prediction = pot_scale_factor*(self.total_bg_prediction + mu*self.expected_bin_contents['signal'])

        likelihood_bin_by_bin = np.where(prediction!=0, -0.5*(obs_bin_contents-prediction)**2/prediction, 0)
        return likelihood_bin_by_bin.sum(axis=-1)

    def chi2WithCov(self, mu, obs_bin_contents, pot_scale_factor):
        prediction = pot_scale_factor*(self.total_bg_prediction + mu*self.expected_bin_contents['signal'])

        statistical_cov = np.diag(prediction)
        if self.is_cov == 'cov':
            this_cov_matrix = self.cov_matrix
        elif self.is_cov == 'frac':
            this_cov_matrix = np.einsum('i,ij,j->ij', prediction, self.cov_matrix, prediction)

        total_cov = statistical_cov + this_cov_matrix

        delta = obs_bin_contents - prediction
        if len(delta.shape) == 1:
            delta = delta.reshape(1, len(delta))
        inv_cov = np.linalg.inv(total_cov)
        chi2 = np.einsum('ij, jk, ki -> i', delta, inv_cov, delta.T)
        return chi2

    def chi2CNP(self, mu, obs_bin_contents, pot_scale_factor):
        prediction = pot_scale_factor*(self.total_bg_prediction + mu*self.expected_bin_contents['signal'])
        if len(obs_bin_contents.shape) == 1: 
            obs_bin_contents = obs_bin_contents[np.newaxis, :]
        
        aux_statistical_cov = np.where(obs_bin_contents!=0,
                            3/(1/obs_bin_contents + 2/prediction),
                            prediction/2)
        statistical_cov = np.einsum('ij,jk->ijk', aux_statistical_cov, np.identity(prediction.shape[0]))
        if self.is_cov == 'cov':
            this_cov_matrix = self.cov_matrix
        elif self.is_cov == 'frac':
            this_cov_matrix = np.einsum('i,ij,j->ij', prediction, self.cov_matrix, prediction)

        total_cov = statistical_cov + this_cov_matrix

        delta = obs_bin_contents - prediction
        if len(delta.shape) == 1:
            delta = delta.reshape(1, len(delta))
        inv_cov = np.linalg.inv(total_cov)
        chi2 = np.einsum('ij, ijk, ki -> i', delta, inv_cov, delta.T)
        return chi2
    
    def effectiveLogLikelihood(self, mu, obs_bin_contents, pot_scale_factor):
        prediction_mean = pot_scale_factor*(self.total_bg_prediction + mu*self.expected_bin_contents['signal'])
        prediction_sigma2 = (self.total_bg_sigma2 + mu**2*self.signal_sigma2)*pot_scale_factor**2
        likelihood_bin_by_bin = LEff_v(obs_bin_contents, prediction_mean, prediction_sigma2)
        return likelihood_bin_by_bin.sum(axis=-1)

    def neymanPearsonLikelihoodRatio(self, mu_0, mu_1, obs_bin_contents, pot_scale_factor):
        return -2*(self.poissonLogLikelihood(mu_0, obs_bin_contents, pot_scale_factor) - self.poissonLogLikelihood(mu_1, obs_bin_contents, pot_scale_factor))

    def gaussianLikelihoodRatio(self, mu_0, mu_1, obs_bin_contents, pot_scale_factor):
        return -2*(self.gaussianLogLikelihood(mu_0, obs_bin_contents, pot_scale_factor) - self.gaussianLogLikelihood(mu_1, obs_bin_contents, pot_scale_factor))

    def deltaChi2(self, mu_0, mu_1, obs_bin_contents, pot_scale_factor):
        return -2*(self.gaussianLogLikelihoodApprox(mu_0, obs_bin_contents, pot_scale_factor) - self.gaussianLogLikelihoodApprox(mu_1, obs_bin_contents, pot_scale_factor))

    def Chi2_mu0(self, mu_0, mu_1, obs_bin_contents, pot_scale_factor):
        return -2*(self.gaussianLogLikelihoodApprox(mu_0, obs_bin_contents, pot_scale_factor))

    def Chi2_mu1(self, mu_0, mu_1, obs_bin_contents, pot_scale_factor):
        return -2*(self.gaussianLogLikelihoodApprox(mu_1, obs_bin_contents, pot_scale_factor))

    def logRatioEffectiveLikelihood(self, mu_0, mu_1, obs_bin_contents, pot_scale_factor):
        return -2*(self.effectiveLogLikelihood(mu_0, obs_bin_contents, pot_scale_factor) - self.effectiveLogLikelihood(mu_1, obs_bin_contents, pot_scale_factor))

    def deltaChi2WithCov(self, mu_0, mu_1, obs_bin_contents, pot_scale_factor):
        return -(self.chi2WithCov(mu_1, obs_bin_contents, pot_scale_factor) - self.chi2WithCov(mu_0, obs_bin_contents, pot_scale_factor))
    
    def Chi2WithCov(self, mu_0, mu_1, obs_bin_contents, pot_scale_factor):
        return self.chi2WithCov(mu_0, obs_bin_contents, pot_scale_factor)
    
    def Chi2CNP(self, mu_0, mu_1, obs_bin_contents, pot_scale_factor):
        return self.chi2CNP(mu_0, obs_bin_contents, pot_scale_factor)
    
    def pseudoExperiments(self, n_toy, pot_scale_factor=1):
        bg_toy = []
        bg_plus_signal_toy = []
        
        bg_pred = self.total_bg_prediction
        bg_plus_signal_pred = self.total_bg_prediction + self.expected_bin_contents['signal']

        reconvert_from_list = False
        if type(pot_scale_factor) != list:
            pot_scale_factor = [pot_scale_factor]
            reconvert_from_list = True

        for pot_scale in pot_scale_factor:
            this_bg_pred = pot_scale*bg_pred
            this_bg_plus_signal_pred = pot_scale*bg_plus_signal_pred

            if self.is_cov == 'cov':
                this_cov_matrix_bg = self.cov_matrix
                this_cov_matrix_bg_plus_signal = self.cov_matrix
            elif self.is_cov == 'frac':
                this_cov_matrix_bg = np.einsum('i,ij,j->ij', this_bg_pred, self.cov_matrix, this_bg_pred)
                this_cov_matrix_bg_plus_signal = np.einsum('i,ij,j->ij', this_bg_plus_signal_pred, self.cov_matrix, this_bg_plus_signal_pred)

            bg_toy_means = np.random.multivariate_normal(this_bg_pred, 
                                                    this_cov_matrix_bg, 
                                                    size=n_toy).clip(min=0)
            bg_plus_signal_toy_means = np.random.multivariate_normal(this_bg_plus_signal_pred, 
                                                            this_cov_matrix_bg_plus_signal, 
                                                            size=n_toy).clip(min=0)
            bg_toy.append(np.random.poisson(bg_toy_means))
            bg_plus_signal_toy.append(np.random.poisson(bg_plus_signal_toy_means))

        if reconvert_from_list:
            bg_toy = bg_toy[0]
            bg_plus_signal_toy = bg_plus_signal_toy[0]

        return bg_toy, bg_plus_signal_toy

    def pvalue2sigma(self, pvalue):
        return mpmath.sqrt(2) * mpmath.erfinv(1 - 2*pvalue)

    def significanceCalculation(self, test_stat_mu0, test_stat_mu1, percentage_values=[16, 50, 84]):
        expected_quantiles = np.percentile(test_stat_mu1, percentage_values)
        expected_pvalues = []
        expected_one_minus_pvalues = []
        expected_significance = []

        for percentage_value, quantile in zip(percentage_values, expected_quantiles):
            one_minus_pvalue = np.less(test_stat_mu0, quantile).sum()/len(test_stat_mu0)
            pvalue = 1. - one_minus_pvalue
            if pvalue != 0:
                significance = float(self.pvalue2sigma(pvalue))
            if pvalue == 0:
                mean = np.mean(test_stat_mu0)
                std = np.std(test_stat_mu0)
                significance = (quantile - mean)/std

            expected_one_minus_pvalues.append(one_minus_pvalue)
            expected_pvalues.append(pvalue)
            expected_significance.append(significance)
        return expected_significance, expected_pvalues, expected_quantiles

    def asimovPlot(self, mu, mu_0, mu_1, pot_scale_factor=1, title=None, ntoy=1000, test_stat='pois_llr', split_bg=False):
        assert self.bin_edges is not None
        bin_centers = (self.bin_edges[1:] + self.bin_edges[:-1])/2
        bin_widths = (self.bin_edges[1:] - self.bin_edges[:-1])
        bg_prediction = self.total_bg_prediction
        signal_prediction = self.expected_bin_contents['signal']

        #plot two hypotheses
        total_bin_contents = np.zeros(bg_prediction.shape)
        if split_bg:
            for name, contents in self.expected_bin_contents.items():
                if 'signal' in name:
                    continue
                plt.bar(bin_centers,
                        pot_scale_factor*contents,
                        bottom=total_bin_contents,
                        width=bin_widths,
                        label=r'$H_0$: ' + name,
                        )
                total_bin_contents += pot_scale_factor*contents
        else:
            plt.bar(bin_centers,
                    pot_scale_factor*(bg_prediction + mu_0*signal_prediction),
                    width=bin_widths,
                    label=r'$H_0$: $\mu$ = {}'.format(mu_0),
                    )
            total_bin_contents += pot_scale_factor*(bg_prediction + mu_0*signal_prediction)
        plt.bar(bin_centers,
                pot_scale_factor*(mu_1-mu_0)*signal_prediction,
                bottom=total_bin_contents,
                width=bin_widths,
                label=r'oscillated nue',
                )

        # plot pseudo data
        bg_toy, bg_plus_signal_toy = self.pseudoExperiments(n_toy=ntoy, pot_scale_factor=pot_scale_factor)
        test_stat_mu0 = self.ts_dict[test_stat](mu_0,
                                                mu_1,
                                                bg_toy,
                                                pot_scale_factor)

        test_stat_mu1 = self.ts_dict[test_stat](mu_0,
                                                mu_1,
                                                pot_scale_factor*(self.total_bg_prediction+self.expected_bin_contents['signal']),
                                                pot_scale_factor)
        
        expected_significance, expected_pvalues, expected_quantiles = self.significanceCalculation(test_stat_mu0, test_stat_mu1, percentage_values=[50])
        pseudo_data = pot_scale_factor*(self.total_bg_prediction + mu_1*self.expected_bin_contents['signal'])
        plt.plot(bin_centers,
                 pseudo_data,
                 'k.',
                 label=r'Asimov dataset $\mu$ = {}'.format(mu) + '\n' + r'signifcance = {:.2f} $\sigma$'.format(expected_significance[0]),
                 )


        plt.legend()
        plt.ylabel("Expected number of entries")
        plt.xlabel("Reconstructed energy [GeV]")
        if title is None:
            title = self.selection_label + '\n' + self.label
        plt.title(title+"\nMicroBooNE Preliminary - {:.2g} POT".format(pot_scale_factor*self.pot), loc='left')

    def testStatisticsPlot(self, mu_0, mu_1, n_toy=100000, percentage_values=[16, 50, 84], pot_scale_factor=1, n_bins=50, log=False, title=None, test_stat='pois_llr', range=None, print_numbers=True):
        bg_toy, bg_plus_signal_toy = self.pseudoExperiments(n_toy, pot_scale_factor)
        toy_mu0 = bg_toy
        toy_mu1 = bg_plus_signal_toy
        test_stat_mu0 = self.ts_dict[test_stat](mu_0, mu_1, toy_mu0, pot_scale_factor)
        test_stat_mu1 = self.ts_dict[test_stat](mu_0, mu_1, toy_mu1, pot_scale_factor)
        expected_significance, expected_pvalues, expected_quantiles = self.significanceCalculation(test_stat_mu0, test_stat_mu1, percentage_values)

        bin_contents_total, bin_edges, _ = plt.hist(
                                             [test_stat_mu0, test_stat_mu1],
                                             bins=n_bins,
                                             range=range,
                                             density=True,
                                             label=[r'$\mu$ = {:.2g}'.format(mu_0), r'$\mu$ = {:.2g}'.format(mu_1)],
                                             alpha=0.7,
                                             histtype='stepfilled',
                                             lw=2,
                                             log=log,
                                             )

        bin_width = bin_edges[1] - bin_edges[0]
        plt.ylabel("Probability / {:.2f}".format(bin_width))
        plt.xlabel(self.ts_labels[test_stat])
        ax = plt.gca()
        ax.set_xlim(test_stat_mu1.mean() - 5*test_stat_mu1.std(), test_stat_mu1.mean() + 5*test_stat_mu1.std())
        ymax = ax.get_ylim()[1]
        heights = {16: 0.5, 50: 0.7, 84: 0.5}
        horizontalalignments = {16: 'right', 50: 'center', 84: 'left'}
        position_offset = {16: -5, 50: 0, 84: +5}

        for i, percentage_value in enumerate(percentage_values):
            quantile = expected_quantiles[i]
            pvalue = expected_pvalues[i]
            significance = expected_significance[i]

            plt.axvline(quantile, ymax=heights[percentage_value]-0.1, color='red', linestyle='--', label='expected {}%'.
                        format(percentage_value))
            if print_numbers:
                plt.text(quantile+position_offset[percentage_value],
                         heights[percentage_value]*ymax,
                         'p = {:.1e}\nZ = {:.2f}'.format(pvalue, significance)+r'$\sigma$',
                         fontsize=10,
                         verticalalignment='center',
                         horizontalalignment=horizontalalignments[percentage_value],
                         )

        handles, labels = ax.get_legend_handles_labels()
        # ax.legend(handles[::-1], labels[::-1], bbox_to_anchor=(1.04, 0.5))
        ax.legend(handles[::-1], labels[::-1], loc="best")

        props = dict(boxstyle='round', facecolor='white', alpha=0.5, linewidth=0.5)
        # plt.text(1.08, 0.5, textstr, transform=ax.transAxes, fontsize=10, verticalalignment='top', bbox=props)
        if title is None:
            title = self.selection_label + '\n' + self.label
        plt.title(title+"\nMicroBooNE Preliminary - {:.2g} POT".format(pot_scale_factor*self.pot), loc='left')
        plt.tight_layout()
        return expected_significance

    def testStatisticsWithData(self, mu_0, mu_1, observed_data, chi2_pdf_superimposed=True, n_toy=100000, pot_scale_factor=1, n_bins=50, log=False, title=None, test_stat='pois_llr', range=None, print_numbers=True):
        
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))

        # plot toy experiment
        bg_toy, bg_plus_signal_toy = self.pseudoExperiments(n_toy, pot_scale_factor)
        toy_mu0 = bg_toy
        test_stat_mu0 = self.ts_dict[test_stat](mu_0, mu_1, toy_mu0, pot_scale_factor)
        
        bin_contents_total, bin_edges, _ = ax[0].hist(
                                             test_stat_mu0,
                                             bins=n_bins,
                                             range=range,
                                             density=True,
                                             label=r'TS distribution with toy experiments',
                                             alpha=0.7,
                                             histtype='stepfilled',
                                             lw=2,
                                             log=log,
                                             )

        # plot chi2 expected distribution
        ndof = self.n_bin
        chi2pdf = sp.stats.chi2(ndof)
        x = np.linspace(0, ndof+10*np.sqrt(ndof), 100)
        if chi2_pdf_superimposed:
            ax[0].plot(x, chi2pdf.pdf(x), 'k-', lw=2, label=r'$\chi^2$ with {} dof'.format(ndof))

        # observed value
        test_stat_observed = self.ts_dict[test_stat](mu_0, mu_1, observed_data, pot_scale_factor)
        # compute pvalues
        test_stat_pvalue = (test_stat_mu0 >= test_stat_observed).sum() / n_toy
        chi2_pvalue = 1 - chi2pdf.cdf(test_stat_observed)

        # plot observed value
        ax[0].axvline(test_stat_observed, ymax=0.5, color='red', linestyle='-', label=f'Obbserved value = {test_stat_observed[0]:.3g}\n'+
                                                                         f'p-value with $\chi^2_{{{ndof}}}$ = {chi2_pvalue[0]:.3g}' +\
                                                                         f'\np-value with toys = {test_stat_pvalue:.3g}\n')
        
        ax[0].set_ylabel("PDF value")
        ax[0].set_xlabel(self.ts_labels[test_stat])
        ax[0].legend(loc='best', frameon=False)
        ax[0].set_title("MicroBooNE Preliminary - {:.2g} POT".format(pot_scale_factor*self.pot), loc='right')
        if title is not None:
            ax[0].set_title(title, loc='left')
        
        # take the p-value toy vs p-value chi2
        p_values_chi2 = 1 - chi2pdf.cdf(x)
        p_values_toys = (test_stat_mu0 >= x[:, np.newaxis]).sum(axis=1) / len(test_stat_mu0)
        
        ax[1].plot(p_values_chi2, p_values_toys)
        ax[1].set_ylabel("p-value using toy epxeriments")
        ax[1].set_xlabel(f"p-value wrt $\chi^2_{{{ndof}}}$")
        ax[1].set_xlim(0, 1)
        ax[1].set_ylim(0, 1)
        ax[1].set_title("MicroBooNE Preliminary - {:.2g} POT".format(pot_scale_factor*self.pot), loc='right')
        if title is not None:
            ax[1].set_title(title, loc='left')
        fig.tight_layout()
        
        return fig, ax
    
    def SignificanceFunctionScaleFactors(self, mu_0, mu_1, n_toy=100000, percentage_values=[16, 50, 84], pot_scale_factors=[1], label='', title='', type='discovery', test_stat='pois_llr'):
        expectation = []
        bg_toy, bg_plus_signal_toy = self.pseudoExperiments(n_toy, pot_scale_factors)

        for i, pot_scale_factor in enumerate(pot_scale_factors):
            toy_mu0 = bg_toy[i]
            toy_mu1 = bg_plus_signal_toy[i]
            test_stat_mu0 = self.ts_dict[test_stat](mu_0, mu_1, toy_mu0, pot_scale_factor)
            test_stat_mu1 = self.ts_dict[test_stat](mu_0, mu_1, toy_mu1, pot_scale_factor)
            aux_significance, aux_pvalues, aux_quantiles = self.significanceCalculation(test_stat_mu0, test_stat_mu1, percentage_values)
            if type == 'discovery':
                expectation.append(aux_significance)
            elif type == 'exclusion':
                expectation.append(aux_pvalues)

        expectation = np.array(expectation)
        print(expectation)
        if type == 'exclusion':
            expectation = 100*(1.-expectation)

        x_axis_labels = self.pot*np.array(pot_scale_factors)
        plt.plot(x_axis_labels, expectation[:, 1], label=label)
        plt.fill_between(x_axis_labels, expectation[:, 0], expectation[:, 2],
            alpha=0.2,
            linewidth=0, antialiased=True, label='expected {:.2g}%'.format(percentage_values[2] - percentage_values[0]))
        plt.legend()
        if type == 'discovery':
            plt.ylabel(r'Expected significance [$\sigma$]')
        elif type == 'exclusion':
            plt.ylabel(r'Esclusion Confidence Level [$\%$]')
        plt.xlabel('Collected POT')
        plt.title(title, loc='left')
        plt.title(title+"\nMicroBooNE Preliminary", loc='left')
        plt.tight_layout()

    def muExtraction(self, mu_test, mu_range, pot_scale_factor):
        obs_bin_contents = pot_scale_factor*(self.total_bg_prediction + mu_test*self.expected_bin_contents['signal'])

        aux_likelihoods = []
        likelihood_offset = self.poissonLogLikelihood(mu_test, obs_bin_contents, pot_scale_factor)
        for mu in mu_range:
            aux_likelihoods.append(-self.poissonLogLikelihood(mu, obs_bin_contents, pot_scale_factor) + likelihood_offset)

        plt.plot(mu_range, aux_likelihoods, label='POT = {:.1g} POT'.format(pot_scale_factor*self.pot))
        plt.legend()
        return aux_likelihoods
