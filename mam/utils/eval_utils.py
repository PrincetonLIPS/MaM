import torch
import numpy as np
from scipy import stats

def create_epoch_metrics():
    epoch_metrics = {
            'log_ll_mam': 0,
            'log_ll_e': 0,
            'log_ll_s': 0,
            'log_ll_err': 0,
            'log_ll_err_var': 0,
            'spearman': 0,
            'spearman_pvalue': 0,
            'spearman_s': 0,
            'spearman_s_pvalue': 0,
            'pearson': 0,
            'pearson_pvalue': 0,
            'pearson_s': 0,
            'pearson_s_pvalue': 0,
            'count': 0,
        }
    return epoch_metrics

def update_epoch_metrics(epoch_metrics, cmp_results, count):
    epoch_metrics['spearman'] += cmp_results['spearman'].correlation * count
    epoch_metrics['spearman_pvalue'] += cmp_results['spearman'].pvalue * count
    epoch_metrics['spearman_s'] += cmp_results['spearman_self'].correlation * count
    epoch_metrics['spearman_s_pvalue'] += cmp_results['spearman_self'].pvalue * count
    epoch_metrics['pearson'] += cmp_results['pearson'][0] * count
    epoch_metrics['pearson_pvalue'] += cmp_results['pearson'][1] * count
    epoch_metrics['pearson_s'] += cmp_results['pearson_self'][0] * count
    epoch_metrics['pearson_s_pvalue'] += cmp_results['pearson_self'][1] * count
    epoch_metrics['log_ll_mam'] += cmp_results['logp_mam'] * count
    epoch_metrics['log_ll_e'] += cmp_results['logp'] * count
    epoch_metrics['log_ll_s'] += cmp_results['logp_2'] * count
    epoch_metrics['log_ll_err'] += cmp_results['logp_err'] * count
    epoch_metrics['log_ll_err_var'] += cmp_results['logp_err_var'] * count
    epoch_metrics['count'] += count
    return

def preprocess_logp(logp, threshold=1.0):
    logp_sorted, indices_arm = torch.sort(logp, dim=-1, descending=True)
    select_ordered_ind = [0]
    prev_logp = logp_sorted[0]
    for i in range(1, logp.shape[0]):
        diff =  prev_logp - logp_sorted[i]
        if diff > threshold:
            select_ordered_ind.append(i)
            prev_logp = logp_sorted[i]
    selected_ind = indices_arm[select_ordered_ind]
    return selected_ind

def compare_logp(logp, logp_2, logp_mam, log_z):
    pearson = stats.pearsonr(logp_mam.cpu().numpy(), logp.cpu().numpy())
    pearson_s = stats.pearsonr(logp.cpu().numpy(), logp_2.cpu().numpy())
    spearman = stats.spearmanr(logp_mam.cpu().numpy(), logp.cpu().numpy())
    spearman_s = stats.spearmanr(logp.cpu().numpy(), logp_2.cpu().numpy())
    logp_err = (logp - logp_mam + log_z).abs().mean()
    logp_err_var = (logp - logp_mam + log_z).var()

    # put results into a dictionary
    results = {
        'logp_mam': logp_mam.mean().item(),
        'logp': logp.mean().item(),
        'logp_2': logp_2.mean().item(),
        'pearson': pearson,
        'pearson_self': pearson_s,
        'spearman': spearman,
        'spearman_self': spearman_s,
        'logp_err': logp_err,
        'logp_err_var': logp_err_var}

    return results