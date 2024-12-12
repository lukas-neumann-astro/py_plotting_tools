import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib import gridspec
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from leastsq import leastsq
from scipy.stats import pearsonr, spearmanr
import scipy.stats
import linmix
import cloud_pdf_model

def plot_binned_data(glxys, xdata_list, ydata_list, xdata_err_list, ydata_err_list, xlabel, ylabel, xrange, yrange,
                     xticks_list=None, yticks_list=None, SNR=3, ploterr=False, ydata_upplim=None, ydata_lowlim=None, 
                     plotupplim=False, plotlowlim=False, fit=False, MC_iter=10000, fitcens=True, showparam=False,
                     showcorr=True, savename=None, saveplot=False, showtitle=False, showplot=True, cyclepos=None,
                     cloud_scale_pc=None, model_prediction=False, x_model=None, y_model=None, annotation=False):
    """
    Takes a list of galaxies and data as multidimensional array and plots the data sets in a multiplot matching the 
    data array shape. The first two dimensions of the data array specify the subplot grid and the third dimension 
    matches the length of the galaxy list. It also takes an ndarray of labels and plot ranges matching the shape of 
    the data arrays to set-up the subplots. A signal-to-noise ratio threshold can be specified and decided whether 
    data below the threshold is plotted. In addition the data can be fitted by an OLS bisector method. Finally the
    plot can be saved to file.
    
    INPUT:
    - glxys: list of galaxies
    - xdata_list: list of x-data arrays
    - ydata_list: list of y-data arrays
    - 
    - cloud-scale in parsec or arcsec
    
    OUPUT:
    - None (except plot to file)
    """

    # set marker styles and colors to cylcle in plots
    color_cycle = ('r','b','darkorange','steelblue','hotpink','gold','c','maroon','darkgreen','darkviolet', 'darkkhaki')
    marker_cycle = ('o','D','^')
    
    # markersize
    ms = 4

    # constants to compute surface densities and dense gas fraction
    alpha_co10 = 4.3  # [M_Sun/pc^2/(K km/s)] (Bolatto et al. 2013)
    alpha_co21 = 1/0.64 * alpha_co10  # (den Brok et al. 2021)
    alpha_HCN = 14  # [M_Sun/pc^2/(K km/s)] (Onus et al. 2018)
    c_dense = alpha_HCN/alpha_co21
    
    # secondary x-axis for CO(2-1) from Sigma_mol
    def ICO21_log_function(Sigmol_log):
        return Sigmol_log - np.log10(alpha_co21)
    def Sigmol_log_function(ICO21_log):
        return ICO21_log + np.log10(alpha_co21)
    
    # secondary x-axis for sigma^2/Sigma from alpha_vir
    if cloud_scale_pc is not None:
        conversion_avir = 3.1 * 150 / cloud_scale_pc  # 3.1 for 150 pc cloud-scale
        def sigSigratio_log_function(avir):
            return avir - np.log10(conversion_avir)
        def avir_log_function(sigSigratio):
            return sigSigratio + np.log10(conversion_avir)

    # secondary y-axis for fdense from HCN/CO
    def f_dense_log_function(int_ratio_log):
        return int_ratio_log + np.log10(c_dense)
    def int_ratio_log_function(f_dense_log):
        return f_dense_log - np.log10(c_dense)
    
    # secondary y-axis for SFE from SFR/HCN
    def Sigma_SFR_Sigma_HCN_function(Sigma_SFR_I_HCN_log):
        return Sigma_SFR_I_HCN_log - np.log10(alpha_HCN)  # units: Myr^-1
    def Sigma_SFR_I_HCN_log_function(Sigma_SFR_Sigma_HCN):
        return Sigma_SFR_Sigma_HCN + np.log10(alpha_HCN)
    
    # get rows, columns and number of plots
    rows = len(ydata_list)
    cols = len(xdata_list)
    n_plots = rows * cols

    # plot size
    aspect_ratio = 1
    if cols > 2:
        width = 7.5
    else:
        width = 3.5
    height = width / aspect_ratio / cols * rows

    # make LaTeX compatible plot
    hp.homogenise_plot(fig_width=width,    # width of the plot; 3.5 is default for a two-column LaTeX article document
                       fig_height=height,  # height of the plot; if None, golden ratio is adopted
                       )         
    
    # box for labels
    box = dict(boxstyle='round', facecolor='white', edgecolor='grey', linewidth=0, alpha=1)  # set box
    
    # uniform arrow properties for annotate
    arrowprops = dict(arrowstyle="-|>,head_width=0.15,head_length=0.3", color='k', shrinkA=0, shrinkB=0, lw=1)
    
    # make figure
    fig = plt.figure()
    fig.subplots_adjust(hspace=0, wspace=0)  # subplots with adjacent axis
    
    # convert upper limit and lower limit bool to correct shape if given as False
    if plotupplim is False:
        plotupplim = np.ones((rows,cols)) * False
    if plotlowlim is False:
        plotlowlim = np.ones((rows,cols)) * False
    if fit is False:
        fit = np.ones((rows,cols)) * False
    elif fit is True:
        fit = np.ones((rows, cols)) * True
    
    # title
    if showtitle:
        # get resolutions from savename
        cloud_res = savename.split('_')[-2]
        large_res = savename.split('_')[-7]
        res_config = savename.split('_')[-1].split('.')[0]
        # make title
        if (cloud_res[-2:] == 'as') & (large_res[-2:] == 'as'):
            plt.suptitle('cloud-scale = %s", large-scale = %s" (%s)' % (cloud_res[:-2], large_res[:-2], res_config), y=0.96)
        if (cloud_res[-2:] == 'pc') & (large_res[-2:] == 'pc'):
            plt.suptitle('cloud-scale = %s pc, large-scale = %s pc (%s)' % (cloud_res[:-2], large_res[:-2], res_config), y=0.96)

    # loop over subplots
    for i in range(1, n_plots+1):

        # make subplot
        ax = fig.add_subplot(rows, cols, i)

        # row and column indices
        id_row = (i-1) // cols
        id_col = (i-1) % cols

        # plot range
        xlim = xrange[id_col]
        ylim = yrange[id_row]

        # arrays to store data (for all galaxies combined)
        x_det_all, y_det_all, x_det_all_err, y_det_all_err = [],[],[],[]  # detections
        x_ul_all, y_ul_all, x_ul_all_err, y_ul_all_err = [],[],[],[]      # upper limits
        x_ll_all, y_ll_all, x_ll_all_err, y_ll_all_err = [],[],[],[]      # lower limits

        # SUBPLOTS
        # loop over galaxies and plot data
        for glxy in glxys:

            # loop element index
            id_glxy = glxys.index(glxy)

            # color and marker cycle index
            if cyclepos != None:
                if type(cyclepos) == int:
                    c = cyclepos
                elif type(cyclepos) == list:
                    c = cyclepos[id_glxy]
                else:
                    print('[ERROR] Variable "cyclepos" has wrong type!')
            else:
                c = id_glxy

            # get data
            x = xdata_list[id_col][id_glxy]
            y = ydata_list[id_row][id_glxy]
            x_err = xdata_err_list[id_col][id_glxy]
            y_err = ydata_err_list[id_row][id_glxy]
            y_ul = ydata_upplim[id_row][id_glxy]  # upper limit
            y_ll = ydata_lowlim[id_row][id_glxy]  # lower limit
                
            # make data copy for upper and/or lower limits
            x_ul, x_ll = np.copy(x), np.copy(x)
            x_ul_err, x_ll_err = np.copy(x_err), np.copy(x_err)
            y_ul_err, y_ll_err = np.copy(y_err), np.copy(y_err)

            # remove nan values
            id_nonnan = np.where(~np.isnan(x) & ~np.isnan(y) & ~np.isnan(x_err) & ~np.isnan(y_err))
            x = x[id_nonnan]
            y = y[id_nonnan]
            x_err = x_err[id_nonnan]
            y_err = y_err[id_nonnan]
            
            id_nonnan_ul = np.where(~np.isnan(x_ul) & ~np.isnan(y_ul) & ~np.isnan(x_ul_err) & ~np.isnan(y_ul_err))
            x_ul = x_ul[id_nonnan_ul]
            y_ul = y_ul[id_nonnan_ul]
            x_ul_err = x_ul_err[id_nonnan_ul]
            y_ul_err = y_ul_err[id_nonnan_ul]
            
            id_nonnan_ll = np.where(~np.isnan(x_ll) & ~np.isnan(y_ll) & ~np.isnan(x_ll_err) & ~np.isnan(y_ll_err))
            x_ll = x_ll[id_nonnan_ll]
            y_ll = y_ll[id_nonnan_ll]
            x_ll_err = x_ll_err[id_nonnan_ll]
            y_ll_err = y_ll_err[id_nonnan_ll]

            # remove non-positive values
            id_pos = (x > 0) & (y > 0) & (x_err > 0) & (y_err > 0)
            x = x[id_pos]
            y = y[id_pos]
            x_err = x_err[id_pos]
            y_err = y_err[id_pos]
            
            id_pos_ul = (x_ul > 0) & (y_ul > 0) & (x_ul_err > 0) & (y_ul_err > 0)
            x_ul = x_ul[id_pos_ul]
            y_ul = y_ul[id_pos_ul]
            x_ul_err = x_ul_err[id_pos_ul]
            y_ul_err = y_ul_err[id_pos_ul]
            
            id_pos_ll = (x_ll > 0) & (y_ll > 0) & (x_ll_err > 0) & (y_ll_err > 0)
            x_ll = x_ll[id_pos_ll]
            y_ll = y_ll[id_pos_ll]
            x_ll_err = x_ll_err[id_pos_ll]
            y_ll_err = y_ll_err[id_pos_ll]
            
            # above SNR threshold
            id_highSNR = (y/y_err) >= SNR
            x1 = x[id_highSNR] 
            y1 = y[id_highSNR]
            x1_err = x_err[id_highSNR]
            y1_err = y_err[id_highSNR]

            # below SNR threshold
            id_lowSNR = (y/y_err) < SNR
            x2 = x[id_lowSNR] 
            y2 = y[id_lowSNR]
            x2_err = x_err[id_lowSNR]
            y2_err = y_err[id_lowSNR]

            # convert to log-log space
            x1_log = np.log10(x1)
            y1_log = np.log10(y1)
            x1_err_up_log = np.abs(np.log10(1 + x1_err/x1))
            x1_err_dw_log = np.abs(np.log10(1 - x1_err/x1))
            y1_err_up_log = np.abs(np.log10(1 + y1_err/y1))
            y1_err_dw_log = np.abs(np.log10(1 - y1_err/y1))
            #x1_err_dex = np.abs(np.log10(1 + x1_err/x1))
            #y1_err_dex = np.abs(np.log10(1 + y1_err/y1))

            x2_log = np.log10(x2)
            y2_log = np.log10(y2)
            x2_err_up_log = np.abs(np.log10(1 + x2_err/x2))
            x2_err_dw_log = np.abs(np.log10(1 - x2_err/x2))
            y2_err_up_log = np.abs(np.log10(1 + y2_err/y2))
            y2_err_dw_log = np.abs(np.log10(1 - y2_err/y2))
            #x2_err_dex = np.abs(np.log10(1 + x2_err/x2))
            #y2_err_dex = np.abs(np.log10(1 + y2_err/y2))
            
            x_ul_log = np.log10(x_ul)
            y_ul_log = np.log10(y_ul)
            x_ul_err_dex = np.full_like(x_ul_log, np.log10(1+2/3))  # 2-sigma uncertainty relative to the upper limit
            y_ul_err_dex = np.full_like(y_ul_log, np.log10(1+2/3))
            #x_ul_err_dex = np.abs(np.log10(1 + x_ul_err/x_ul))
            #y_ul_err_dex = np.abs(np.log10(1 + y_ul_err/y_ul))

            x_ll_log = np.log10(x_ll)
            y_ll_log = np.log10(y_ll)
            x_ll_err_dex = np.full_like(x_ll_log, np.log10(1+2/3))  # 2-sigma uncertainty relative to the upper limit
            y_ll_err_dex = np.full_like(y_ll_log, np.log10(1+2/3))
            #x_ll_err_dex = np.abs(np.log10(1 + x_ll_err/x_ll))
            #y_ll_err_dex = np.abs(np.log10(1 + y_ll_err/y_ll))

                    
            # plot binned data above SNR threshold filled
            if ploterr:
                ax.errorbar(x1_log, y1_log, yerr=(y1_err_dw_log, y1_err_up_log), xerr=(x1_err_dw_log, x1_err_up_log), linestyle='none', 
                            capsize=1.5, ecolor=color_cycle[c%11], elinewidth=1, capthick=1, zorder=5)
                
            ax.plot(x1_log, y1_log, linestyle='none', marker=marker_cycle[c%3], markerfacecolor=color_cycle[c%11], 
                    markeredgecolor='k', label=glxy[3:], zorder=6, markersize=ms, markeredgewidth=0.5)

                
            # plot upper limits
            if plotupplim[id_row]:
                (_, caps_ul, _) = ax.errorbar(x_ul_log, y_ul_log, yerr=abs(ylim[0]-ylim[1])/20, uplims=True, linestyle='none',
                                           marker='None', elinewidth=1, color=color_cycle[c%11], zorder=4, alpha=0.5)
                for cap in caps_ul:
                    cap.set_markersize(ms)
                    cap.set_markeredgewidth(0)
                
            # plot lower limits
            if plotlowlim[id_row]:
                (_, caps_ll, _) = ax.errorbar(x_ll_log, y_ll_log, yerr=abs(ylim[0]-ylim[1])/20, lolims=True, linestyle='none',
                                              marker='None', elinewidth=1, color=color_cycle[c%11], zorder=4, alpha=0.5)
                for cap in caps_ll:
                    cap.set_markersize(ms)
                    cap.set_markeredgewidth(0)
                
            # concatenate data to combined arrays; use the downward log uncertainty as the dex uncertainty
            x_det_all, y_det_all = np.concatenate((x_det_all, x1_log)), np.concatenate((y_det_all, y1_log))
            x_det_all_err, y_det_all_err = np.concatenate((x_det_all_err, x1_err_dw_log)), np.concatenate((y_det_all_err, y1_err_dw_log))
            
            x_ul_all, y_ul_all = np.concatenate((x_ul_all, x_ul_log)), np.concatenate((y_ul_all, y_ul_log)) 
            x_ul_all_err, y_ul_all_err = np.concatenate((x_ul_all_err, x_ul_err_dex)), np.concatenate((y_ul_all_err, y_ul_err_dex))
            
            x_ll_all, y_ll_all = np.concatenate((x_ll_all, x_ll_log)), np.concatenate((y_ll_all, y_ll_log))
            x_ll_all_err, y_ll_all_err = np.concatenate((x_ll_all_err, x_ll_err_dex)), np.concatenate((y_ll_all_err, y_ll_err_dex))

            
        # get Spearman rank correlation and corresponding p-value
        try:
            rank_corr, rank_p_value = spearmanr(x_det_all, y_det_all)
        except:
            rank_corr, rank_p_value = np.nan, np.nan
            
        # get Pearson correlation coefficient and corresponding p-value
        try:
            pearson_corr, pearson_p_value = pearsonr(x_det_all, y_det_all)
        except:
            pearson_corr, pearson_p_value = np.nan, np.nan

        
        if fit[id_row][id_col]:

            # get censored data
            if ylabel[id_row][:6] == 'fdense':
                x_cens = x_ul_all
                y_cens = y_ul_all
                x_cens_err = x_ul_all_err
                y_cens_err = y_ul_all_err
                fit_method = 'upplim'
                
            elif ylabel[id_row][:3] == 'SFE':
                x_cens = x_ll_all
                y_cens = y_ll_all
                x_cens_err = x_ll_all_err
                y_cens_err = y_ll_all_err
                fit_method = 'lowlim'
                
            # set x-offset where the interception should be measured
            if xlabel[id_col] == 'Sigmol_avg':
                x_off = 2.5
            elif xlabel[id_col] == 'vdis_avg':
                x_off = 1.1
            elif xlabel[id_col] == 'avir_avg':
                x_off = 0.2
            elif xlabel[id_col] == 'Ptur_avg':
                x_off = 6.5
                
            # plot vertical line to indicate intercept reference
            #ax.axvline(x_off, lw=1 , c='k', ls='-', zorder=2)
            
            # make x-list for plotting credibility intervals
            x_cre = np.linspace(xlim[0], xlim[1], 50)

            #fit_method = None
            out_linmix = fit_linmix(x_det_all-x_off, y_det_all, x_det_err=x_det_all_err, y_det_err=y_det_all_err, 
                                    x_cens=x_cens-x_off, y_cens=y_cens, x_cens_err=x_cens_err, y_cens_err=y_cens_err, 
                                    method=fit_method, x_cre=x_cre-x_off, K=3, seed=7, MC_iter=MC_iter, debug=False, 
                                    fitcens=fitcens)
            
            if out_linmix is not None:
                slope, slope_unc, intercept, intercept_unc, corr, corr_unc, scatter_intrinsic, scatter_intrinsic_unc, stats = out_linmix
                scatter_residuals = stats['scatter_residuals_y']
            else:
                slope, slope_unc, intercept, intercept_unc = np.nan,(np.nan,np.nan), np.nan,(np.nan,np.nan)
                corr, corr_unc, scatter_intrinsic, scatter_intrinsic_unc = np.nan,(np.nan,np.nan), np.nan,(np.nan,np.nan)
                stats = dict.fromkeys(['p_value', 'scatter_residuals', '-1sigma', '+1sigma', '-2sigma', '+2sigma', '-3sigma', '+3sigma'])
                stats['p_value'], stats['scatter_residuals'], stats['-1sigma'], stats['+1sigma'] = np.nan, np.nan, np.nan, np.nan
                stats['-2sigma'], stats['+2sigma'], stats['-3sigma'], stats['+3sigma'] = np.nan, np.nan, np.nan, np.nan

            # plot linmix median line fit
            x_fit = np.linspace(xlim[0], xlim[1], 10)
            y_fit = intercept + slope * (x_fit-x_off)
            ax.plot(x_fit, y_fit, 'k-', lw=3, zorder=3)
            
            # scatter of the residuals about the median fit line
            y_scatter = scatter_residuals
            y_sca_n = intercept + slope * (x_fit-x_off) - y_scatter
            y_sca_p = intercept + slope * (x_fit-x_off) + y_scatter
            ax.fill_between(x_fit, y_sca_n, y_sca_p, color='grey', lw=0.1, alpha=0.5, zorder=2)
            
            # credibility areas of fit parameters
            if out_linmix is not None:
                #ax.fill_between(x_cre, stats['-1sigma'], stats['+1sigma'], color='grey', lw=0.1, alpha=0.8, zorder=2)
                #ax.fill_between(x_cre, stats['-2sigma'], stats['+2sigma'], color='grey', lw=0.1, alpha=0.5, zorder=2)
                #ax.fill_between(x_cre, stats['-3sigma'], stats['+3sigma'], color='grey', lw=0.1, alpha=0.3, zorder=2)
                ax.plot(x_cre, stats['-1sigma'], color='k', ls='dashed', lw=1, zorder=3)
                ax.plot(x_cre, stats['+1sigma'], color='k', ls='dashed', lw=1, zorder=3)
                #ax.plot(x_cre, stats['-2sigma'], color='k', ls='dotted', lw=0.5, zorder=3)
                #ax.plot(x_cre, stats['+2sigma'], color='k', ls='dotted', lw=0.5, zorder=3)
                #ax.plot(x_cre, stats['-3sigma'], color='k', ls='dashdot', lw=0.5, zorder=3)
                #ax.plot(x_cre, stats['+3sigma'], color='k', ls='dashdot', lw=0.5, zorder=3)
            
            # p_value
            p_value = stats['p_value']
            
            # plot fit parameter results
            if showparam:
                if slope < 0:
                    text = 'slope = %.2f (%.2f)\ninterc. = %.2f (%.2f)\nscatter = %.2f\n$r$ ($p$) = %.2f (%.3f)' \
                           % (slope, np.nanmean(np.abs(slope_unc)), intercept, np.nanmean(np.abs(intercept_unc)), scatter_residuals, corr, p_value)
                    ax.text(0.96, 0.96, text, ha='right', va='top', transform = ax.transAxes, bbox=box, zorder=7)
                else:
                    text = 'slope = %.2f (%.2f)\ninterc. = %.2f (%.2f)\nscatter = %.2f\n$r$ ($p$) = %.2f (%.3f)' \
                           % (slope, np.nanmean(np.abs(slope_unc)), intercept, np.nanmean(np.abs(intercept_unc)), scatter_residuals, corr, p_value)
                    ax.text(0.96, 0.04, text, ha='right', va='bottom', transform = ax.transAxes, bbox=box, zorder=7) 
            elif showcorr:
                text = r'$\rho = %.2f$' % corr
                if id_row == 0:
                    ax.text(0.95, 0.05, text, ha='right', va='bottom', transform=ax.transAxes, bbox=box, zorder=7)
                else:
                    ax.text(0.95, 0.95, text, ha='right', va='top', transform=ax.transAxes, bbox=box, zorder=7)
                
        else:
            if showparam:
                if id_row == 0:
                    # plot Person r and p-value
                    text = '$r$ ($p$) = %.2f (%.3f)' % (pearson_corr, pearson_p_value)
                    ax.text(0.96, 0.04, text, ha='right', va='bottom', transform = ax.transAxes, bbox=box, zorder=7)
                else:
                    # plot Person r and p-value
                    text = '$r$ ($p$) = %.2f (%.3f)' % (pearson_corr, pearson_p_value)
                    ax.text(0.96, 0.96, text, ha='right', va='top', transform = ax.transAxes, bbox=box, zorder=7)       
            elif showcorr:
                text = r'$\rho = %.2f$' % pearson_corr
                if id_row == 0:
                    ax.text(0.95, 0.05, text, ha='right', va='bottom', transform=ax.transAxes, bbox=box, zorder=7)
                else:
                    ax.text(0.95, 0.95, text, ha='right', va='top', transform=ax.transAxes, bbox=box, zorder=7)

            
        # x-axis label
        if xlabel[id_col] == 'Sigmol_avg':
            xsteps = 0.5
            if id_row == (rows - 1):
                ax.set_xlabel(r'$\log_{10}\,\langle\Sigma_\mathrm{mol}\rangle\;[\SI{}{\Msun\per\square\parsec}]$')
            elif id_row == 0:
                secxax = ax.secondary_xaxis('top', functions=(ICO21_log_function, Sigmol_log_function)) # secondary x-axis at top for Sigma_mol
                secxax.set_xlabel(r'$\log_{10}\,\langle W_{\rm CO(2-1)}\rangle\;[\SI{}{\kelvin\kilo\metre\per\second}]$', labelpad=8)
                secxax.tick_params(which='major', direction='out')
                secxax.tick_params(which='minor', length=0)
        elif xlabel[id_col] == 'vdis_avg':
            xsteps = 0.25
            if id_row == (rows - 1):
                ax.set_xlabel(r'$\log_{10}\,\langle\sigma_\mathrm{mol}\rangle\;[\SI{}{\kilo\metre\per\second}]$')
        elif xlabel[id_col] == 'avir_avg':
            xsteps = 0.25
            if id_row == (rows - 1):
                ax.set_xlabel(r'$\log_{10}\,\langle\alpha_\mathrm{vir}\rangle$')
            elif (id_row == 0) and (cloud_scale_pc is not None):
                secxax = ax.secondary_xaxis('top', functions=(sigSigratio_log_function, avir_log_function)) # secondary x-axis at top for Sigma_mol
                secxax.set_xlabel(r'$\log_{10}\,\langle \sigma_\mathrm{mol}^2/\Sigma_\mathrm{mol}\rangle\;[(\SI{}{\kilo\metre\per\second})^2/(\SI{}{\Msun\per\square\parsec})]$', loc='left', labelpad=8)
                secxax.tick_params(which='major', direction='out')
                secxax.tick_params(which='minor', length=0)
        elif xlabel[id_col] == 'Ptur_avg':
            xsteps = 1
            if id_row == (rows - 1):
                ax.set_xlabel(r'$\log_{10}\,\langle P_\mathrm{turb}\rangle\;[\SI{}{\kB\kelvin\per\cubic\centi\metre}]$')
        else:
            print('[ERROR] x-label not known!')

        # y-axis label (upper row)
        if ylabel[id_row] == 'fdense_HCN':
            if id_col == 0:
                ax.set_ylabel(r'$\log_{10}\,\mathrm{HCN}/\mathrm{CO}$')
            elif id_col == (cols - 1):
                secyax = ax.secondary_yaxis('right', functions=(f_dense_log_function, int_ratio_log_function)) # secondary y-axis for f_dense
                secyax.set_ylabel(r'$\log_{10}\,f_\mathrm{dense}$')
                secyax.tick_params(which='major', direction='out')
                secyax.tick_params(which='minor', length=0)
            else:
                pass
        elif ylabel[id_row] == 'fdense_HCOP':
                ax.set_ylabel(r'$\log_{10}\,\mathrm{HCO}^+/\mathrm{CO}$')
        elif ylabel[id_row] == 'fdense_CS':
            if id_col == 0:
                ax.set_ylabel(r'$\log_{10}\,\mathrm{CS}/\mathrm{CO}$')

        # y-axis label (lower row)
        elif ylabel[id_row] == 'SFE_HCN':
            if id_col == 0:
                ax.set_ylabel(r'$\log_{10}\,\mathrm{SFR}/\mathrm{HCN}$')  
                #ax.set_ylabel(r'$\log_{10}\,\mathrm{SFR}/\mathrm{HCN}\;[\SI{}{\Msun\per\year\per\square\kilo\parsec}/(\SI{}{\kelvin\kilo\metre\per\second})]$')    
            elif id_col == (cols - 1):
                secyax = ax.secondary_yaxis('right', functions=(Sigma_SFR_Sigma_HCN_function, Sigma_SFR_I_HCN_log_function)) # secondary y-axis for SFE
                secyax.set_ylabel(r'$\log_{10}\,\mathrm{SFE}_\mathrm{dense}\;[\mathrm{Myr}^{-1}]$')
                secyax.tick_params(which='major', direction='out')
                secyax.tick_params(which='minor', length=0)
            else:
                pass
        elif ylabel[id_row] == 'SFE_HCOP':
            if id_col == 0:
                ax.set_ylabel(r'$\log_{10}\,\mathrm{SFR}/\mathrm{HCO}^+$')
        elif ylabel[id_row] == 'SFE_CS':
            if id_col == 0:
                ax.set_ylabel(r'$\log_{10}\,\mathrm{SFR}/\mathrm{CS}$')
        else:
            print('[ERROR] y-label not known!')

            
        # plot parameters
        # plot range
        ax.set_xlim(xlim[0], xlim[1])
        ax.set_ylim(ylim[0], ylim[1])
        # x-ticks
        if xticks_list is None:
            xticks = np.arange(xlim[0]-xlim[0]%xsteps, xlim[1]+xsteps, xsteps)
            xticks = xticks[xticks >= xlim[0]]
        else:
            xticks = xticks_list[id_col]
        ax.set_xticks(xticks)
        # y-ticks
        if yticks_list is None:
            ysteps = 0.5
            yticks = np.arange(ylim[0]-ylim[0]%ysteps, ylim[1]+ysteps, ysteps)
            yticks = yticks[yticks >= ylim[0]]
        else:
            yticks = yticks_list[id_row]
        ax.set_yticks(yticks)
        
        if id_row != (rows - 1):
            ax.tick_params(axis='x', labelbottom=False)
            
        if id_col > 0:
            ax.tick_params(axis='y', labelleft=False)

        # grid
        ax.grid(ls='dotted', lw=1, zorder=1)
        
        # background color
        ax.set_facecolor('white')
        
        # legend
        if (xlabel[id_col] == 'vdis_avg') & (id_row == 0):
            legend = ax.legend(loc='lower left', bbox_to_anchor=(0, 1), ncol=5, columnspacing=1,
                               facecolor='white', edgecolor='k', fancybox=True, framealpha=1, borderpad=0.4,
                               fontsize=5, labelspacing=0.1, handlelength=1, handletextpad=0.5)
            legend.get_frame().set_linewidth(0.5)
            legend.set_zorder(10)
        elif (rows==1) & (cols==1):
            legend = ax.legend(loc='upper left', ncol=5, columnspacing=0.5,
                               facecolor='white', edgecolor='k', fancybox=True, framealpha=1, borderpad=0.4,
                               fontsize=5, labelspacing=0.1, handlelength=0.7, handletextpad=0.5)
            legend.get_frame().set_linewidth(0.5)
            legend.set_zorder(10)

            
        # model predictions
        if model_prediction:
            
            # plot model predictions
            ax.scatter(np.log10(x_model[id_col]), np.log10(y_model[id_row]), ec='none', fc='maroon', s=100, alpha=0.5, zorder=-3)
            ax.scatter(np.log10(x_model[id_col]), np.log10(y_model[id_row]), ec='none', fc='white', s=70, alpha=1, zorder=-2)
            ax.scatter(np.log10(x_model[id_col]), np.log10(y_model[id_row]), ec='none', fc='mistyrose', s=70, alpha=0.5, zorder=-1)
            
            # add in-plot annotations
            if i == 1:
                if annotation:
                    plt.annotate('model', xy=(1.9, -2), xytext=(2.7, -1.9), arrowprops=arrowprops, zorder=10)

        # in-plot annotations
        if annotation:
            if i == 1:
                # upper limit
                plt.annotate('upper limit', xy=(2.23, -1.15), xytext=(1.5, -0.97), arrowprops=arrowprops, zorder=10)
                # significant data
                plt.annotate(r'$\geq\,%i \sigma$ data' % SNR, xy=(1.95, -1.27), xytext=(1.1, -1.1), arrowprops=arrowprops, zorder=10)

            if i == 2:
                # fit line
                plt.annotate('fit line', xy=(0.45, -1.93), xytext=(0.4, -1.2), arrowprops=arrowprops, zorder=10)
                # fit uncertainty
                plt.annotate(r'$\pm 1\sigma$ fit unc.', xy=(0.52, -1.85), xytext=(0.6, -1), arrowprops=arrowprops, zorder=10)            
                # data scatter
                plt.annotate(r'$\pm 1\sigma$scatter', xy=(1.27, -1.45), xytext=(1.15, -1.8), arrowprops=arrowprops, zorder=10)

            if i == 4:
                # lower limit
                plt.annotate('lower limit', xy=(1.6, -1.22), xytext=(1.1, -1.5), arrowprops=arrowprops, zorder=10)


    # SAVE PLOT
    if saveplot:
        plt.savefig(savename)
        
        if not showplot:
            plt.close()

    # show plot
    if showplot:
        plt.show()
        
    # revert changes to the configuration made by homogenize_plot:
    hp.revert_params()
    
    # make dictionary
    keys = ['slope', 'slope_unc', 'intercept', 'intercept_unc', 'scatter_residuals', 'scatter_intrinsic', 'scatter_intrinsic_unc', 'corr_linmix', 'corr_linmix_unc', 'p_value_linmix', 
            'pearson_corr', 'pearson_p_value', 'rank_corr', 'rank_p_value']
    output = dict.fromkeys(keys)
    
    # assign pearson and spearman correlation coefficients and corresponding p-values (only detection)
    output['pearson_corr'] = pearson_corr
    output['pearson_p_value'] = pearson_p_value
    output['rank_corr'] = rank_corr
    output['rank_p_value'] = rank_p_value
    
    # assign fit parameters (if computed)
    # so far only for the last fit of the iteration
    if fit[id_row][id_col]:
        output['slope'] = slope
        output['slope_unc'] = slope_unc
        output['intercept'] = intercept
        output['intercept_unc'] = intercept_unc
        output['scatter_residuals'] = scatter_residuals
        output['scatter_intrinsic'] = scatter_intrinsic
        output['scatter_intrinsic_unc'] = scatter_intrinsic_unc
        output['corr_linmix'] = corr
        output['corr_linmix_unc'] = corr_unc
        output['p_value_linmix'] = p_value
        
    else:
        output['slope'] = np.nan
        output['slope_unc'] = (np.nan, np.nan)
        output['intercept'] = np.nan
        output['intercept_unc'] = (np.nan, np.nan)
        output['scatter_residuals'] = np.nan
        output['scatter_intrinsic'] = np.nan
        output['scatter_intrinsic_unc'] = (np.nan, np.nan)
        output['corr_linmix'] = np.nan
        output['corr_linmix_unc'] = (np.nan, np.nan)
        output['p_value_linmix'] = np.nan
    
    return output


def plot_centre_vs_disc(glxys, xdata, ydata, xdata_err, ydata_err, xlabel, ylabel, xrange, yrange,
                        ydata_upplim=None, ydata_lowlim=None, plotupplim=False, plotlowlim=False,
                        xticks_list=None, yticks_list=None, SNR=3, ploterr=False, 
                        plotBar=False, plotAGN=False, fit=False, MC_iter=10000, showparam=False, showcorr=True, 
                        savename=None, saveplot=False, showplot=True, cloud_scale_pc=None,
                        alphaCOunc=False):
    """
    TBD
    
    INPUT:
    - 
    
    OUPUT:
    - None (except plot to file)
    """
    
    # markersize
    ms = 4

    # constants to compute surface densities and dense gas fraction
    alpha_co10 = 4.3  # [M_Sun/pc^2/(K km/s)] (Bolatto et al. 2013)
    alpha_co21 = 1/0.64 * alpha_co10  # (den Brok et al. 2021)
    alpha_HCN = 14  # [M_Sun/pc^2/(K km/s)] (Onus et al. 2018)
    c_dense = alpha_HCN/alpha_co21

    # define functions to make secondary axis
    # secondary x-axis for CO(2-1) from Sigma_mol
    def ICO21_log_function(Sigmol_log):
        return Sigmol_log - np.log10(alpha_co21)
    def Sigmol_log_function(ICO21_log):
        return ICO21_log + np.log10(alpha_co21)
    
    # secondary x-axis for sigma^2/Sigma from alpha_vir
    if cloud_scale_pc is not None:
        conversion_avir = 3.1 * 150 / cloud_scale_pc  # 3.1 for 150 pc cloud-scale
        def sigSigratio_log_function(avir):
            return avir - np.log10(conversion_avir)
        def avir_log_function(sigSigratio):
            return sigSigratio + np.log10(conversion_avir)

    # secondary y-axis for fdense from HCN/CO
    def f_dense_log_function(int_ratio_log):
        return int_ratio_log + np.log10(c_dense)
    def int_ratio_log_function(f_dense_log):
        return f_dense_log - np.log10(c_dense)
    
    # secondary y-axis for SFE from SFR/HCN
    def Sigma_SFR_Sigma_HCN_function(Sigma_SFR_I_HCN_log):
        return Sigma_SFR_I_HCN_log - np.log10(alpha_HCN)  # [Myr^-1]
    def Sigma_SFR_I_HCN_log_function(Sigma_SFR_Sigma_HCN):
        return Sigma_SFR_Sigma_HCN + np.log10(alpha_HCN)
    
    # get rows, columns and number of plots
    rows = len(ydata)
    cols = len(xdata)
    n_plots = rows * cols

    # plot size
    aspect_ratio = 1
    if cols == 1:
        width = 2.5
    elif cols == 2:
        width = 5
    else:
        width = 7.5
    height = width / aspect_ratio / cols * rows

    # make LaTeX compatible plot
    hp.homogenise_plot(fig_width=width,    # width of the plot; 3.5 is default for a two-column LaTeX article document
                       fig_height=height,  # height of the plot; if None, golden ratio is adopted
                       )         

    # make figure
    fig = plt.figure()
    fig.subplots_adjust(hspace=0, wspace=0)  # subplots with adjacent axis
    
    # convert upper limit and lower limit bool to correct shape if given as False
    if plotupplim is False:
        plotupplim = np.ones((rows,cols)) * False
    if plotlowlim is False:
        plotlowlim = np.ones((rows,cols)) * False
    if fit is False:
        fit = np.ones((rows,cols)) * False
    elif fit is True:
        fit = np.ones((rows, cols)) * True

    # loop over subplots
    for i in range(1, n_plots+1):

        # make subplot
        ax = fig.add_subplot(rows, cols, i)

        # row and column indices
        id_row = (i-1) // cols
        id_col = (i-1) % cols

        # plot range
        xlim = xrange[id_col]
        ylim = yrange[id_row]

        # SUBPLOTS
        # loop over disc and centres data and plot data
        for j in range(2):
            
            # arrays to store data (for all galaxies combined)
            x_det_all, y_det_all, x_det_all_err, y_det_all_err = [],[],[],[]  # detections
            x_ul_all, y_ul_all, x_ul_all_err, y_ul_all_err = [],[],[],[]      # upper limits
            x_ll_all, y_ll_all, x_ll_all_err, y_ll_all_err = [],[],[],[]      # lower limits
            
            # copy galaxies list
            glxys1 = glxys.copy()
            
            # color and marker
            if j == 0:
                # disc
                color = 'tab:blue'
                marker = 'o'
                size = ms
                label = 'disc'
                zorder_err = 5
                alpha_lim = 0.5
            else:
                # centre
                color = 'tab:orange'
                marker = '*'
                size = ms*2
                label = 'centre'
                zorder_err = 6
                alpha_lim = 1
                
            # initiate counter for labels to only appear once
            c_bar, c_agn = 0, 0
         
            # get data
            x = xdata[id_col][j]
            y = ydata[id_row][j]
            x_err = xdata_err[id_col][j]
            y_err = ydata_err[id_row][j]
            y_ul = ydata_upplim[id_row][j]  # upper limit
            y_ll = ydata_lowlim[id_row][j]  # lower limit
                
            # make data copy for upper and/or lower limits
            x_ul, x_ll = np.copy(x), np.copy(x)
            x_ul_err, x_ll_err = np.copy(x_err), np.copy(x_err)
            y_ul_err, y_ll_err = np.copy(y_err), np.copy(y_err)

            # remove nan values
            id_nonnan = ~np.isnan(x) & ~np.isnan(y) & ~np.isnan(x_err) & ~np.isnan(y_err)
            x = x[id_nonnan]
            y = y[id_nonnan]
            x_err = x_err[id_nonnan]
            y_err = y_err[id_nonnan]
            if (j == 1) & (sum(id_nonnan) < len(glxys1)):
                glxys1.remove(glxys1[np.where(id_nonnan == False)[0][0]])
            
            id_nonnan_ul = ~np.isnan(x_ul) & ~np.isnan(y_ul) & ~np.isnan(x_ul_err) & ~np.isnan(y_ul_err)
            x_ul = x_ul[id_nonnan_ul]
            y_ul = y_ul[id_nonnan_ul]
            x_ul_err = x_ul_err[id_nonnan_ul]
            y_ul_err = y_ul_err[id_nonnan_ul]
            
            id_nonnan_ll = ~np.isnan(x_ll) & ~np.isnan(y_ll) & ~np.isnan(x_ll_err) & ~np.isnan(y_ll_err)
            x_ll = x_ll[id_nonnan_ll]
            y_ll = y_ll[id_nonnan_ll]
            x_ll_err = x_ll_err[id_nonnan_ll]
            y_ll_err = y_ll_err[id_nonnan_ll]

            # remove non-positive values
            id_pos = (x > 0) & (y > 0) & (x_err > 0) & (y_err > 0)
            x = x[id_pos]
            y = y[id_pos]
            x_err = x_err[id_pos]
            y_err = y_err[id_pos]
            if (j == 1) & (sum(id_pos) < len(glxys1)):
                glxys1.remove(glxys1[np.where(id_pos == False)[0][0]])
            
            id_pos_ul = (x_ul > 0) & (y_ul > 0) & (x_ul_err > 0) & (y_ul_err > 0)
            x_ul = x_ul[id_pos_ul]
            y_ul = y_ul[id_pos_ul]
            x_ul_err = x_ul_err[id_pos_ul]
            y_ul_err = y_ul_err[id_pos_ul]
            
            id_pos_ll = (x_ll > 0) & (y_ll > 0) & (x_ll_err > 0) & (y_ll_err > 0)
            x_ll = x_ll[id_pos_ll]
            y_ll = y_ll[id_pos_ll]
            x_ll_err = x_ll_err[id_pos_ll]
            y_ll_err = y_ll_err[id_pos_ll]
            
            # above SNR threshold
            id_highSNR = (y/y_err) >= SNR
            x1 = x[id_highSNR] 
            y1 = y[id_highSNR]
            x1_err = x_err[id_highSNR]
            y1_err = y_err[id_highSNR]
            if (j == 1) & (sum(id_highSNR) < len(glxys1)):
                glxys1.remove(glxys1[np.where(id_highSNR == False)[0][0]])

            # below SNR threshold
            id_lowSNR = (y/y_err) < SNR
            x2 = x[id_lowSNR] 
            y2 = y[id_lowSNR]
            x2_err = x_err[id_lowSNR]
            y2_err = y_err[id_lowSNR]

            # convert to log-log space
            x1_log = np.log10(x1)
            y1_log = np.log10(y1)
            x1_err_up_log = np.abs(np.log10(1 + x1_err/x1))
            x1_err_dw_log = np.abs(np.log10(1 - x1_err/x1))
            y1_err_up_log = np.abs(np.log10(1 + y1_err/y1))
            y1_err_dw_log = np.abs(np.log10(1 - y1_err/y1))
            #x1_err_dex = np.abs(np.log10(1 + x1_err/x1))
            #y1_err_dex = np.abs(np.log10(1 + y1_err/y1))

            x2_log = np.log10(x2)
            y2_log = np.log10(y2)
            x2_err_up_log = np.abs(np.log10(1 + x2_err/x2))
            x2_err_dw_log = np.abs(np.log10(1 - x2_err/x2))
            y2_err_up_log = np.abs(np.log10(1 + y2_err/y2))
            y2_err_dw_log = np.abs(np.log10(1 - y2_err/y2))
            #x2_err_dex = np.abs(np.log10(1 + x2_err/x2))
            #y2_err_dex = np.abs(np.log10(1 + y2_err/y2))
            
            x_ul_log = np.log10(x_ul)
            y_ul_log = np.log10(y_ul)
            x_ul_err_dex = np.full_like(x_ul_log, np.log10(1+2/3))  # 2-sigma uncertainty relative to the upper limit
            y_ul_err_dex = np.full_like(y_ul_log, np.log10(1+2/3))
            #x_ul_err_dex = np.abs(np.log10(1 + x_ul_err/x_ul))
            #y_ul_err_dex = np.abs(np.log10(1 + y_ul_err/y_ul))

            x_ll_log = np.log10(x_ll)
            y_ll_log = np.log10(y_ll)
            x_ll_err_dex = np.full_like(x_ll_log, np.log10(1+2/3))  # 2-sigma uncertainty relative to the upper limit
            y_ll_err_dex = np.full_like(y_ll_log, np.log10(1+2/3))
            #x_ll_err_dex = np.abs(np.log10(1 + x_ll_err/x_ll))
            #y_ll_err_dex = np.abs(np.log10(1 + y_ll_err/y_ll))

                    
            # plot binned data above SNR threshold filled
            if ploterr:
                ax.errorbar(x1_log, y1_log, yerr=(y1_err_dw_log, y1_err_up_log), xerr=(x1_err_dw_log, x1_err_up_log), linestyle='none', 
                            capsize=1.5, ecolor=color, elinewidth=1, capthick=1, zorder=zorder_err)
                
            if j == 0:
                ax.plot(x1_log, y1_log, linestyle='none', marker=marker, markerfacecolor=color, 
                        markeredgecolor='k', label=label, zorder=7, markersize=size, markeredgewidth=0.5)
                
            # for centres (indicate Bar and/or AGN)
            elif j == 1:
                
                # plot centres data
                ax.plot(x1_log, y1_log, linestyle='none', marker=marker, markerfacecolor=color, 
                    markeredgecolor='k', label=label, zorder=7, markersize=size, markeredgewidth=0.5)
                
                # plot arrow indicating alphaCO variations
                if alphaCOunc:
                    
                    aCO_factor = 2  # if centres had factor 2 lower alpha_CO
                    aCO_factor_log = np.log10(aCO_factor)
                    box_aCO = dict(boxstyle='round', facecolor='white', edgecolor='k', linewidth=0.5, alpha=1)  # set box
                    
                    if (xlabel[id_col] == 'Sigmol_avg') & (ylabel[id_row][:3] == 'SFE'):
                        ax.text(2.0, -0.125, r'$.\;\,\quad\quad\alpha_\mathrm{CO}/ 2$', color='white', ha='left', va='center', bbox=box_aCO, zorder=10)
                        ax.text(3.2-0.75, -0.125, r'$\alpha_\mathrm{CO}/ 2$', ha='left', va='center', zorder=11)
                        ax.arrow(3.1-0.75, -0.125, -aCO_factor_log, 0, head_width=0.02, lw=1, color='tab:orange', clip_on=False, zorder=12)
                        ax.arrow(3.1-0.75, -0.125, -aCO_factor_log, 0, head_width=0.025, lw=1.5, color='k', clip_on=False, zorder=11)

                    elif (xlabel[id_col] == 'avir_avg') & (ylabel[id_row][:3] == 'SFE'):
                        ax.text(1.0-0.2, -0.125, r'$\alpha_\mathrm{CO}/2\quad\quad\quad\quad\quad.$', color='white', ha='right', va='center', bbox=box_aCO, zorder=10)
                        ax.text(0.64-0.2, -0.125, r'$\alpha_\mathrm{CO}/2$', ha='right', va='center', zorder=11)
                        ax.arrow(0.67-0.2, -0.125, aCO_factor_log, 0, head_width=0.02, head_length=0.015, lw=1, color='tab:orange', clip_on=False, zorder=12)
                        ax.arrow(0.67-0.2, -0.125, aCO_factor_log, 0, head_width=0.025, head_length=0.018, lw=1.5, color='k', clip_on=False, zorder=11)
                        pass

                # loop over galaxies (one point per galaxy)
                for glxy in glxys1:

                    # get list index
                    id_glxy = glxys1.index(glxy)
                    
                    # read PHANGS database (sample table)
                    PHANGS_database_path = '/Users/lneumann/Documents/PhD/Data/PHANGS/tables/phangs_sample_table_v1p6.csv'
                    database = pd.read_csv(PHANGS_database_path, comment='#', skiprows=1)
                    
                    # get index of galaxy in PHANGS database
                    idx = np.where(database['name']==glxy)[0][0]
                    
                    # Bar
                    bar = database['morph_bar_flag'][idx]  # 1 indicates bar [Querrejeta+21]
    
                    # AGN (Veron)
                    agn = database['agn_veron_y_n'][idx]  # '1' indicates AGN, '0' indicates no AGN
 
                    if plotBar & (bar == 1):
                        if c_bar == 0:
                            # label only ones
                            ax.plot(x1_log[id_glxy], y1_log[id_glxy], linestyle='none', marker='s', markerfacecolor='none', 
                                    markeredgecolor='k', zorder=8, markersize=size-1, markeredgewidth=0.5, label='bar')
                            c_bar += 1
                        else:
                            ax.plot(x1_log[id_glxy], y1_log[id_glxy], linestyle='none', marker='s', markerfacecolor='none', 
                                    markeredgecolor='k', zorder=8, markersize=size-1, markeredgewidth=0.5)
                        
                    if plotAGN & (agn == '1'):
                        if c_agn == 0:
                            # label only ones
                            ax.plot(x1_log[id_glxy], y1_log[id_glxy], linestyle='none', marker='.', color='k', 
                                    zorder=9, markersize=size-5, label='AGN')
                            c_agn += 1
                        else:
                            ax.plot(x1_log[id_glxy], y1_log[id_glxy], linestyle='none', marker='.', color='k', 
                                    zorder=9, markersize=size-5)
                    

            # plot upper limits
            if plotupplim[id_row]:
                (_, caps_ul, _) = ax.errorbar(x_ul_log, y_ul_log, yerr=abs(ylim[0]-ylim[1])/20, uplims=True, linestyle='none',
                                           marker='None', elinewidth=1, color=color, zorder=zorder_err, alpha=alpha_lim)
                for cap in caps_ul:
                    cap.set_markersize(ms)
                    cap.set_markeredgewidth(0)
                
            # plot lower limits
            if plotlowlim[id_row]:
                (_, caps_ll, _) = ax.errorbar(x_ll_log, y_ll_log, yerr=abs(ylim[0]-ylim[1])/20, lolims=True, linestyle='none',
                                              marker='None', elinewidth=1, color=color, zorder=zorder_err, alpha=alpha_lim)
                for cap in caps_ll:
                    cap.set_markersize(ms)
                    cap.set_markeredgewidth(0)
                
            # concatenate data to combined arrays; use the downward log uncertainty as the dex uncertainty
            x_det_all, y_det_all = np.concatenate((x_det_all, x1_log)), np.concatenate((y_det_all, y1_log))
            x_det_all_err, y_det_all_err = np.concatenate((x_det_all_err, x1_err_dw_log)), np.concatenate((y_det_all_err, y1_err_dw_log))
            
            x_ul_all, y_ul_all = np.concatenate((x_ul_all, x_ul_log)), np.concatenate((y_ul_all, y_ul_log)) 
            x_ul_all_err, y_ul_all_err = np.concatenate((x_ul_all_err, x_ul_err_dex)), np.concatenate((y_ul_all_err, y_ul_err_dex))
            
            x_ll_all, y_ll_all = np.concatenate((x_ll_all, x_ll_log)), np.concatenate((y_ll_all, y_ll_log))
            x_ll_all_err, y_ll_all_err = np.concatenate((x_ll_all_err, x_ll_err_dex)), np.concatenate((y_ll_all_err, y_ll_err_dex))

            
            # get Spearman rank correlation and corresponding p-value
            try:
                rank_corr, rank_p_value = spearmanr(x_det_all, y_det_all)
            except:
                rank_corr, rank_p_value = np.nan, np.nan

            # get Pearson correlation coefficient and corresponding p-value
            try:
                pearson_corr, pearson_p_value = pearsonr(x_det_all, y_det_all)
            except:
                pearson_corr, pearson_p_value = np.nan, np.nan


            if fit[id_row][id_col]:

                # get censored data
                if ylabel[id_row][:6] == 'fdense':
                    x_cens = x_ul_all
                    y_cens = y_ul_all
                    x_cens_err = x_ul_all_err
                    y_cens_err = y_ul_all_err
                    fit_method = 'upplim'

                elif ylabel[id_row][:3] == 'SFE':
                    x_cens = x_ll_all
                    y_cens = y_ll_all
                    x_cens_err = x_ll_all_err
                    y_cens_err = y_ll_all_err
                    fit_method = 'lowlim'

                # set x-offset where the interception should be measured
                if xlabel[id_col] == 'Sigmol_avg':
                    x_off = 2.5
                elif xlabel[id_col] == 'vdis_avg':
                    x_off = 1.1
                elif xlabel[id_col] == 'avir_avg':
                    x_off = 0.2
                elif xlabel[id_col] == 'Ptur_avg':
                    x_off = 6.5

                # plot vertical line to indicate intercept reference
                #ax.axvline(x_off, lw=1 , c='k', ls='-', zorder=2)

                # make x-list for plotting credibility intervals
                x_cre = np.linspace(xlim[0], xlim[1], 50)

                #fit_method = None
                out_linmix = fit_linmix(x_det_all-x_off, y_det_all, x_det_err=x_det_all_err, y_det_err=y_det_all_err, 
                                        x_cens=x_cens-x_off, y_cens=y_cens, x_cens_err=x_cens_err, y_cens_err=y_cens_err, 
                                        method=fit_method, x_cre=x_cre-x_off, K=3, seed=7, MC_iter=MC_iter, debug=False, 
                                        data_output=False)
                
                if out_linmix is not None:
                    slope, slope_unc, intercept, intercept_unc, corr, corr_unc, scatter_intrinsic, scatter_intrinsic_unc, stats = out_linmix
                    scatter_residuals = stats['scatter_residuals_y']
                else:
                    slope, slope_unc, intercept, intercept_unc = np.nan,(np.nan,np.nan), np.nan,(np.nan,np.nan)
                    corr, corr_unc, scatter_intrinsic, scatter_intrinsic_unc = np.nan,(np.nan,np.nan), np.nan,(np.nan,np.nan)
                    stats = dict.fromkeys(['p_value', 'scatter_residuals', '-1sigma', '+1sigma', '-2sigma', '+2sigma', '-3sigma', '+3sigma'])
                    stats['p_value'], stats['scatter_residuals'], stats['-1sigma'], stats['+1sigma'] = np.nan, np.nan, np.nan, np.nan
                    stats['-2sigma'], stats['+2sigma'], stats['-3sigma'], stats['+3sigma'] = np.nan, np.nan, np.nan, np.nan


                # plot linmix median line fit
                x_fit = np.linspace(xlim[0], xlim[1], 10)
                y_fit = intercept + slope * (x_fit-x_off)
                ax.plot(x_fit, y_fit, 'k-', lw=3, zorder=4)
                ax.plot(x_fit, y_fit, ls='-', color=color, lw=1.5, zorder=5)

                # scatter of the residuals about the median fit line
                y_scatter = scatter_residuals
                y_sca_n = intercept + slope * (x_fit-x_off) - y_scatter
                y_sca_p = intercept + slope * (x_fit-x_off) + y_scatter
                ax.fill_between(x_fit, y_sca_n, y_sca_p, color=color, lw=0, alpha=0.5, zorder=2)

                
                # credibility areas of fit parameters
               # if out_linmix is not None:
                   # ax.fill_between(x_cre, stats['-1sigma'], stats['+1sigma'], color=color, lw=0.1, alpha=0.5, zorder=2)
                   # ax.fill_between(x_cre, stats['-2sigma'], stats['+2sigma'], color=color, lw=0.1, alpha=0.5, zorder=2)
                   # ax.fill_between(x_cre, stats['-3sigma'], stats['+3sigma'], color=color, lw=0.1, alpha=0.3, zorder=2)
                    
                # credibility areas of fit parameters
                if out_linmix is not None:
                    ax.plot(x_cre, stats['-1sigma'], color=color, ls='dashed', lw=1, zorder=3)
                    ax.plot(x_cre, stats['+1sigma'], color=color, ls='dashed', lw=1, zorder=3)
                  #  ax.plot(x_cre, stats['-2sigma'], color=color, ls='dotted', lw=0.5, zorder=3)
                  #  ax.plot(x_cre, stats['+2sigma'], color=color, ls='dotted', lw=0.5, zorder=3)
                  #  ax.plot(x_cre, stats['-3sigma'], color=color, ls='dashdot', lw=0.5, zorder=3)
                  #  ax.plot(x_cre, stats['+3sigma'], color=color, ls='dashdot', lw=0.5, zorder=3)

                # p_value
                p_value = stats['p_value']

                # plot fit parameter results
                if showparam:
                    if slope > 0:
                        if j == 0:
                            text = 'slope = %.2f (%.2f)\ninterc. = %.2f (%.2f)\nscatter = %.2f\n$r$ ($p$) = %.2f (%.3f)' \
                                   % (slope, np.nanmean(np.abs(slope_unc)), intercept, np.nanmean(np.abs(intercept_unc)), scatter_residuals, corr, p_value)
                            box = dict(boxstyle='round', facecolor='white', edgecolor=color, linewidth=1, alpha=1)  # set box
                            ax.text(0.04, 0.96, text, ha='left', va='top', transform = ax.transAxes, bbox=box, zorder=10, fontsize=5)
                        else:
                            text = 'slope = %.2f (%.2f)\ninterc. = %.2f (%.2f)\nscatter = %.2f\n$r$ ($p$) = %.2f (%.3f)' \
                                   % (slope, np.nanmean(np.abs(slope_unc)), intercept, np.nanmean(np.abs(intercept_unc)), scatter_residuals, corr, p_value)
                            box = dict(boxstyle='round', facecolor='white', edgecolor=color, linewidth=1, alpha=1)  # set box
                            ax.text(0.96, 0.04, text, ha='right', va='bottom', transform = ax.transAxes, bbox=box, zorder=10, fontsize=5) 
                    else:
                        if j == 0:
                            text = 'slope = %.2f (%.2f)\ninterc. = %.2f (%.2f)\nscatter = %.2f\n$r$ ($p$) = %.2f (%.3f)' \
                                   % (slope, np.nanmean(np.abs(slope_unc)), intercept, np.nanmean(np.abs(intercept_unc)), scatter_residuals, corr, p_value)
                            box = dict(boxstyle='round', facecolor='white', edgecolor=color, linewidth=1, alpha=1)  # set box
                            ax.text(0.04, 0.04, text, ha='left', va='bottom', transform = ax.transAxes, bbox=box, zorder=10, fontsize=5)
                        else:
                            text = 'slope = %.2f (%.2f)\ninterc. = %.2f (%.2f)\nscatter = %.2f\n$r$ ($p$) = %.2f (%.3f)' \
                                   % (slope, np.nanmean(np.abs(slope_unc)), intercept, np.nanmean(np.abs(intercept_unc)), scatter_residuals, corr, p_value)
                            box = dict(boxstyle='round', facecolor='white', edgecolor=color, linewidth=1, alpha=1)  # set box
                            ax.text(0.96, 0.96, text, ha='right', va='top', transform = ax.transAxes, bbox=box, zorder=10, fontsize=5) 
                            
                elif showcorr:
                    text = r'$\rho = %.2f$' % corr
                    box = dict(boxstyle='round', facecolor='white', edgecolor='none', linewidth=0, alpha=1, pad=0.2)  # set box
                    if id_row == 0:
                        if j == 0:
                            # disc
                            ax.text(0.95, 0.15, text, ha='right', va='bottom', color=color, transform=ax.transAxes, bbox=box, zorder=7)
                        else:
                            # centre
                            ax.text(0.95, 0.08, text, ha='right', va='bottom', color=color, transform=ax.transAxes, bbox=box, zorder=7)
                    else:
                        if j == 0:
                            # disc
                            ax.text(0.95, 0.92, text, ha='right', va='top', color=color, transform=ax.transAxes, bbox=box, zorder=7)
                        else:
                            # centre
                            ax.text(0.95, 0.85, text, ha='right', va='top', color=color, transform=ax.transAxes, bbox=box, zorder=7)
                
            else:
                if showparam:
                    if id_row == 0:
                        if j == 0:
                            # plot Person r and p-value
                            text = '$r$ ($p$) = %.2f (%.3f)' % (pearson_corr, pearson_p_value)
                            box = dict(boxstyle='round', facecolor='white', edgecolor=color, linewidth=0.5, alpha=1)  # set box
                            ax.text(0.96, 0.12, text, ha='right', va='bottom', transform = ax.transAxes, bbox=box, zorder=10, fontsize=5)
                        else:
                            # plot Person r and p-value
                            text = '$r$ ($p$) = %.2f (%.3f)' % (pearson_corr, pearson_p_value)
                            box = dict(boxstyle='round', facecolor='white', edgecolor=color, linewidth=0.5, alpha=1)  # set box
                            ax.text(0.96, 0.04, text, ha='right', va='bottom', transform = ax.transAxes, bbox=box, zorder=10, fontsize=5)
                    else:
                        if j == 0:
                            # plot Person r and p-value
                            text = '$r$ ($p$) = %.2f (%.3f)' % (pearson_corr, pearson_p_value)
                            box = dict(boxstyle='round', facecolor='white', edgecolor=color, linewidth=0.5, alpha=1)  # set box
                            ax.text(0.96, 0.88, text, ha='right', va='top', transform = ax.transAxes, bbox=box, zorder=10, fontsize=5)
                        else:
                            # plot Person r and p-value
                            text = '$r$ ($p$) = %.2f (%.3f)' % (pearson_corr, pearson_p_value)
                            box = dict(boxstyle='round', facecolor='white', edgecolor=color, linewidth=0.5, alpha=1)  # set box
                            ax.text(0.96, 0.96, text, ha='right', va='top', transform = ax.transAxes, bbox=box, zorder=10, fontsize=5)       
                            
                elif showcorr:
                    text = r'$\rho = %.2f$' % pearson_corr
                    box = dict(boxstyle='round', facecolor='white', edgecolor='none', linewidth=0, alpha=1, pad=0.2)  # set box
                    if id_row == 0:
                        if j == 0:
                            # disc
                            ax.text(0.95, 0.15, text, ha='right', va='bottom', color=color, transform=ax.transAxes, bbox=box, zorder=7)
                        else:
                            # centre
                            ax.text(0.95, 0.08, text, ha='right', va='bottom', color=color, transform=ax.transAxes, bbox=box, zorder=7)
                    else:
                        if j == 0:
                            # disc
                            ax.text(0.95, 0.92, text, ha='right', va='top', color=color, transform=ax.transAxes, bbox=box, zorder=7)
                        else:
                            # centre
                            ax.text(0.95, 0.85, text, ha='right', va='top', color=color, transform=ax.transAxes, bbox=box, zorder=7)

            
        # x-axis label
        if xlabel[id_col] == 'Sigmol_avg':
            xsteps = 0.5
            if id_row == (rows - 1):
                ax.set_xlabel(r'$\log_{10}\,\langle\Sigma_\mathrm{mol}\rangle\;[{\rm M}_{\odot}\,{\rm pc}^{-2}]$')
            if id_row == 0:
                secxax = ax.secondary_xaxis('top', functions=(ICO21_log_function, Sigmol_log_function)) # secondary x-axis at top for Sigma_mol
                secxax.set_xlabel(r'$\log_{10}\,\langle W_{\rm CO(2-1)}\rangle\;[{\rm K}\,{\rm km}\,{\rm s}^{-1}]$', labelpad=8)
                secxax.tick_params(which='major', direction='out')
                secxax.tick_params(which='minor', length=0)
        elif xlabel[id_col] == 'vdis_avg':
            xsteps = 0.25
            if id_row == (rows - 1):
                ax.set_xlabel(r'$\log_{10}\,\langle\sigma_\mathrm{mol}\rangle\;[{\rm km}\,{\rm s}^{-1}]$')
        elif xlabel[id_col] == 'avir_avg':
            xsteps = 0.25
            if id_row == (rows - 1):
                ax.set_xlabel(r'$\log_{10}\,\langle\alpha_\mathrm{vir}\rangle$')
            elif (id_row == 0) and (cloud_scale_pc is not None):
                secxax = ax.secondary_xaxis('top', functions=(sigSigratio_log_function, avir_log_function)) # secondary x-axis at top for Sigma_mol
                secxax.set_xlabel(r'$\log_{10}\,\langle \sigma_\mathrm{mol}^2/\Sigma_\mathrm{mol}\rangle\;[({\rm km}\,{\rm s}^{-1})^2/({\rm M}_{\odot}\,{\rm pc}^{-2})]$', labelpad=8)
                secxax.tick_params(which='major', direction='out')
                secxax.tick_params(which='minor', length=0)
        elif xlabel[id_col] == 'Ptur_avg':
            xsteps = 1
            if id_row == (rows - 1):
                ax.set_xlabel(r'$\log_{10}\,\langle P_\mathrm{turb}\rangle\;[k_{\rm B}\,{\rm K}\,{\rm cm}^{-3}]$')
        else:
            print('[ERROR] x-label not known!')

        # y-axis label (upper row)
        if ylabel[id_row] == 'fdense_HCN':
            if cols == 1:
                ax.set_ylabel(r'$\log_{10}\,\mathrm{HCN}/\mathrm{CO}$')
                secyax = ax.secondary_yaxis('right', functions=(f_dense_log_function, int_ratio_log_function)) # secondary y-axis for f_dense
                secyax.set_ylabel(r'$\log_{10}\,f_\mathrm{dense}$')
                secyax.tick_params(which='major', direction='out')
                secyax.tick_params(which='minor', length=0)
            else:
                if id_col == 0:
                    ax.set_ylabel(r'$\log_{10}\,\mathrm{HCN}/\mathrm{CO}$')
                elif id_col == (cols - 1):
                    secyax = ax.secondary_yaxis('right', functions=(f_dense_log_function, int_ratio_log_function)) # secondary y-axis for f_dense
                    secyax.set_ylabel(r'$\log_{10}\,f_\mathrm{dense}$')
                    secyax.tick_params(which='major', direction='out')
                    secyax.tick_params(which='minor', length=0)
                else:
                    pass
        elif ylabel[id_row] == 'fdense_HCOP':
            if id_col == 0:
                ax.set_ylabel(r'$\log_{10}\,\mathrm{HCO}^+/\mathrm{CO}$')
        elif ylabel[id_row] == 'fdense_CS':
            if id_col == 0:
                ax.set_ylabel(r'$\log_{10}\,\mathrm{CS}/\mathrm{CO}$')

        # y-axis label (lower row)
        elif ylabel[id_row] == 'SFE_HCN':
            if id_col == 0:
                ax.set_ylabel(r'$\log_{10}\,\mathrm{SFR}/\mathrm{HCN}$')    
            elif id_col == (cols - 1):
                secyax = ax.secondary_yaxis('right', functions=(Sigma_SFR_Sigma_HCN_function, Sigma_SFR_I_HCN_log_function)) # secondary y-axis for SFE
                secyax.set_ylabel(r'$\log_{10}\,\mathrm{SFE}_\mathrm{dense}\;[\mathrm{Myr}^{-1}]$')
                secyax.tick_params(which='major', direction='out')
                secyax.tick_params(which='minor', length=0)
            else:
                pass
        elif ylabel[id_row] == 'SFE_HCOP':
            if id_col == 0:
                ax.set_ylabel(r'$\log_{10}\,\Sigma_\mathrm{SFR}/I_{\mathrm{HCO}^+}$')
        elif ylabel[id_row] == 'SFE_CS':
            if id_col == 0:
                ax.set_ylabel(r'$\log_{10}\,\Sigma_\mathrm{SFR}/I_\mathrm{CS}$')
        else:
            print('[ERROR] y-label not known!')

            
        # plot parameters
        # plot range
        ax.set_xlim(xlim[0], xlim[1])
        ax.set_ylim(ylim[0], ylim[1])
        # x-ticks
        if xticks_list is None:
            xticks = np.arange(xlim[0]-xlim[0]%xsteps, xlim[1]+xsteps, xsteps)
            xticks = xticks[xticks >= xlim[0]]
        else:
            xticks = xticks_list[id_col]
        ax.set_xticks(xticks)
        # y-ticks
        if yticks_list is None:
            ysteps = 0.5
            yticks = np.arange(ylim[0]-ylim[0]%ysteps, ylim[1]+ysteps, ysteps)
            yticks = yticks[yticks >= ylim[0]]
        else:
            yticks = yticks_list[id_row]
        ax.set_yticks(yticks)
        
        if id_row == (rows - 1):
            ax.tick_params(axis='x')
        else:
            ax.tick_params(axis='x', labelbottom=False)
            
        if id_col == 0:
            ax.tick_params(axis='y')
        else:
            ax.tick_params(axis='y', labelleft=False)

        # grid
        ax.grid(ls='dotted', zorder=1)
        
        # legend
        if showparam:
            if (rows == 2) & (cols > 2) & (xlabel[id_col] == 'avir_avg') & (id_row == 1):
                legend = ax.legend(loc='center left', bbox_to_anchor=(0.02, 1), facecolor='white', edgecolor='k', fancybox=True, 
                                   framealpha=1, labelspacing=0.2, handlelength=0.5, handletextpad=0.5)
                legend.get_frame().set_linewidth(0.5)
                legend.set_zorder(10)
            elif (cols == 1) & (rows == 1):
                legend = ax.legend(loc='lower left', facecolor='white', edgecolor='k', fancybox=True, framealpha=1, 
                                   labelspacing=0.2, handlelength=0.5, handletextpad=0.5)
                legend.get_frame().set_linewidth(0.5)
                legend.set_zorder(10)
        else:
            if (rows == 2) & (cols > 2) & (xlabel[id_col] == 'vdis_avg') & (id_row == 0):
                legend = ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1), facecolor='white', edgecolor='k', fancybox=True, 
                                   ncol=2, columnspacing=1, framealpha=1, labelspacing=0.5, handlelength=0.5, handletextpad=0.5)
                legend.get_frame().set_linewidth(0.5)
                legend.set_zorder(10)
            elif (cols == 1) & (rows == 1):
                legend = ax.legend(loc='lower right', facecolor='white', edgecolor='k', fancybox=True, framealpha=1, 
                                   labelspacing=0.2, handlelength=0.5, handletextpad=0.5, ncols=2, columnspacing=0.5)
                legend.get_frame().set_linewidth(0.5)
                legend.set_zorder(10)            

    # SAVE PLOT
    if saveplot:
        plt.savefig(savename)
        
        if not showplot:
            plt.close()


    # show plot
    if showplot:
        plt.show()
        
    # revert changes to the configuration made by homogenize_plot:
    hp.revert_params()
    
    # make dictionary
    keys = ['slope', 'slope_unc', 'intercept', 'intercept_unc', 'scatter_residuals', 'scatter_intrinsic', 'scatter_intrinsic_unc', 'corr_linmix', 'corr_linmix_unc', 'p_value_linmix', 
            'pearson_corr', 'pearson_p_value', 'rank_corr', 'rank_p_value']
    output = dict.fromkeys(keys)
    
    # assign pearson and spearman correlation coefficients and corresponding p-values (only detection)
    output['pearson_corr'] = pearson_corr
    output['pearson_p_value'] = pearson_p_value
    output['rank_corr'] = rank_corr
    output['rank_p_value'] = rank_p_value
    
    # assign fit parameters (if computed)
    # so far only for the last fit of the iteration
    if fit[id_row][id_col]:
        output['slope'] = slope
        output['slope_unc'] = slope_unc
        output['intercept'] = intercept
        output['intercept_unc'] = intercept_unc
        output['scatter_residuals'] = scatter_residuals
        output['scatter_intrinsic'] = scatter_intrinsic
        output['scatter_intrinsic_unc'] = scatter_intrinsic_unc
        output['corr_linmix'] = corr
        output['corr_linmix_unc'] = corr_unc
        output['p_value_linmix'] = p_value
        
    else:
        output['slope'] = np.nan
        output['slope_unc'] = (np.nan, np.nan)
        output['intercept'] = np.nan
        output['intercept_unc'] = (np.nan, np.nan)
        output['scatter_residuals'] = np.nan
        output['scatter_intrinsic'] = np.nan
        output['scatter_intrinsic_unc'] = (np.nan, np.nan)
        output['corr_linmix'] = np.nan
        output['corr_linmix_unc'] = (np.nan, np.nan)
        output['p_value_linmix'] = np.nan

    
    return output