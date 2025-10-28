import numpy as np
from astropy.io import fits
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import os
import time
import sys
import csv
from datetime import datetime

class region_of_interest:
    def __init__(self, x_c, y_c, r):
        self.x_c = x_c
        self.y_c = y_c
        self.r = r
        self.x_min = int(x_c - r)
        self.x_max = int(x_c + r)
        self.y_min = int(y_c - r)
        self.y_max = int(y_c + r)

class log_data:
    def __init__(self):
        # [value, error]
        self.A_px = [0, 0]
        self.x0_px = [0, 0]
        self.y0_px = [0, 0]
        self.σx_px = [0, 0]
        self.σy_px = [0, 0]
        self.B_px = [0, 0]
        self.RMSE = [0, 0]
        self.RMSE_ratio = [0, 0]
        self.FWHM_x_px = [0, 0]
        self.FWHM_y_px = [0, 0]
        self.FWHM_avg_px = [0, 0]
        self.pixel_scale = [0, 0]
        self.σx_um = [0, 0]
        self.σy_um = [0, 0]
        self.FWHM_x_um = [0, 0]
        self.FWHM_y_um = [0, 0]
        self.FWHM_avg_um = [0, 0]
        self.MFD_x_um = [0, 0]
        self.MFD_y_um = [0, 0]

# Define 2D Gaussian model
def gauss2d(coords, A, x0, y0, sigma_x, sigma_y, B):
    x, y = coords
    exponent = ((x-x0)**2)/(2*sigma_x**2) + ((y-y0)**2)/(2*sigma_y**2)
    
    return A * np.exp(-exponent) + B

# Define 2D Super-Gaussian
def supergauss_2d(coords, A, x0, y0, sigma_x, sigma_y, n, B):
    x, y = coords
    expo_x = ((x - x0)**2 / (2 * sigma_x**2))**n
    expo_y = ((y - y0)**2 / (2 * sigma_y**2))**n

    return A * np.exp(-expo_x - expo_y) + B

# Define the rotated elliptical 2D Gaussian model
def rotated_gaussian(coords, A, x0, y0, sigma_x, sigma_y, theta, B):
    x, y = coords
    x_shift = x - x0
    y_shift = y - y0
    # Rotate coordinates
    x_rot = x_shift * np.cos(theta) + y_shift * np.sin(theta)
    y_rot = -x_shift * np.sin(theta) + y_shift * np.cos(theta)
    exponent = (x_rot**2) / (2 * sigma_x**2) + (y_rot**2) / (2 * sigma_y**2)

    return A * np.exp(-exponent) + B

class gauss_cal:
    def __init__(self,gauss_type):
        # Plot data of each fitting method
        self.data_org = None
        self.data_conven = None
        self.data_super = None
        self.data_elip = None
        # Brightest point of each fitting method
        self.brightest_point_org = (0,0)
        self.brightest_point_conven = (0,0)
        self.brightest_point_super = (0,0)
        self.brightest_point_elip = (0,0)
        # Log data of each fitting method
        self.log_data_conven = log_data()
        self.log_data_super = log_data()
        self.log_data_elip = log_data()

    def conven_2d_gauss(self, output_folder, data, roi, sensor_value):
        ############ Conventional 2D Gaussian fitting ############
        # Get region of interest
        x_c = roi.x_c
        y_c = roi.y_c
        r = roi.r
        x_min = roi.x_min
        x_max = roi.x_max
        y_min = roi.y_min
        y_max = roi.y_max

        # Select data region
        sub_data = data[y_min : y_max+1,   # rows
                        x_min : x_max+1]   # columns

        # Build the full-coordinate grids in the original pixel frame
        x_inds = np.arange(x_min, x_max+1)
        y_inds = np.arange(y_min, y_max+1)
        x_grid, y_grid = np.meshgrid(x_inds, y_inds)

        # Create a boolean mask for the circle
        mask = (x_grid - x_c)**2 + (y_grid - y_c)**2 <= r**2

        # Apply the mask to pick only pixels inside the circle
        xdata = np.vstack((x_grid[mask], y_grid[mask]))   # shape (2, N_pixels_in_circle)
        zdata = sub_data[mask]                            # shape (N_pixels_in_circle,)

        # Initial guesses for [A, x0, y0, σx, σy, B]
        A0   = sub_data.max() - sub_data.min()
        x0_0 = x_inds[np.argmax(np.sum(sub_data, axis=0))]
        y0_0 = y_inds[np.argmax(np.sum(sub_data, axis=1))]
        σx0  = (x_max - x_min) / 4
        σy0  = (y_max - y_min) / 4
        B0   = sub_data.min()
        p0   = [A0, x0_0, y0_0, σx0, σy0, B0]

        # Calculate boundary for each parameter fitting
        lb = [0, x_min, y_min, 0, 0,  0]
        ub = [np.inf, x_max, y_max, np.inf, np.inf, np.inf]

        progress_bar(50, 100)
        time.sleep(0.05)  # simulate work

        # Fit
        popt, pcov = curve_fit(gauss2d, xdata, zdata, p0=p0, bounds=(lb,ub))
        A_fit, x0_fit, y0_fit, σx_fit, σy_fit, B_fit = popt

        # Parameter errors
        perr = np.sqrt(np.diag(pcov))
        σx_err, σy_err = perr[3], perr[4]

        # RMSE
        fit_img = gauss2d((x_grid, y_grid), *popt).reshape(sub_data.shape)
        rmse = np.sqrt(np.mean((fit_img - sub_data)**2))

        # FWHM in pixels for Gaussian: FWHM = 2*sqrt(2*ln2)*σ
        fwhm_factor = 2 * np.sqrt(2 * np.log(2))
        fwhm_x_px    = fwhm_factor * σx_fit
        fwhm_y_px    = fwhm_factor * σy_fit
        fwhm_x_err_px = fwhm_factor * σx_err
        fwhm_y_err_px = fwhm_factor * σy_err

        # Unit conversion to object-plane microns
        p_sens = sensor_value[0]    # µm per sensor pixel
        M      = sensor_value[1]  # magnification
        p_obj  = p_sens / M  # µm per object-plane pixel

        # σ in µm
        σx_um     = σx_fit * p_obj
        σy_um     = σy_fit * p_obj
        σx_um_err = σx_err * p_obj
        σy_um_err = σy_err * p_obj

        # FWHM in µm
        fwhm_x_um     = fwhm_x_px * p_obj
        fwhm_y_um     = fwhm_y_px * p_obj
        fwhm_x_um_err = fwhm_x_err_px * p_obj
        fwhm_y_um_err = fwhm_y_err_px * p_obj

        # Mode Field Diameter (1/e^2) in µm: MFD = 4*σ
        MFD_x_um     = 4 * σx_um
        MFD_y_um     = 4 * σy_um
        MFD_x_um_err = 4 * σx_um_err
        MFD_y_um_err = 4 * σy_um_err

        # Average FWHM in pixels
        fwhm_px_avg = (fwhm_x_px + fwhm_y_px) / 2
        fwhm_px_avg_err = 0.5*np.sqrt(fwhm_x_err_px**2 + fwhm_y_err_px**2)

        # Average FWHM in µm
        fwhm_um_avg = (fwhm_x_um + fwhm_y_um) / 2
        fwhm_um_avg_err = 0.5*np.sqrt(fwhm_x_um_err**2 + fwhm_y_um_err**2)

        # Create output result folder
        output_folder = output_folder + "\\Conventional_2D_Gaussian_fitting"
        if not os.path.isdir(output_folder):
            os.mkdir(output_folder)

        # Plot fit result and save image
        fig, (ax1, ax2) = plt.subplots(1,2, figsize=(10,4), sharex=True, sharey=True)
        ax1.imshow(sub_data, origin='lower', cmap='viridis')
        ax1.set_title('Cropped Data')
        ax2.imshow(fit_img, origin='lower', cmap='viridis')
        ax2.set_title(f'Gaussian Fit (RMSE={rmse:.2f})')
        plt.savefig(output_folder + "\\conven_fiber.png", dpi=300, bbox_inches='tight')

        # Plot residual map and save image
        residual = sub_data - fit_img
        plt.clf()
        plt.figure(figsize=(5,5))
        plt.imshow(residual, origin='lower', cmap='seismic')
        plt.title("Residual Map: Conventional-Gaussian Fit")
        plt.colorbar(label="Residual (ADU)")
        plt.savefig(output_folder + "\\conven_residual.png", dpi=300, bbox_inches='tight')

        # Plot histogram of residual value and save image
        plt.clf()
        plt.hist(residual.ravel(), bins=100)
        plt.savefig(output_folder + "\\conven_hist.png", dpi=300, bbox_inches='tight')

        # Find brightest point from before and after fitting
        max_val_org = np.max(sub_data)
        y_org, x_org = np.unravel_index(np.argmax(sub_data), sub_data.shape)
        max_val_fit = np.max(fit_img)
        y_fit, x_fit = np.unravel_index(np.argmax(fit_img), fit_img.shape)

        # Collect data for summary and log
        self.data_org = sub_data
        self.data_conven = fit_img
        self.log_data_conven.A_px = [A_fit, perr[0]]
        self.log_data_conven.x0_px = [x0_fit, perr[1]]
        self.log_data_conven.y0_px = [y0_fit, perr[2]]
        self.log_data_conven.σx_px = [σx_fit, σx_err]
        self.log_data_conven.σy_px = [σy_fit, σy_err]
        self.log_data_conven.B_px = [B_fit, perr[5]]
        self.log_data_conven.RMSE = [rmse, '-']
        self.log_data_conven.RMSE_ratio = [rmse / A_fit, '-']
        self.log_data_conven.FWHM_x_px = [fwhm_x_px, fwhm_x_err_px]
        self.log_data_conven.FWHM_y_px = [fwhm_y_px, fwhm_y_err_px]
        self.log_data_conven.FWHM_avg_px = [fwhm_px_avg, fwhm_px_avg_err]
        self.log_data_conven.pixel_scale = [p_obj, '-']
        self.log_data_conven.σx_um = [σx_um, σx_um_err]
        self.log_data_conven.σy_um = [σy_um, σy_um_err]
        self.log_data_conven.FWHM_x_um = [fwhm_x_um, fwhm_x_um_err]
        self.log_data_conven.FWHM_y_um = [fwhm_y_um, fwhm_y_um_err]
        self.log_data_conven.FWHM_avg_um = [fwhm_um_avg, fwhm_um_avg_err]
        self.log_data_conven.MFD_x_um = [MFD_x_um, MFD_x_um_err]
        self.log_data_conven.MFD_y_um = [MFD_y_um, MFD_y_um_err]
        self.brightest_point_org = (x_org,y_org)
        self.brightest_point_conven = (x_fit,y_fit)

        progress_bar(100, 100)
        time.sleep(0.05)  # simulate work

    def super_2d_gauss(self, output_folder, data, roi, sensor_value):
        # Get region of interest
        x_min = roi.x_min
        x_max = roi.x_max
        y_min = roi.y_min
        y_max = roi.y_max

        # Select data region
        sub_data = data[y_min:y_max+1, x_min:x_max+1]

        # Build global coordinate grids
        x_inds = np.arange(x_min, x_max+1)
        y_inds = np.arange(y_min, y_max+1)
        x_grid, y_grid = np.meshgrid(x_inds, y_inds)
        xdata = np.vstack((x_grid.ravel(), y_grid.ravel()))
        zdata = sub_data.ravel()

        # Initial guesses
        A0      = sub_data.max() - sub_data.min()
        x0_0    = x_inds[np.argmax(np.sum(sub_data, axis=0))]
        y0_0    = y_inds[np.argmax(np.sum(sub_data, axis=1))]
        σx0     = (x_max - x_min) / 4
        σy0     = (y_max - y_min) / 4
        n0      = 2.0   # start with a mild Super-Gaussian
        B0      = sub_data.min()
        p0      = [A0, x0_0, y0_0, σx0, σy0, n0, B0]

        # Calculate boundary for each parameter fitting
        lb = [0, x_min, y_min, 0, 0, 1.0, 0]
        ub = [np.inf, x_max, y_max, np.inf, np.inf, 10.0, np.inf]

        progress_bar(50, 100)
        time.sleep(0.05)  # simulate work

        # Fit
        popt, pcov = curve_fit(supergauss_2d, xdata, zdata, p0=p0, bounds=(lb,ub))
        A_fit, x0_fit, y0_fit, σx_fit, σy_fit, n_fit, B_fit = popt

        # Parameter errors (1σ)
        perr = np.sqrt(np.diag(pcov))
        σx_err, σy_err, n_err = perr[3], perr[4], perr[5]

        # RMSE of the fit
        fit_img = supergauss_2d((x_grid, y_grid), *popt).reshape(sub_data.shape)
        rmse = np.sqrt(np.mean((fit_img - sub_data)**2))

        # FWHM in pixels for Super-Gaussian:
        # FWHM = 2 * sigma * sqrt(2 * (ln2)^(1/n))
        factor = 2 * np.sqrt(2 * (np.log(2)**(1.0/n_fit)))
        fwhm_x_px = factor * σx_fit
        fwhm_y_px = factor * σy_fit
        fwhm_x_err_px = factor * σx_err
        fwhm_y_err_px = factor * σy_err

        # Unit conversion to object-plane microns
        p_sens = sensor_value[0]    # µm per sensor pixel
        M      = sensor_value[1]  # magnification
        p_obj  = p_sens / M  # µm per object-plane pixel

        # σ in µm
        σx_um     = σx_fit * p_obj
        σy_um     = σy_fit * p_obj
        σx_um_err = σx_err * p_obj
        σy_um_err = σy_err * p_obj

        # FWHM in µm
        fwhm_x_um     = fwhm_x_px * p_obj
        fwhm_y_um     = fwhm_y_px * p_obj
        fwhm_x_um_err = fwhm_x_err_px * p_obj
        fwhm_y_um_err = fwhm_y_err_px * p_obj

        # Mode Field Diameter (1/e^2) in µm: MFD = 4*σ
        MFD_x_um     = 4 * σx_um
        MFD_y_um     = 4 * σy_um
        MFD_x_um_err = 4 * σx_um_err
        MFD_y_um_err = 4 * σy_um_err

        # Average FWHM in pixels
        fwhm_px_avg = (fwhm_x_px + fwhm_y_px) / 2
        fwhm_px_avg_err = 0.5*np.sqrt(fwhm_x_err_px**2 + fwhm_y_err_px**2)

        # Average FWHM in µm
        fwhm_um_avg = (fwhm_x_um + fwhm_y_um) / 2
        fwhm_um_avg_err = 0.5*np.sqrt(fwhm_x_um_err**2 + fwhm_y_um_err**2)
        
        # Create output result folder
        output_folder = output_folder + "\\Super_2D_Gaussian_fitting"
        if not os.path.isdir(output_folder):
            os.mkdir(output_folder)

        # Plot fit result and save image
        fig, (ax1, ax2) = plt.subplots(1,2,figsize=(10,4), sharex=True, sharey=True)
        ax1.imshow(sub_data, origin='lower', cmap='viridis')
        ax1.set_title('Cropped Data')
        ax2.imshow(fit_img, origin='lower', cmap='viridis')
        ax2.set_title(f'Super-Gaussian Fit (RMSE={rmse:.2f})')
        plt.savefig(output_folder + "\\super_fiber.png", dpi=300, bbox_inches='tight')

        # Plot residual map and save image
        residual = sub_data - fit_img
        plt.clf()
        plt.figure(figsize=(5,5))
        plt.imshow(residual, origin='lower', cmap='seismic')
        plt.title("Residual Map: Super-Gaussian Fit")
        plt.colorbar(label="Residual (ADU)")
        plt.savefig(output_folder + "\\super_residual.png", dpi=300, bbox_inches='tight')

        # Plot histogram of residual value and save image
        plt.clf()
        plt.hist(residual.ravel(), bins=100)
        plt.savefig(output_folder + "\\super_hist.png", dpi=300, bbox_inches='tight')

        # Find brightest point from after fitting
        max_val_fit = np.max(fit_img)
        y_fit, x_fit = np.unravel_index(np.argmax(fit_img), fit_img.shape)

        # Collect data for summary and log
        self.data_super = fit_img
        self.log_data_super.A_px = [A_fit, perr[0]]
        self.log_data_super.x0_px = [x0_fit, perr[1]]
        self.log_data_super.y0_px = [y0_fit, perr[2]]
        self.log_data_super.σx_px = [σx_fit, σx_err]
        self.log_data_super.σy_px = [σy_fit, σy_err]
        self.log_data_super.B_px = [B_fit, perr[5]]
        self.log_data_super.RMSE = [rmse, '-']
        self.log_data_super.RMSE_ratio = [rmse / A_fit, '-']
        self.log_data_super.FWHM_x_px = [fwhm_x_px, fwhm_x_err_px]
        self.log_data_super.FWHM_y_px = [fwhm_y_px, fwhm_y_err_px]
        self.log_data_super.FWHM_avg_px = [fwhm_px_avg, fwhm_px_avg_err]
        self.log_data_super.pixel_scale = [p_obj, '-']
        self.log_data_super.σx_um = [σx_um, σx_um_err]
        self.log_data_super.σy_um = [σy_um, σy_um_err]
        self.log_data_super.FWHM_x_um = [fwhm_x_um, fwhm_x_um_err]
        self.log_data_super.FWHM_y_um = [fwhm_y_um, fwhm_y_um_err]
        self.log_data_super.FWHM_avg_um = [fwhm_um_avg, fwhm_um_avg_err]
        self.log_data_super.MFD_x_um = [MFD_x_um, MFD_x_um_err]
        self.log_data_super.MFD_y_um = [MFD_y_um, MFD_y_um_err]
        self.brightest_point_super = (x_fit,y_fit)

        progress_bar(100, 100)
        time.sleep(0.05)  # simulate work

    def special_gauss_elip(self, output_folder, data, roi, sensor_value):
        # Get region of interest
        x_c = roi.x_c
        y_c = roi.y_c
        r = roi.r
        x_min = roi.x_min
        x_max = roi.x_max
        y_min = roi.y_min
        y_max = roi.y_max

        # Select data region
        sub_data = data[y_min : y_max+1,   # rows
                        x_min : x_max+1]   # columns

        # Build the full-coordinate grids in the original pixel frame
        x_inds = np.arange(x_min, x_max+1)
        y_inds = np.arange(y_min, y_max+1)
        x_grid, y_grid = np.meshgrid(x_inds, y_inds)

        # Create a boolean mask for the circle
        mask = (x_grid - x_c)**2 + (y_grid - y_c)**2 <= r**2

        # Apply the mask to pick only pixels inside the circle
        xdata = np.vstack((x_grid[mask], y_grid[mask]))   # shape (2, N_pixels_in_circle)
        zdata = sub_data[mask]                            # shape (N_pixels_in_circle,)

        # Initial guess for parameters
        A0 = np.max(zdata) - np.min(zdata)
        x0_0 = xdata[0].mean()
        y0_0 = xdata[1].mean()
        sigma_x0 = (x_max - x_min) / 4
        sigma_y0 = (y_max - y_min) / 4
        theta0 = 0  # no rotation initially
        B0 = np.min(zdata)
        p0 = [A0, x0_0, y0_0, sigma_x0, sigma_y0, theta0, B0]

        # Calculate boundary for each parameter fitting
        lb = [0, x_min, y_min, 0, 0, -np.pi/4, 0]
        ub = [np.inf, x_max, y_max, np.inf, np.inf, np.pi/4, np.inf]

        progress_bar(50, 100)
        time.sleep(0.05)  # simulate work

        # Perform the fit
        popt, pcov = curve_fit(rotated_gaussian, xdata, zdata, p0=p0, bounds=(lb,ub))
        A_fit, x0_fit, y0_fit, σx_fit, σy_fit, theta_fit, B_fit = popt

        # Parameter errors (1σ)
        perr = np.sqrt(np.diag(pcov))
        σx_err, σy_err, n_err = perr[3], perr[4], perr[5]

        # Evaluate fit and compute residuals
        fit_image = rotated_gaussian((x_grid, y_grid), *popt).reshape(sub_data.shape)
        rmse = np.sqrt(np.mean((fit_image - sub_data)**2))

        # Unit conversion to object-plane microns
        p_sens = sensor_value[0]    # µm per sensor pixel
        M      = sensor_value[1]  # magnification
        p_obj  = p_sens / M  # µm per object-plane pixel

        # FWHM in pixels
        fwhm_factor = 2 * np.sqrt(2 * np.log(2))
        fwhm_x_px = fwhm_factor * σx_fit
        fwhm_y_px = fwhm_factor * σy_fit
        fwhm_x_err_px = fwhm_factor * σx_err
        fwhm_y_err_px = fwhm_factor * σy_err

        # σ in µm
        σx_um     = σx_fit * p_obj
        σy_um     = σy_fit * p_obj
        σx_um_err = σx_err * p_obj
        σy_um_err = σy_err * p_obj

        # FWHM in µm
        fwhm_x_um     = fwhm_x_px * p_obj
        fwhm_y_um     = fwhm_y_px * p_obj
        fwhm_x_um_err = fwhm_x_err_px * p_obj
        fwhm_y_um_err = fwhm_y_err_px * p_obj

        # Mode Field Diameter (1/e^2) in µm: MFD = 4*σ
        MFD_x_um     = 4 * σx_um
        MFD_y_um     = 4 * σy_um
        MFD_x_um_err = 4 * σx_um_err
        MFD_y_um_err = 4 * σy_um_err

        # Average FWHM in pixels
        fwhm_px_avg = (fwhm_x_px + fwhm_y_px) / 2
        fwhm_px_avg_err = 0.5*np.sqrt(fwhm_x_err_px**2 + fwhm_y_err_px**2)

        # Average FWHM in µm
        fwhm_um_avg = (fwhm_x_um + fwhm_y_um) / 2
        fwhm_um_avg_err = 0.5*np.sqrt(fwhm_x_um_err**2 + fwhm_y_um_err**2)

        # Create output result folder
        output_folder = output_folder + "\\Elliptical_Gaussian_fitting"
        if not os.path.isdir(output_folder):
            os.mkdir(output_folder)

        # Plot fit result and save image
        fig, (ax1, ax2) = plt.subplots(1,2, figsize=(10,4), sharex=True, sharey=True)
        ax1.imshow(sub_data, origin='lower', cmap='viridis')
        ax1.set_title('Cropped Data')
        ax2.imshow(fit_image, origin='lower', cmap='viridis')
        ax2.set_title(f'Elliptical Gaussian Fit (RMSE={rmse:.2f})')
        plt.savefig(output_folder + "\\ellip_fiber.png", dpi=300, bbox_inches='tight')

        # Plot residual map and save image
        residual = sub_data - fit_image
        plt.clf()
        plt.figure(figsize=(5, 5))
        plt.imshow(residual, origin='lower', cmap='seismic')
        plt.title("Residual Map: Rotated Elliptical Gaussian")
        plt.colorbar(label="Residual (ADU)")
        plt.savefig(output_folder + "\\ellip_residual.png", dpi=300, bbox_inches='tight')

        # Plot histogram of residual value and save image
        plt.clf()
        plt.hist(residual.ravel(), bins=100)
        plt.savefig(output_folder + "\\ellip_hist.png", dpi=300, bbox_inches='tight')

        # Find brightest point from after fitting
        max_val_fit = np.max(fit_image)
        y_fit, x_fit = np.unravel_index(np.argmax(fit_image), fit_image.shape)

        # Collect data for summary and log
        self.data_elip = fit_image
        self.log_data_elip.A_px = [A_fit, perr[0]]
        self.log_data_elip.x0_px = [x0_fit, perr[1]]
        self.log_data_elip.y0_px = [y0_fit, perr[2]]
        self.log_data_elip.σx_px = [σx_fit, σx_err]
        self.log_data_elip.σy_px = [σy_fit, σy_err]
        self.log_data_elip.B_px = [B_fit, perr[5]]
        self.log_data_elip.RMSE = [rmse, '-']
        self.log_data_elip.RMSE_ratio = [rmse / A_fit, '-']
        self.log_data_elip.FWHM_x_px = [fwhm_x_px, fwhm_x_err_px]
        self.log_data_elip.FWHM_y_px = [fwhm_y_px, fwhm_y_err_px]
        self.log_data_elip.FWHM_avg_px = [fwhm_px_avg, fwhm_px_avg_err]
        self.log_data_elip.pixel_scale = [p_obj, '-']
        self.log_data_elip.σx_um = [σx_um, σx_um_err]
        self.log_data_elip.σy_um = [σy_um, σy_um_err]
        self.log_data_elip.FWHM_x_um = [fwhm_x_um, fwhm_x_um_err]
        self.log_data_elip.FWHM_y_um = [fwhm_y_um, fwhm_y_um_err]
        self.log_data_elip.FWHM_avg_um = [fwhm_um_avg, fwhm_um_avg_err]
        self.log_data_elip.MFD_x_um = [MFD_x_um, MFD_x_um_err]
        self.log_data_elip.MFD_y_um = [MFD_y_um, MFD_y_um_err]
        self.brightest_point_elip = (x_fit,y_fit)

        progress_bar(100, 100)
        time.sleep(0.05)  # simulate work
    
def progress_bar(iteration, total, length=30):
    percent = iteration / total
    filled_length = int(length * percent)
    bar = '█' * filled_length + '-' * (length - filled_length)
    sys.stdout.write(f'\r|{bar}| {percent*100:.1f}%')
    sys.stdout.flush()

def read_fit_data(file_name):
    with fits.open(file_name) as hdul:
        data = hdul[0].data.astype(float)

    return data

def plot_image(output_folder, gauss):
    # Get result data from each fitting
    data_org = gauss.data_org
    data_conven = gauss.data_conven
    data_super = gauss.data_super
    data_elip = gauss.data_elip
    rmse_conven = gauss.log_data_conven.RMSE[0]
    rmse_super = gauss.log_data_super.RMSE[0]
    rmse_elip = gauss.log_data_elip.RMSE[0]
    brightest_point_org = gauss.brightest_point_org
    brightest_point_conven = gauss.brightest_point_conven
    brightest_point_super = gauss.brightest_point_super
    brightest_point_elip = gauss.brightest_point_elip

    # Create figure with 2 rows and 4 columns
    # 2 rows = 1 fitting data, 1 residual
    # 4 columns = 1 original, 3 fitting method
    fig = plt.figure(figsize=(15, 6))
    gs = fig.add_gridspec(2, 4)

    # Plot original data
    ax = fig.add_subplot(gs[0, 0])
    im = ax.imshow(data_org, origin='lower', cmap='viridis')
    ax.set_title(f'Original Cropped Data\nMax@({brightest_point_org[0]},{brightest_point_org[1]})')
    plt.colorbar(im, ax=ax)

    # Plot Conventional Gaussian fitting data and residual map
    ax = fig.add_subplot(gs[0, 1])
    im = ax.imshow(data_conven, origin='lower', cmap='viridis')
    ax.set_title(f'Gaussian Fit (RMSE={rmse_conven:.2f})\nMax@({brightest_point_conven[0]},{brightest_point_conven[1]})')
    plt.colorbar(im, ax=ax)
    ax = fig.add_subplot(gs[1, 1])
    im = ax.imshow(data_conven-data_org, origin='lower', cmap='seismic')
    ax.set_title('Residual Map')
    plt.colorbar(im, ax=ax, label="Residual (ADU)")

    # Plot Super Gaussian fitting data and residual map
    ax = fig.add_subplot(gs[0, 2])
    im = ax.imshow(data_super, origin='lower', cmap='viridis')
    ax.set_title(f'Super-Gaussian Fit (RMSE={rmse_super:.2f})\nMax@({brightest_point_super[0]},{brightest_point_super[1]})')
    plt.colorbar(im, ax=ax)
    ax = fig.add_subplot(gs[1, 2])
    im = ax.imshow(data_super-data_org, origin='lower', cmap='seismic')
    ax.set_title('Residual Map')
    plt.colorbar(im, ax=ax, label="Residual (ADU)")

    # Plot Elliptical Gaussian fitting data and residual map
    ax = fig.add_subplot(gs[0, 3])
    im = ax.imshow(data_elip, origin='lower', cmap='viridis')
    ax.set_title(f'Elliptical Gaussian Fit (RMSE={rmse_elip:.2f})\nMax@({brightest_point_elip[0]},{brightest_point_elip[1]})')
    plt.colorbar(im, ax=ax)
    ax = fig.add_subplot(gs[1, 3])
    im = ax.imshow(data_elip-data_org, origin='lower', cmap='seismic')
    ax.set_title('Residual Map')
    plt.colorbar(im, ax=ax, label="Residual (ADU)")

    # Save image as .png file
    plt.tight_layout()
    plt.savefig(output_folder + "\\Summary_fitting.png", dpi=300, bbox_inches='tight')
    plt.close('all')

def log_data_to_csv(output_folder, gauss):
    # Get current date and time
    current_datetime = datetime.now()
    formatted_datetime_string = current_datetime.strftime("%Y/%m/%d %H:%M:%S")

    # Create and open output csv file
    with open(output_folder + "\\Fitting_raw_data.csv", 'w', newline='') as f:
        writer = csv.writer(f)
        # Write log header
        log_header = [
                    ['Fiber mode fitting result : ' + formatted_datetime_string],
                    ['File : ' + output_folder],
                    ['', 'Conventional', '', 'Super', '', 'Ellip'],
                    ['', 'Value','Error', 'Value', 'Error', 'Value', 'Error']
                ]
        writer.writerows(log_header)
        
        # Write fitting data result of each fitting method
        writer.writerow(['A(px)'] + gauss.log_data_conven.A_px + gauss.log_data_super.A_px + gauss.log_data_elip.A_px)
        writer.writerow(['x0(px)'] + gauss.log_data_conven.x0_px + gauss.log_data_super.x0_px + gauss.log_data_elip.x0_px)
        writer.writerow(['y0(px)'] + gauss.log_data_conven.y0_px + gauss.log_data_super.y0_px + gauss.log_data_elip.y0_px)
        writer.writerow(['sigma_x(px)'] + gauss.log_data_conven.σx_px + gauss.log_data_super.σx_px + gauss.log_data_elip.σx_px)
        writer.writerow(['sigma_y(px)'] + gauss.log_data_conven.σy_px + gauss.log_data_super.σy_px + gauss.log_data_elip.σy_px)
        writer.writerow(['B'] + gauss.log_data_conven.B_px + gauss.log_data_super.B_px + gauss.log_data_elip.B_px)
        writer.writerow(['RMSE(ADU)'] + gauss.log_data_conven.RMSE + gauss.log_data_super.RMSE + gauss.log_data_elip.RMSE)
        writer.writerow(['RMSE/A'] + gauss.log_data_conven.RMSE_ratio + gauss.log_data_super.RMSE_ratio + gauss.log_data_elip.RMSE_ratio)
        writer.writerow(['FWHMx(px)'] + gauss.log_data_conven.FWHM_x_px + gauss.log_data_super.FWHM_x_px + gauss.log_data_elip.FWHM_x_px)
        writer.writerow(['FWHMy(px)'] + gauss.log_data_conven.FWHM_y_px + gauss.log_data_super.FWHM_y_px + gauss.log_data_elip.FWHM_y_px)
        writer.writerow(['Average FWHM(px)'] + gauss.log_data_conven.FWHM_avg_px + gauss.log_data_super.FWHM_avg_px + gauss.log_data_elip.FWHM_avg_px)
        writer.writerow(['Pixel scale(µm/pixel)'] + gauss.log_data_conven.pixel_scale + gauss.log_data_super.pixel_scale + gauss.log_data_elip.pixel_scale)
        writer.writerow(['sigma_x(µm)'] + gauss.log_data_conven.σx_um + gauss.log_data_super.σx_um + gauss.log_data_elip.σx_um)
        writer.writerow(['sigma_y(µm)'] + gauss.log_data_conven.σy_um + gauss.log_data_super.σy_um + gauss.log_data_elip.σy_um)
        writer.writerow(['FWHMx(µm)'] + gauss.log_data_conven.FWHM_x_um + gauss.log_data_super.FWHM_x_um + gauss.log_data_elip.FWHM_x_um)
        writer.writerow(['FWHMy(µm)'] + gauss.log_data_conven.FWHM_y_um + gauss.log_data_super.FWHM_y_um + gauss.log_data_elip.FWHM_y_um)
        writer.writerow(['Average FWHM(µm)'] + gauss.log_data_conven.FWHM_avg_um + gauss.log_data_super.FWHM_avg_um + gauss.log_data_elip.FWHM_avg_um)
        writer.writerow(['MFDx(µm)'] + gauss.log_data_conven.MFD_x_um + gauss.log_data_super.MFD_x_um + gauss.log_data_elip.MFD_x_um)
        writer.writerow(['MFDy(µm)'] + gauss.log_data_conven.MFD_y_um + gauss.log_data_super.MFD_y_um + gauss.log_data_elip.MFD_y_um)

def gaussian_fitting(fit_file, roi_r, sensor_value, roi_xc, roi_yc):
    # Set region of interest
    roi = region_of_interest(roi_xc, roi_yc, roi_r)

    # Read fit data file
    data = read_fit_data(fit_file)

    output_folder = os.path.splitext(fit_file)[0]
    if not os.path.isdir(output_folder):
        os.mkdir(output_folder)

    gauss = gauss_cal('conven') # Create object for gaussian calculation

    print("====== Conventional 2D Gaussian fitting ======")
    progress_bar(1, 100)
    time.sleep(0.05)  # simulate work
    gauss.conven_2d_gauss(output_folder, data, roi, sensor_value)

    print("\n====== Super 2D Gaussian fitting ======")
    progress_bar(1, 100)
    time.sleep(0.05)  # simulate work
    gauss.super_2d_gauss(output_folder, data, roi, sensor_value)

    print("\n====== Rotated Elliptical Gaussian Fit ======")
    progress_bar(1, 100)
    time.sleep(0.05)  # simulate work
    gauss.special_gauss_elip(output_folder, data, roi, sensor_value)

    # Plot summary fitting result
    plot_image(output_folder, gauss)

    # Log raw fitting data
    log_data_to_csv(output_folder, gauss)

    print("\nDONE")
