# The following code refers to the methodology for determining optimal energy management in individual households
# and household networks, which was developed for the master's thesis 'Empowering Homes -
# A bottom-up approach to improving self-sufficiency through optimised electricity management in photovoltaic-based
# household networks' (2023) by Nico Schumann at the University of Trier.
# The main aim is to minimise any interaction, i.e., energy imports and exports with the main power supply by utilising
# large flexible electrical devices or appliances and storage units in form of batteries in the best possible way.
#
# Supervision:
# Prof. Dr Martin Schmidt (supervisor),
# Prof. Dr Volker Schulz (second supervisor)

# Global Imports
import numpy as np  # NumPy is especially required for linear interpolation.
import gurobipy as gp  # Gurobi is used as a solver for the Optimisation Problem (SSM).
from gurobipy import GRB  # REMARK: Make sure to have a valid license for Gurobi.
import matplotlib.pyplot as plt  # Matplotlib is required to generate graphs for visualisation of the results.
import matplotlib.dates as mdates
import pandas as pd  # The Pandas package is required to summarise the data in tabular form.
import zipfile  # Use Zipfile to save all results.


# REMARK: Although not explicitly mentioned, it is important to have the "openpyxl" package installed to export
# an Excel file with the exact result values.


class SelfSufficiencyModel:
    """
    The following class contains an approach to identify optimal times for the operation or charging of large
    loads (referred to as Shiftable Loads (SL)) within an individual household or household network. The
    optimal utilisation of corresponding storage units (batteries) is also determined. Further data can be
    transmitted as required.

    The programme works as follows:
    One or more households are considered a single system, each ideally equipped with a photovoltaic system. A whole
    system usually consists of modules, an inverter and, optionally, a storage unit (battery). While the data on
    modules and inverters are only required to predict the corresponding PV yields based on certain weather data, the
    charging and discharging options of the individual households' batteries play a decisive role in the energy
    management.

    First, forecasts regarding the expected PV yields are carried out. The corresponding results are saved in form
    of plots. The so-called Self Sufficiency Model (SSM; see thesis) is , secondly,  initialised and an optimal solution
    is calculated based on the input data. The corresponding results and individual variables' values are then, thirdly,
    collated and exported graphically and in tabular form.

    Accordingly, as well as visualising the data using a variety of graphs, each individual calculation will also be
    saved as an Excel file, making it easier to view and compare the results.
    """

    def __init__(self, T, tau, H, J, legend_households, legend_sls, start_date, periods, resolution, mip_gap,
                 time_limit, es_stc, ts_noct, ns, cs_temp, etas_pr, etas_inv, gr_progs_h, temp_progs_h, Ns, es_load_ind,
                 ls_sltotal, runtimes_sl, es_slmax, etas_bats, etas_batc, etas_batd, cs_bat, socs_initial, es_batcmax,
                 es_batdmax):

        # Initialise the 'size' of the following model:
        self.T = T  # Number of time periods t under consideration
        self.tau = tau  # Correction factor: Number of time periods t within one hour (here: 4; quarterly hours)
        self.H = H  # Number of Households considered
        self.J = J  # Number of Shiftable Loads (SL) considered

        self.legend_households = legend_households  # Labelling for the respective households
        self.legend_sls = legend_sls  # Labelling for the respective SLs

        # Initialise data on date, time and time span to be considered (number of periods and resolution).
        # This is necessary to obtain clear graphs with correct labels. It will also make it easier to
        # identify the individual energy values in the results in relation to the respective investigation periods.
        self.start_date = start_date  # Initialise start date and time
        self.periods = periods  # Periods is always an integer equal to T (Number of time periods t)
        self.resolution = resolution  # Length of the time periods
        self.x_fmt = mdates.DateFormatter("%Y-%m-%d %H:%M")  # Time Format for the plots
        self.sigma = 24 * self.tau  # If multiple days are considered, a dividing line is drawn between them in
        # the plots to obtain a better overview.

        # The following statement ensures that every current period t can be related to a certain "real" point in time.
        self.times = pd.date_range(self.start_date, periods=self.periods, freq=self.resolution)

        # The following input data are required to calculate the estimated PV energy.
        # They depend on the PV-system to be considered.
        self.es_stc = es_stc  # Nominal Powers under Standard Test Conditions (STC) of the different PV modules
        self.ts_noct = ts_noct  # Nominal Operational Cell Temperatures (NOCT) of the different PV modules
        self.ns = ns  # Numbers of PV modules per household
        self.cs_temp = cs_temp  # Temperature Coefficients for the Power of a PV system
        self.etas_pr = etas_pr  # Performance Ratios that include shading, soiling, transport losses, etc.
        self.etas_inv = etas_inv  # Efficiencies of the Inverters of the different PV system
        self.gr_progs_h = gr_progs_h  # Prognostic Global Radiation data for the respective time period (hourly)
        self.temp_progs_h = temp_progs_h  # Prognostic Ambient Temperature data for the respective time period (hourly)
        self.es_load_ind = es_load_ind  # Fixed Load data for each individual household

        # Initialise some lists in order of their usage during the next methods to obtain data required to calculate or
        # represent the PV energy available in period t and Fixed Load of each household and also for the network.
        self.gr_total = [[] for _ in range(self.H)]  # Global Radiation data in the right format.
        self.temp_total = [[] for _ in range(self.H)]  # Ambient Temperature data in the right format.
        self.t_cell = [[] for _ in range(self.H)]  # Operating Cell Temperature of the PV cells at period t.
        self.e_pv_ind = [[] for _ in range(self.H)]  # List of individual PV yields (per household).
        self.e_pv = []  # Cumulative PV yield available for the network considered.
        self.load_ind = [[] for _ in range(self.H)]  # Individual Fixed Load values
        self.es_l = []  # Cumulative Fixed Load values for the network considered.

        # The following values are fixed constants to calculate the estimated PV energy for the whole investigation
        # period.
        self.g_stc = 1000  # global radiation under STC (fixed).
        self.t_cell_stc = 25  # Operating Cell Temperature under STC (fixed).
        self.t_noct_ambient = 20  # Ambient temperature under NOCT conditions (fixed).
        self.g_noct = 800  # Global Radiation under NOCT conditions (fixed).

        # The following input data is required to solve the Self Sufficiency Problem:
        self.Ns = Ns  # Lists of points in time when it is not possible to use/charge SL j.
        self.ls_sltotal = ls_sltotal  # List of the total amounts of energy required to operate/charge SL j.
        self.runtimes_sl = runtimes_sl  # List of maximum numbers of time periods t in which SL j may operate or should
        # be charged.
        self.es_slmax = es_slmax  # List of maximum amounts of energy to run/charge SL j in one time period t.
        self.etas_bats = etas_bats  # List, containing the self-discharging rates for each of the Batteries.
        self.etas_batc = etas_batc  # List, containing the charging rates for each of the Batteries.
        self.etas_batd = etas_batd  # List, containing the discharging rates for each of the Batteries.
        self.cs_bat = cs_bat  # List, containing the maximum capacity of each Battery.
        self.socs_initial = socs_initial  # List of initial Status of Charge (SOC) of each Battery.
        self.es_batcmax = es_batcmax  # List of maximum amounts of energy that can be charged into each Battery in a
        # single time period t.
        self.es_batdmax = es_batdmax  # List of maximum amounts of energy that can be discharged from each Battery in a
        # single time period t.

        # Initialise the objective function of the Optimisation Problem.
        self.f = None

        # Initialise the Tolerance the Optimisation Problem should be solved with.
        self.mip_gap = mip_gap

        # Initialise a value for the time limit in which the Optimisation Problem is to be solved (in seconds).
        self.time_limit = time_limit

        # Initialise lists for the different Variables of Optimisation Problem. Each of these variables is T-dimensional
        # (i.e., T many entries), each representing the corresponding time periods t.

        # Continuous Variables:
        self.e_im = []  # Import Energy.
        self.e_ex = []  # Export Energy.
        self.e_batc = [[] for _ in range(self.H)]  # Energy, charged into the batteries.
        self.e_batd = [[] for _ in range(self.H)]  # Energy, discharged from the batteries.
        self.soc_bat = [[] for _ in range(self.H)]  # Current Status of Charge of the batteries.
        self.e_sl = [[] for _ in range(self.J)]  # Energy required to charge/run a certain SL.

        # Binary Variables (Decision Variables):
        self.delta_pbatc = [[] for _ in range(self.H)]  # Battery h is charged or not
        self.delta_pbatd = [[] for _ in range(self.H)]  # Battery h is discharged or not
        self.delta_sl = [[] for _ in range(self.J)]  # SL j is running/being charged or not
        self.delta_slon = [[] for _ in range(self.J)]  # SL j is switched on
        self.delta_sloff = [[] for _ in range(self.J)]  # SL j is switched off

        # Value lists of the different continuous Variables (required to plot the different graphs; analogue to the
        # variables):
        self.e_im_values = []
        self.e_ex_values = []
        self.e_batc_values = [[] for _ in range(self.H)]
        self.e_batd_values = [[] for _ in range(self.H)]
        self.soc_bat_values = [[] for _ in range(self.H)]
        self.e_sl_values = [[] for _ in range(self.J)]

        # List of plots to be saved for later interpretation of the results.
        self.plots = []

        # Initialise the Optimisation Model (Self Sufficiency Model (SSM))
        self.model = gp.Model("Self_Sufficiency_Model")

    def get_estimated_pv_data(self):
        # The following steps check that the input weather data on global radiation and ambient temperature is in the
        # correct format required for subsequent calculations. If the data matches the desired resolution, it is
        # just adopted. Otherwise, the data will be interpolated from hourly values in the desired resolution.

        # Global radiation and Ambient Temperature data are mostly available in a 1-hour-resolution. Therefore, it is
        # necessary to calculate corresponding values for each time period t to be considered. For this purpose,the
        # following approach uses linear interpolation to obtain this data.
        # REMARK: For a meaningful interpolation, one hourly value more than the hours actually mapped in the analysis
        # is required to determine the values of the last hour of the observed period (e.g. instead of 72
        # hourly values, 73 are necessary to determine the values of the 72nd hour in the corresponding desired temporal
        # resolution! This applies to Global Radiation data and Ambient Temperature data as well as Fixed Load profile
        # data (see following method).

        for h in range(self.H):
            # Interpolation of the Global Radiation data for each Household, if necessary
            if len(self.gr_progs_h[h]) == (self.T / self.tau) + 1:
                self.gr_total[h] = np.interp(list(range(self.T)), list(range(0, self.T + 1, self.tau)),
                                             self.gr_progs_h[h])

            # If the data is already in the right format, the corresponding list is just adopted.
            elif len(self.gr_progs_h[h]) == self.T:
                self.gr_total[h] = self.gr_progs_h[h]

            else:
                print(f"Inappropriate list length of radiation data for Household {h}")

            # Interpolation of the ambient temperature data for each Household (also depending on its
            # geographical location).
            if len(self.temp_progs_h[h]) == (self.T / self.tau) + 1:
                self.temp_total[h] = np.interp(list(range(self.T)), list(range(0, self.T + 1, self.tau)),
                                               self.temp_progs_h[h])

            # Analogue: If data in a suitable time resolution is available, it is just adopted.
            elif len(self.temp_progs_h[h]) == self.T:
                self.temp_total[h] = self.temp_progs_h[h]

            else:
                print(f"Inappropriate list length of ambient temperature data for Household {h}")

            # The following algorithm (the so-called NOCT-method) calculates the estimated Operating Cell
            # Temperature of the PV modules of each Household in any time period t.
            for t in range(self.T):
                t_cell_t = self.temp_total[h][t] + ((self.gr_total[h][t] / self.g_noct)
                                                    * (self.ts_noct[h] - self.t_noct_ambient))
                self.t_cell[h].append(t_cell_t)

            # In the next step, Algorithm 1 (see chapter 4.3. in the thesis) provides estimated PV energy
            # available in each household in the respective time periods t:
            for t in range(self.T):
                e_pv_t = (1 / self.tau) * self.ns[h] * self.es_stc[h] * (self.gr_total[h][t] / self.g_stc) \
                         * (1 + (self.cs_temp[h] * (self.t_cell[h][t] - self.t_cell_stc))) * self.etas_pr[h] \
                         * self.etas_inv[h]
                self.e_pv_ind[h].append(e_pv_t)

        # Determine the cumulative PV energy available within the whole system for every time period t.
        for t in range(self.T):
            cumulative_e_pv = 0
            for h in range(self.H):
                cumulative_e_pv += self.e_pv_ind[h][t]
            self.e_pv.append(cumulative_e_pv)

        # Two plots are generated and saved below: The first shows Global Radiation and Ambient Temperature for the
        # individual households, while the second shows the individual estimated PV yields as well as the cumulative PV
        # yields in any time period of the investigation.

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6))
        for h in range(self.H):
            cm = plt.get_cmap("Wistia", self.H)
            color = cm(h)
            ax1.plot(self.times, self.gr_total[h], color=color, label="Global Radiation")
        ax1.set_title("GLOBAL RADIATION AND AMBIENT TEMPERATURE PROFILE FOR THE DIFFERENT HOUSEHOLDS")
        ax1.set_ylabel("Power in W/m^2")
        for i in range(0, self.T, self.sigma):
            ax1.axvline(x=self.times[i], color="dimgray", linestyle='--')
        ax1.legend([f"Global Radiation ({name})" for name in legend_households])
        ax1.grid(True)
        for h in range(self.H):
            cm = plt.get_cmap("autumn", self.H)
            color = cm(h)
            ax2.plot(self.times, self.temp_total[h], color=color, label="Ambient Temperature")
        ax2.set_ylabel("Temperature in °C")
        ax2.xaxis.set_major_formatter(self.x_fmt)
        ax2.set_xlabel("Time")
        for i in range(0, self.T, self.sigma):
            ax2.axvline(x=self.times[i], color='dimgray', linestyle='--')
        ax2.legend([f"Ambient Temperature ({name})" for name in legend_households])
        ax2.grid(True)
        fig.autofmt_xdate()

        self.plots.append(fig)

        legend_pv = ["Total PV ENERGY"]
        for name in legend_households:
            legend_pv_ind = f"PV Energy ({name})"
            legend_pv.append(legend_pv_ind)

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(self.times, self.e_pv, color="orange", linewidth=2, label="PV energy at time t")
        for h in range(self.H):
            cm = plt.get_cmap("autumn", self.H)
            color = cm(h)
            ax.plot(self.times, self.e_pv_ind[h], color=color,
                    label=f"Partial prognostic PV energy from Household {h}")
        ax.set_ylabel("Energy in Wh")
        ax.xaxis.set_major_formatter(self.x_fmt)
        ax.set_xlabel("Time")
        ax.set_title("PROGNOSTIC PV ENERGY")
        for i in range(0, self.T, self.sigma):
            ax.axvline(x=self.times[i], color="dimgray", linestyle='--')
        ax.legend(legend_pv)
        ax.grid(True)
        fig.autofmt_xdate()

        self.plots.append(fig)

    def get_total_load_profile(self):
        #  As before, the values for a Fixed Load Profile are interpolated if the desired resolution is not available.
        for h in range(self.H):
            if len(self.es_load_ind[h]) == (self.T / self.tau) + 1:
                self.load_ind[h] = np.interp(list(range(self.T)), list(range(0, self.T + 1, self.tau)),
                                             self.es_load_ind[h])

            # If it is, the corresponding list is just adopted.
            elif len(self.es_load_ind[h]) == self.T:
                self.load_ind[h] = es_load_ind[h]

            else:
                print(f"Inappropriate list length for load profile {h}")

        for t in range(self.T):
            cumulative_load = 0
            for h in range(self.H):
                cumulative_load += self.load_ind[h][t]
            self.es_l.append(cumulative_load)

        legend_fixed_load = ["Fixed Load total"]
        for name in legend_households:
            fixed_load_profile = f"Fixed load ({name})"
            legend_fixed_load.append(fixed_load_profile)

        #  Plot both the individual Load Profiles and a cumulative Load Profile as the sum of the individual Load
        #  Profiles for each time period t.
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(self.times, self.es_l, color="red", linewidth=2, label="Total fixed load at time t")
        for h in range(self.H):
            cm = plt.get_cmap("cool", self.H)
            color = cm(h)
            ax.plot(self.times, self.load_ind[h], color=color, label=f"Fixed load profile of Household {h}")
        ax.set_ylabel("Energy in Wh")
        ax.xaxis.set_major_formatter(self.x_fmt)
        ax.set_xlabel("Time")
        for i in range(0, self.T, self.sigma):
            ax.axvline(x=self.times[i], color="dimgray", linestyle='--')
        ax.set_title("FIXED LOAD PROFILES")
        ax.legend(legend_fixed_load)
        ax.grid(True)
        fig.autofmt_xdate()

        self.plots.append(fig)

    # Everything up to here was just preliminary work to ensure that the results of the following Optimisation Problem
    # are as realistic as possible.
    def solve_self_sufficiency_model(self):
        # Initialise the different Variables.
        # Continuous Variables (with lower bound: 0)
        lbs = [0] * self.T
        self.e_im = self.model.addVars(self.T, vtype=GRB.CONTINUOUS, lb=lbs, name="e_im")
        self.e_ex = self.model.addVars(self.T, vtype=GRB.CONTINUOUS, lb=lbs, name="e_ex")
        for h in range(self.H):
            self.e_batc[h] = self.model.addVars(self.T, vtype=GRB.CONTINUOUS, lb=lbs, name=f"e_batc{h}")
            self.e_batd[h] = self.model.addVars(self.T, vtype=GRB.CONTINUOUS, lb=lbs, name=f"e_batd{h}")
            self.soc_bat[h] = self.model.addVars(self.T, vtype=GRB.CONTINUOUS, lb=lbs, name=f"soc_bat{h}")

        for j in range(self.J):
            self.e_sl[j] = self.model.addVars(self.T, vtype=GRB.CONTINUOUS, lb=lbs, name=f"e_sl{j}")

        # Binary Variables (Since Gurobi allows you to specify a variable type directly (in this case binary),
        # Constraint 18 (see thesis) is obsolete.
        for h in range(self.H):
            self.delta_pbatc[h] = self.model.addVars(self.T, vtype=GRB.BINARY, name=f"delta_pbatc{h}")
            self.delta_pbatd[h] = self.model.addVars(self.T, vtype=GRB.BINARY, name=f"delta_pbatd{h}")
        for j in range(self.J):
            self.delta_sl[j] = self.model.addVars(self.T, vtype=GRB.BINARY, name=f"delta_sl{j}")
            self.delta_slon[j] = self.model.addVars(self.T, vtype=GRB.BINARY, name=f"delta_slon{j}")
            self.delta_sloff[j] = self.model.addVars(self.T, vtype=GRB.BINARY, name=f"delta_sloff{j}")

        # Initialise the Objective Function. As discussed in the thesis, other Objective Functions could also be
        # conceivable for certain purposes. These are provided below in comments.
        #
        # LINEAR VERSION: self.f = gp.quicksum(self.e_im[t] for t in range(self.T))
        #               + gp.quicksum(self.e_ex[t] for t in range(self.T))
        # 'IMPORT ONLY' VERSION: self.f = gp.quicksum(self.e_im[t] * self.e_im[t] for t in range(self.T))

        self.f = gp.quicksum(self.e_im[t] * self.e_im[t] for t in range(self.T)) + \
            gp.quicksum(self.e_ex[t] * self.e_ex[t] for t in range(self.T))

        # Submit the Objective Function
        self.model.setObjective(self.f, GRB.MINIMIZE)

        # The various Constraints of the SSM are set up below. Order and structure relates directly to the thesis.
        #
        # BALANCE EQUATION
        # Constraint 1: Balance Equation.
        for t in range(0, self.T):
            self.model.addConstr(self.e_im[t] + self.e_pv[t]
                                 + gp.quicksum(self.e_batd[h][t] for h in range(self.H)) - self.es_l[t]
                                 - gp.quicksum(self.e_batc[h][t] for h in range(self.H))
                                 - gp.quicksum(self.e_sl[j][t] for j in range(self.J))
                                 - self.e_ex[t] == 0, name="Balance equation")

        # BATTERY RELATED (IN-)EQUATIONS
        # Constraint 2: Battery Equations for the Batteries of each Household.
        for h in range(self.H):
            for t in range(1, self.T):  # only true for t > 0, because there is no period before t == 0
                self.model.addConstr(self.soc_bat[h][t] == (1 - self.etas_bats[h]) * self.soc_bat[h][t - 1]
                                     + self.etas_batc[h] * self.e_batc[h][t - 1]
                                     - self.e_batd[h][t - 1] / self.etas_batd[h],
                                     name=f"Battery equation for Household {h}")

        # Constraint 3: Initial SOC of the Batteries.
        for h in range(self.H):
            self.model.addConstr(self.soc_bat[h][0] == self.socs_initial[h],
                                 name=f"Initial SOC of Battery {h}")

        # Constraint 4: Upper bound for the respective SOC by Capacity.
        for h in range(self.H):
            for t in range(0, self.T):
                self.model.addConstr(self.soc_bat[h][t] <= self.cs_bat[h], name=f"Capacity of Battery {h}")

        # Constraint 5: Maximum amount of energy to be charged into the respective Battery in any time period t.
        for h in range(self.H):
            for t in range(0, self.T):
                self.model.addConstr(self.e_batc[h][t] <= self.delta_pbatc[h][t] * self.es_batcmax[h],
                                     name=f"Maximum charging rate of Battery {h}")

        # Constraint 6: Maximum amount of energy to be discharged from the respective Battery in any time period t.
        for h in range(self.H):
            for t in range(0, self.T):
                self.model.addConstr(self.e_batd[h][t] <= self.delta_pbatd[h][t] * self.es_batdmax[h],
                                     name=f"Maximum discharging rate of Battery {h}")

        # Constraint 7: Avoidance of illogical charging quantities for the Batteries in the last time period.
        for h in range(self.H):
            self.model.addConstr(self.e_batc[h][self.T - 1] <= self.cs_bat[h] - self.soc_bat[h][self.T - 1],
                                 name=f"Avoid illogical charging behaviour of Battery {h} in the last time period")

        # Constraint 8: Avoidance of illogical discharging quantities for the Batteries in the last time period.
        for h in range(self.H):
            self.model.addConstr(self.e_batd[h][self.T - 1] <= self.soc_bat[h][self.T - 1],
                                 name=f"Avoid illogical discharging behaviour of Battery {h}")

        # Constraint 9: Prohibition of simultaneous charging and discharging of the Batteries.
        for h in range(self.H):
            for t in range(0, self.T):
                self.model.addConstr(self.delta_pbatc[h][t] + self.delta_pbatd[h][t] <= 1,
                                     name=f"Avoid illocgical charging/discharging behaviour of Battery {h}")

        # SL RELATED (IN-)EQUATIONS
        # Constraint 10: Total amount of energy scheduled for each SL j.
        for j in range(self.J):
            self.model.addConstr(gp.quicksum(self.e_sl[j][t] for t in range(self.T)) == self.ls_sltotal[j],
                                 name=f"Total load for SL {j}")

        # Constraint 11: Prohibited time periods to run/charge SL j.
        for j in range(self.J):
            for t in self.Ns[j]:  # N is the set of times t, when the car is not available to charge
                self.model.addConstr(self.e_sl[j][t] == 0, name=f"Forbidden times to run/charge SL {j}")

        # Constraint 12: Maximum amount of energy to run/charge SL j in any time period t.
        for j in range(self.J):
            for t in range(0, self.T):
                self.model.addConstr(self.e_sl[j][t] <= self.delta_sl[j][t] * self.es_slmax[j],
                                     name=f"Maximum amount of energy used to run/charge SL {j} in any period t")

        # Constraint 13: Logical condition for the operating/charging times of  SL j (see: Unit Commitment).
        for j in range(self.J):
            for t in range(1, self.T):
                self.model.addConstr(self.delta_sl[j][t] - self.delta_sl[j][t - 1] - self.delta_slon[j][t]
                                     + self.delta_sloff[j][t] == 0, name=f"Logical constraint (UCP) for SL {j}")

        # Constraint 14: Prohibition to switch on/off or charge/discharge SL j at the same time.
        for j in range(self.J):
            for t in range(self.T):
                self.model.addConstr(self.delta_slon[j][t] + self.delta_sloff[j][t] <= 1,
                                     name=f"Do not switch SL {j} on and off at the same time")

        # Constraint 15: Maximum runtime for SLs that are tied to a specific runtime for a single cycle.
        for j in range(self.J):
            self.model.addConstr(gp.quicksum(self.delta_sl[j][t] for t in range(self.T)) <= self.runtimes_sl[j],
                                 name=f"Maximum runtime of SL {j}")
            # In the event that no such restriction is provided (see e.g. electric cars), set r_j = T - 2 (see thesis)

        for j in range(self.J):
            # Constraint 16 (SL j is allowed to run/being charged at any time):
            if not Ns[j]:
                # Constraint 16a: SL j is switched on exactly once, but there is enough time left to run a full cycle.
                self.model.addConstr(gp.quicksum(self.delta_slon[j][t] for t in range(self.T - self.runtimes_sl[j] - 1))
                                     == 1, name=f"Switch on SL {j} exactly once within the resp. time span")

                # Constraint 16b: Prohibition that SL j is switched on 'too late' (see Constraint 16a).
                self.model.addConstr(gp.quicksum(self.delta_slon[j][t] for t in range(self.T - self.runtimes_sl[j]
                                                                                      - 1, self.T)) == 0,
                                     name="Prohibit illogical Sl behaviour")

                # Constraint 16c: SL j is switched off exactly once.
                self.model.addConstr(gp.quicksum(self.delta_sloff[j][t] for t in range(self.T)) == 1,
                                     name=f"Switch off SL {j} exactly once within the resp. time span")

            # Constraint 17 (SL j is not allowed to run/being charged at any time):
            else:
                # Constraint 17a: SL j is switched on exactly once.
                self.model.addConstr(gp.quicksum(self.delta_slon[j][t] for t in range(self.T)) == 1,
                                     name=f"Switch on SL {j} exactly once within the resp. time span")

                # Constraint 17b: SL j is switched off exactly once.
                self.model.addConstr(gp.quicksum(self.delta_sloff[j][t] for t in range(self.T)) == 1,
                                     name=f"Switch off SL {j} exactly once within the resp. time span")

        # Set Tolerance: Accuracy of Gurobi's solution of the SSM.
        self.model.Params.MIPGap = self.mip_gap

        # Set Time Limit: Maximum time before Gurobi terminates.
        self.model.Params.TimeLimit = self.time_limit

        # Solve the model by using the respective input data.
        self.model.optimize()

        # Check the status of the model.
        if self.model.status == GRB.OPTIMAL:
            print("Optimal solution found.")
            # Get the values of the different Variables.
            for v in self.model.getVars():
                print(f"{v.varName}: {v.x}")

            # Get the optimal Objective Function value.
            obj_val_rounded = round(self.model.objVal, 3)
            print(f"Optimal objective function value: {obj_val_rounded}")

            # One could additionally introduce factors gamma_import and gamma_export (each in unit (€/(k)Wh)) in order
            # to obtain a rather economic Perspective.
            self.e_im_values = [self.e_im[t].x for t in range(self.T)]
            self.e_ex_values = [self.e_ex[t].x for t in range(self.T)]
            sum_e_im = round(sum(self.e_im_values), 2)
            sum_e_ex = round(sum(self.e_ex_values), 2)
            print("Total amount of energy imports:", sum_e_im, "Wh")
            print("Total amount of energy exports:", sum_e_ex, "Wh")

        else:
            print("No optimal solution found.")

        # Listing of the Variable values in any time period (required format for the following graphs).
        for h in range(self.H):
            self.e_batc_values[h] = [self.e_batc[h][t].x for t in range(self.T)]
            self.e_batd_values[h] = [self.e_batd[h][t].x for t in range(self.T)]
            self.soc_bat_values[h] = [self.soc_bat[h][t].x for t in range(self.T)]
        for j in range(self.J):
            self.e_sl_values[j] = [self.e_sl[j][t].x for t in range(self.T)]

    def get_profiles(self):
        # In the following, further plots are generated with  matplotlib, which are later exported along
        # with the exact data to analyse the results. However, with a larger number of variables, it might be
        # conceivable to choose different colours or to refer to the Excel table when evaluating the SLs.

        # SSM related plot 1: ENERGY BALANCE PROFILE
        legend_balance = ["PV Energy", "Fixed Load", "Import Energy", "Export Energy"]
        for name in legend_sls:
            new_name = f"Load ({name})"
            legend_balance.append(new_name)

        fig1, ax1 = plt.subplots(figsize=(12, 6))
        ax1.plot(self.times, self.e_pv, color="orange", label="PV Energy at time t")
        ax1.plot(self.times, [-p for p in self.es_l], color="red", label="Fixed Load at time t")
        ax1.plot(self.times, self.e_im_values, linewidth=2, color="seagreen", label="Import Energy")
        ax1.plot(self.times, [-p for p in self.e_ex_values], linewidth=2, color="orangered",
                 label="Export Energy")
        for j in range(self.J):
            cm = plt.get_cmap("viridis", self.J)
            color = cm(j)
            ax1.plot(self.times, [-p for p in self.e_sl_values[j]], color=color, label=f"Load SL {j}")
        ax1.set_title("ENERGY BALANCE")
        ax1.xaxis.set_major_formatter(self.x_fmt)
        fig1.autofmt_xdate()
        ax1.set_xlabel("Time")
        ax1.set_ylabel("Energy in Wh")
        for i in range(0, self.T, self.sigma):
            ax1.axvline(x=self.times[i], color="dimgray", linestyle='--')
        ax1.legend(legend_balance)
        ax1.grid(True)

        self.plots.append(fig1)

        # SSM related plot 2: SHIFTABLE LOADS ONLY
        legend_loads = []
        for name in legend_sls:
            sl_name = f"Load ({name})"
            legend_loads.append(sl_name)

        fig2, ax2 = plt.subplots(figsize=(12, 6))
        for j in range(self.J):
            cm = plt.get_cmap("viridis", self.J)
            color = cm(j)
            ax2.plot(self.times, self.e_sl_values[j], color=color, label=f"Load SL {j}")
        ax2.set_title("LOAD OF THE DIFFERENT SL's")
        ax2.set_xlabel("Time")
        ax2.set_ylabel("Energy in Wh")
        ax2.xaxis.set_major_formatter(self.x_fmt)
        fig2.autofmt_xdate()
        for i in range(0, self.T, self.sigma):
            ax2.axvline(x=self.times[i], color="dimgray", linestyle='--')
        ax2.legend(legend_loads)
        ax2.grid(True)

        self.plots.append(fig2)

        # SSM related plot(s) 3: PROFILE FOR EACH BATTERY INVOLVED
        for h in range(self.H):
            legend_batteries = [f"Energy charged into Battery of {legend_households[h]}",
                                f"Energy drawn from Battery of {legend_households[h]}",
                                f"SOC Battery of {legend_households[h]}"]
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(self.times, self.e_batc_values[h], color="royalblue", label=f"Energy charged into Battery {h}")
            ax.plot(self.times, [-p for p in self.e_batd_values[h]], color="darkorange",
                    label=f"Energy drawn from Battery {h}")
            ax.plot(self.times, self.soc_bat_values[h], color="mediumseagreen", label=f"Energy stored in Battery {h}")
            ax.set_title(f"BATTERY PROFILE {legend_households[h]}")
            ax.xaxis.set_major_formatter(self.x_fmt)
            ax.set_xlabel("Time")
            ax.set_ylabel("Energy in Wh")
            for i in range(0, self.T, self.sigma):
                ax.axvline(x=self.times[i], color="dimgray", linestyle='--')
            ax.legend(legend_batteries)
            fig.autofmt_xdate()
            ax.grid(True)

            self.plots.append(fig)

        # SSM related plot 4: IMPORT AND EXPORT ENERGY ONLY
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(self.times, self.e_im_values, color="seagreen", label="Import Energy")
        ax.plot(self.times, [-p for p in self.e_ex_values], color="orangered", label="Export Energy")
        ax.set_title("IMPORT AND EXPORT ENERGY")
        ax.xaxis.set_major_formatter(self.x_fmt)
        ax.set_xlabel("Time")
        ax.set_ylabel("Energy in Wh")
        for i in range(0, self.T, self.sigma):
            ax.axvline(x=self.times[i], color="dimgray", linestyle='--')
        ax.legend()
        fig.autofmt_xdate()
        ax.grid(True)

        self.plots.append(fig)

    def get_datasets(self):
        # Design of DataFrames containing the various Variable values in an Excel file (5 sheets).

        # Sheet 1: MAIN DATA
        # Data Values:
        results_main = [[] for _ in range(self.T)]
        for t in range(self.T):
            results_main[t] = [self.times[t], self.e_pv[t], self.es_l[t], self.e_im_values[t], self.e_ex_values[t]]
            for h in range(self.H):
                results_main[t].append(self.e_batc_values[h][t])
                results_main[t].append(self.e_batd_values[h][t])
                results_main[t].append(self.soc_bat_values[h][t])
            for j in range(self.J):
                results_main[t].append(self.e_sl_values[j][t])

        # Column Headers:
        columns_main = ["Time", "PV Energy (total)", "Fixed Load (total)", "Import Energy", "Total Export Energy"]
        for h in range(self.H):
            columns_main.append(f"Battery charged ({legend_households[h]})")
            columns_main.append(f"Battery discharged ({legend_households[h]})")
            columns_main.append(f"SOC Battery {legend_households[h]}")
        for j in range(self.J):
            columns_main.append(f"Load {legend_sls[j]}")

        # Get Dataset and round entries to 2 decimals.
        dataset_main = pd.DataFrame([data for data in results_main], columns=columns_main)
        dataset_main_rounded = dataset_main.round(2)

        # Sheet 2: TOTALS (including: sum of imported, exported, PV produced and fixed load energy).
        # Data Values:
        sum_p_im = round(sum(self.e_im_values), 2)
        sum_p_ex = round(sum(self.e_ex_values), 2)
        sum_p_pv = round(sum(self.e_pv), 2)
        sum_ps_l = round(sum(self.es_l), 2)
        results_objective = [sum_p_im, sum_p_ex, sum_p_pv, sum_ps_l]

        # Column Headers:
        columns_objective = ["Total Import Energy", "Total Export Energy", "Total PV Energy", "Total Fixed Load"]

        # Get Dataset and round entries to 2 decimals.
        dataset_objective = pd.DataFrame([results_objective], columns=columns_objective)

        # Sheet 3: PV ENERGY
        # Data Values:
        results_pv = [[] for _ in range(self.T)]
        for t in range(self.T):
            results_pv[t] = [self.times[t], self.e_pv[t]]
            for h in range(self.H):
                results_pv[t].append(self.e_pv_ind[h][t])

        # Column Headers:
        columns_pv = ["Time", "PV Energy (total)"]
        for h in range(self.H):
            columns_pv.append(f"Partial PV Energy ({legend_households[h]})")

        # Get Dataset and round entries to 2 decimals.
        dataset_pv = pd.DataFrame([data for data in results_pv], columns=columns_pv)
        dataset_pv_rounded = dataset_pv.round(2)

        # Sheet 4: FIXED LOAD
        # REMARK: For input data sets in a suitable resolution this step might be redundant but useful to
        # compare stereotypical household profiles.
        # Data Values:
        results_load = [[] for _ in range(self.T)]
        for t in range(self.T):
            results_load[t] = [self.times[t], self.es_l[t]]
            for h in range(self.H):
                results_load[t].append(self.load_ind[h][t])

        # Column Headers:
        columns_load = ["Times", "Fixed Load (total)"]
        for h in range(self.H):
            columns_load.append(f"Fixed Load ({legend_households[h]})")

        # Get Dataset and round entries to 2 decimals.
        dataset_load = pd.DataFrame([data for data in results_load], columns=columns_load)
        dataset_load_rounded = dataset_load.round(2)

        # Sheet 5: BATTERIES
        # Data Values:
        results_battery = [[] for _ in range(self.T)]
        for t in range(self.T):
            results_battery[t] = [self.times[t]]
            for h in range(self.H):
                results_battery[t].append(self.e_batc_values[h][t])
                results_battery[t].append(self.e_batd_values[h][t])
                results_battery[t].append(self.soc_bat_values[h][t])

        # Column Headers:
        columns_battery = ["Time"]
        for h in range(self.H):
            columns_battery.append(f"Battery charged ({legend_households[h]})")
            columns_battery.append(f"Battery discharged ({legend_households[h]})")
            columns_battery.append(f"SOC Battery ({legend_households[h]})")

        # Get Dataset and round entries to 2 decimals.
        dataset_battery = pd.DataFrame([data for data in results_battery], columns=columns_battery)
        dataset_battery_rounded = dataset_battery.round(2)

        # Generate and export Excel file.
        with pd.ExcelWriter("DATA FRAMES.xlsx") as writer:
            dataset_main_rounded.to_excel(writer, sheet_name="MAIN DATA SET", index=False)
            dataset_objective.to_excel(writer, sheet_name="TOTALS", index=False)
            dataset_pv_rounded.to_excel(writer, sheet_name="PV ENERGY", index=False)
            dataset_load_rounded.to_excel(writer, sheet_name="FIXED LOAD", index=False)
            dataset_battery_rounded.to_excel(writer, sheet_name="BATTERIES", index=False)

        with zipfile.ZipFile("Self-Sufficiency-Data.zip", "x") as zf:
            zf.write("DATA FRAMES.xlsx", "DATA FRAMES.xlsx")
            for plot in self.plots:
                title = plot.get_axes()[0].get_title()
                if title:
                    filename = f"{title}.png"
                    plot.savefig(filename)
                    zf.write(filename, filename)


# SUBMIT DATA:
if __name__ == "__main__":
    """
    The data required for the calculations is transferred below. The corresponding list lengths along with further 
    details are commented to avoid input errors. 
    The data itself is currently 'set' to H1 - Spring (see Thesis) by default, but can be modified as desired. All other 
    data examined in the thesis are also available below at the appropriate point: The whole sample data is initially 
    presented and then the data set of interest is submitted in a separate row.
    """

    # MODEL SIZE DATA
    T = 288  # Number of time periods t (Default: 288; -> 72h in quarter-hourly resolution).
    tau = 4  # Resolution-factor (e.g.: tau = 4 -> quarter-hourly resolution).
    H = 1  # Number of households involved.
    J = 4  # Number of SLs involved.

    # LABELLING DATA
    # Labelling of the Households or Systems involved (as discussed in the Thesis it would also be conceivable to add
    # only energy producing or storage systems).
    # List legend_households has LIST LENGTH H
    legend_households_h1 = ["Household 1"]
    legend_households_h2 = ["Household 2"]
    legend_households_h3 = ["Household 3"]
    legend_households_h4 = ["Household 4"]
    # SUBMISSION:
    legend_households = legend_households_h1

    # Labelling of the SLs involved.
    # List legend_sls has LIST LENGTH J
    legend_sls_h1 = ["Washing Machine 1", "Dryer 1", "Dishwasher 1", "Electric Car 1"]
    legend_sls_h2 = ["Washing Machine 2", "Dryer 2", "Dishwasher 2", "Electric Car 2"]
    legend_sls_h3 = ["Washing Machine 3", "Dryer 3", "Dishwasher 3", "Electric Car 3"]
    legend_sls_h4 = ["Washing Machine 4", "Dryer 4", "Dishwasher 4", "Electric Car 4"]
    # SUBMISSION:
    legend_sls = legend_sls_h1

    # TERMINATION DATA
    # Set Tolerance
    default_mip_gap = 0.05
    # SUBMISSION:
    mip_gap = default_mip_gap

    # Set Time Limit
    default_time_limit = 180 * 60  # 2h
    # SUBMISSION:
    time_limit = default_time_limit

    # DATE AND FORMAT DATA
    # Set start date and time, as well as the number of time periods their length. The model days mentioned in the
    # thesis are used as starting points.
    periods = T  # Default value! Please do not change!
    start_date_spring = '2022-03-11'
    start_date_summer = '2022-06-14'
    start_date_autumn = '2022-09-14'
    start_date_winter = '2022-12-15'
    # SUBMISSION:
    start_date = start_date_spring
    resolution = '15min'

    # WEATHER RELATED DATA
    # As discussed in the thesis it is also conceivable to initially submit data in the suitable temporal
    # resolution, this would lead to a LIST LENGTH T. However, as the required data on Global Radiation and
    # Ambient Temperature was only available in hourly resolution, these are submitted accordingly in hourly values.
    # The data can be changed or rearranged as desired. For this purpose, the various data sets were merged from the
    # individual days in order to facilitate any rearrangement.
    # A 'daily dataset' consists of 24 values (00:00 to 23:00). In addition, the 00:00 value of the day after the
    # investigation period is added.

    # Global Radiation data for each Season (Unit: W/m^2; Temporal resolution: 1h; LIST LENGTH: (T/tau) + 1)
    gr_dataset_spring_day1 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.556, 38.333, 180.278, 331.389, 466.667, 566.111, 613.889,
                              606.111, 546.944, 347.222, 170.0, 116.111, 19.444, 0.0, 0.0, 0.0, 0.0, 0.0]
    gr_dataset_spring_day2 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.556, 36.944, 87.222, 106.111, 168.333, 236.111, 400.0,
                              279.444, 234.444, 276.389, 225.0, 98.333, 14.444, 0.0, 0.0, 0.0, 0.0, 0.0]
    gr_dataset_spring_day3 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.111, 21.944, 135.556, 386.111, 479.722, 573.889, 470.0,
                              295.0, 128.611, 111.667, 92.5, 28.889, 9.167, 0.0, 0.0, 0.0, 0.0, 0.0]
    gr_dataset_spring_total = gr_dataset_spring_day1 + gr_dataset_spring_day2 + gr_dataset_spring_day3 + [0.0]

    gr_dataset_summer_day1 = [0.0, 0.0, 0.0, 0.0, 0.0, 6.389, 80.278, 223.333, 387.222, 550.278, 805.833, 810.556,
                              889.167, 925.278, 884.722, 812.222, 692.222, 554.722, 413.611, 202.222, 66.111, 30.0,
                              0.278, 0.0]
    gr_dataset_summer_day2 = [0.0, 0.0, 0.0, 0.0, 0.0, 6.667, 79.167, 222.222, 385.833, 548.056, 692.778, 825.556,
                              889.444, 913.333, 892.222, 834.167, 734.444, 581.944, 382.778, 271.667, 117.222, 16.111,
                              0.0, 0.0]
    gr_dataset_summer_day3 = [0.0, 0.0, 0.0, 0.0, 0.0, 4.722, 68.611, 203.333, 367.222, 536.667, 677.222, 797.778,
                              884.722, 920.0, 913.611, 857.5, 741.944, 590.278, 328.333, 269.722, 130.0, 27.5, 0.0, 0.0]
    gr_dataset_summer_total = gr_dataset_summer_day1 + gr_dataset_summer_day2 + gr_dataset_summer_day3 + [0.0]

    gr_dataset_autumn_day1 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 10.278, 52.222, 70.278, 212.222, 439.722, 488.056,
                              310.556, 175.556, 225.833, 242.5, 62.222, 40.556, 16.111, 0.0, 0.0, 0.0, 0.0]
    gr_dataset_autumn_day2 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 8.611, 62.5, 182.222, 448.611, 461.389, 529.722,
                              468.889, 516.944, 338.056, 290.0, 116.667, 74.167, 11.111, 0.0, 0.0, 0.0, 0.0]
    gr_dataset_autumn_day3 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.333, 60.833, 182.222, 261.389, 305.556, 316.389,
                              245.556, 184.444, 261.944, 243.056, 216.111, 81.389, 17.778, 0.0, 0.0, 0.0, 0.0]
    gr_dataset_autumn_total = gr_dataset_autumn_day1 + gr_dataset_autumn_day2 + gr_dataset_autumn_day3 + [0.0]

    gr_dataset_winter_day1 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 4.167, 28.056, 51.667, 86.111, 106.667, 165.278,
                              190.0, 69.444, 13.056, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    gr_dataset_winter_day2 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 5.0, 50.833, 169.167, 281.944, 311.944, 281.944,
                              215.556, 115.0, 20.556, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    gr_dataset_winter_day3 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.333, 31.389, 65.278, 90.0, 112.778, 118.611,
                              88.889, 47.778, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    gr_dataset_winter_total = gr_dataset_winter_day1 + gr_dataset_winter_day2 + gr_dataset_winter_day3 + [0.0]

    # If multiple households are involved, the main list has to contain a corresponding number of data lists
    # (i.e., LIST LENGTH H).
    # Remark: As these are generally prognostic and not historic data, they are already named accordingly.
    # SUBMISSION:
    gr_progs_h = [gr_dataset_spring_total]

    # Ambient Temperature data for each Season (Unit: °C; Temporal resolution: 1h; LIST LENGTH: (T/tau) + 1)
    temp_dataset_spring_day1 = [8.7, 7.8, 5.9, 7, 5.2, 5.6, 4.2, 3.5, 4.5, 6.1, 7.5, 10.5, 12.6, 13.9, 14.5, 14.4, 13.6,
                                13.1, 12.4, 10.3, 9.7, 9.3, 10.1, 10]
    temp_dataset_spring_day2 = [9.8, 9.5, 8.8, 8.8, 8.6, 8.4, 8.5, 7.9, 7.8, 9.2, 9.8, 10.4, 11.6, 11.6, 12, 13, 13,
                                12.7, 12, 10.3, 9.1, 8.9, 9.1, 7.1]
    temp_dataset_spring_day3 = [6.9, 6.4, 5.4, 5.2, 4.9, 4.4, 3.7, 4.1, 4.3, 6.5, 8.6, 11.1, 13.3, 14.7, 13.7, 13.6,
                                12.4, 11.7, 10.6, 9.8, 9.3, 8.5, 8.3, 7.8]
    temp_dataset_spring_total = temp_dataset_spring_day1 + temp_dataset_spring_day2 + temp_dataset_spring_day3 + [7.8]

    temp_dataset_summer_day1 = [14.3, 13.4, 12.1, 10.5, 9.7, 9.2, 9.1, 9.5, 12.6, 14.7, 17.4, 18.8, 20.7, 22.1, 23, 24,
                                25.3, 25.9, 25.5, 26, 24, 22.5, 26.5, 18.7]
    temp_dataset_summer_day2 = [16.6, 16.2, 14.6, 13.8, 13.3, 12.2, 12.1, 11.7, 15.2, 17.4, 19.8, 22.3, 25.9, 27.3,
                                27.1, 28.3, 29.8, 30.2, 29.4, 29.8, 28.4, 27, 23.3, 22.3]
    temp_dataset_summer_day3 = [21, 19.4, 17.6, 17.8, 17.4, 15.5, 15.8, 15.9, 18.9, 20.3, 22.3, 24.5, 25.1, 26.7, 27,
                                27.7, 28, 28.2, 26.6, 25.7, 24.7, 22.5, 19.9, 18.7]
    temp_dataset_summer_total = temp_dataset_summer_day1 + temp_dataset_summer_day2 + temp_dataset_summer_day3 + [17.9]

    temp_dataset_autumn_day1 = [17.9, 17.3, 18.1, 17.1, 16.7, 16.6, 16.5, 16.4, 16.5, 16.6, 17.2, 18.8, 21.3, 21.6,
                                22.3, 22.1, 22.4, 23, 17.3, 17.6, 17, 16.7, 16.7, 16.4]
    temp_dataset_autumn_day2 = [16.5, 16.5, 16.7, 16.8, 16.9, 15, 13.1, 13.1, 13.5, 13.6, 15, 16.3, 16.3, 17.8, 18.9,
                                19.2, 18.6, 18.1, 17.8, 17.3, 15.8, 15.6, 14.9, 14.4]
    temp_dataset_autumn_day3 = [14.2, 13.8, 13.5, 13, 12.6, 12.5, 12.2, 12.1, 12.1, 12.3, 13.1, 13.7, 14.5, 15.5, 11.7,
                                12.9, 12.9, 13.1, 12.1, 11.8, 11.4, 10.9, 10.6, 18.7]
    temp_dataset_autumn_total = temp_dataset_autumn_day1 + temp_dataset_autumn_day2 + temp_dataset_autumn_day3 + [9]

    temp_dataset_winter_day1 = [-3.8, -4.2, -4.4, -5.2, -5.2, -5.2, -5.6, -5.6, -5.5, -5.6, -6.1, -5.9, -5.5, -5.3,
                                -4.3, -3.4, -3.5, -4.1, -4.6, -5, -5.7, -5.7, -6.2, -6.6]
    temp_dataset_winter_day2 = [-7.1, -7.2, -7.2, -7.4, -7.8, -8.6, -9.5, -9.6, -9.9, -9.5, -8.5, -7.3, -5.8, -3.3,
                                -1.9, -1.6, -2.8, -3.8, -4.8, -5, -5.9, -6.1, -6.7, -7.2]
    temp_dataset_winter_day3 = [-7.8, -7.5, -7.8, -7.8, -7.9, -7.9, -8, -8.5, -8.4, -8.8, -8.7, -7.6, -7.1, -7, -6.5,
                                -6.5, -6.5, -7.1, -7.1, -7.3, -7.5, -7.6, -7.9, -8]
    temp_dataset_winter_total = temp_dataset_winter_day1 + temp_dataset_winter_day2 + temp_dataset_winter_day3 + [-8.3]

    # If multiple households are involved, the list has to contain a corresponding number of data lists
    # (i.e., LIST LENGTH H).
    # Remark: As these are generally prognostic and not historic data, they are already named accordingly.
    # SUBMISSION
    temp_progs_h = [temp_dataset_spring_total]

    # PV ENERGY RELATED DATA
    # List of 'nominal hourly energy' values under STC (discussed in the thesis in 4.1. and 4.3.) of individual
    # households' PV modules (Unit: Wh; LIST LENGTH: H). These values normally refer to a single module. However,
    # if only the total output of the system is known, enter this value.
    e_stc_h1 = 18000
    e_stc_h2 = 3280
    e_stc_h3 = 11720
    e_stc_h4 = 3690
    # SUBMISSION:
    es_stc = [e_stc_h1]

    # Number of PV modules of each Household (LIST LENGTH: H).
    # Set to 1 if only the total output of the PV modules is known.
    # SUBMISSION:
    ns = [1]

    # List of NOCT values for the PV modules of each Household (Unit: °C; LIST LENGTH: H).
    default_t_noct = 41
    # SUBMISSION:
    ts_noct = [default_t_noct]

    # List of Temperature Coefficients of the PV systems of each Household (no unit, LIST LENGTH: H).
    default_c_temp = -0.035
    # SUBMISSION:
    cs_temp = [default_c_temp]

    # List of Performance Ratios of the PV systems of each Household (LIST LENGTH: H).
    default_eta_pr = 0.7
    # SUBMISSION:
    etas_pr = [default_eta_pr]

    # List of Efficiencies of the different inverters of each Household (LIST LENGTH: H).
    default_eta_inv_h13 = 0.975  # Efficiencies for the Inverters of H1 and H3.
    default_eta_inv_h24 = 0.972  # Efficiencies for the Inverters of H2 and H4.
    # SUBMISSION:
    etas_inv = [default_eta_inv_h13]

    # FIXED LOAD DATA
    # In the following, the days for each households' Fixed Load Profiles are again noted individually so that they can
    # be rearranged and manipulated as desired for further research.
    h1_day1 = [48.3325, 39.4175, 32.25, 32.75, 31.1675, 41.3325, 38.3325, 40.5825, 33.5, 32.6675, 38.25, 38.75, 39.1675,
               33.5825, 32.75, 31.1675, 54.5825, 36.3325, 38.9175, 26.5, 35.6675, 35.9175, 37.0825, 38.8325, 76.5825,
               85.045, 79.175, 189.5275, 146.6975, 243.925, 376.65, 97.8125, 82.265, 75.07, 160.8325, 62.505, 122.635,
               27.85, 65.7525, 16.63, 127.7725, 213.525, 196.1125, 348.7625, 21.915, 329.5025, 68.24, 10.25, 0.0, 0.0,
               0.0, 0.0, 0.0, 3.0325, 17.0325, 36.2425, 15.355, 0.0, 16.9975, 10.255, 23.0925, 36.4675, 30.725, 38.375,
               63.57, 89.2125, 192.82, 63.355, 52.66, 171.97, 55.5075, 37.6, 51.555, 51.93, 43.46, 38.1475, 44.695,
               42.5375, 47.5225, 41.94, 50.57, 67.7075, 81.7375, 77.7725, 85.1925, 92.69, 81.8075, 57.575, 84.3775,
               75.09, 72.4425, 86.4925, 76.9825, 35.015, 31.43, 38.925]
    h1_day2 = [40.83, 37.2825, 43.79, 37.7625, 37.095, 44.1375, 44.34, 41.07, 34.155, 33.82, 40.23, 41.725, 45.53,
               39.0925, 35.88, 40.13, 42.625, 42.4925, 39.575, 37.1075, 33.52, 285.47, 76.625, 84.88, 93.6675, 63.2,
               64.52, 64.17, 71.76, 57.77, 52.8775, 35.9675, 4.8975, 1.8175, 14.21, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
               0.0, 0.0, 96.955, 0.0, 0.0, 7.98, 20.63, 28.9725, 11.5075, 74.5175, 25.8, 0.0, 0.0, 125.6075, 179.59,
               234.3575, 237.855, 184.2425, 27.6925, 34.6575, 36.325, 38.9875, 46.39, 38.6975, 39.2075, 51.8, 46.955,
               88.0625, 167.85, 95.575, 320.645, 63.9225, 170.72, 73.675, 76.1575, 156.97, 65.225, 67.12, 69.88,
               63.6175, 164.6775, 195.8425, 54.955, 44.8425, 49.17, 41.1825, 40.92, 38.95, 32.9375, 40.595, 40.9775,
               42.0575, 35.6025, 38.1675]
    h1_day3 = [36.45, 36.68, 37.6575, 36.895, 34.555, 35.0025, 32.92, 37.6875, 32.6575, 41.9025, 41.3625, 36.46,
               39.1125, 32.5675, 32.375, 41.64, 30.98, 39.8125, 41.2325, 44.695, 174.835, 107.275, 111.12, 100.74,
               359.04, 423.28, 61.8975, 49.835, 239.63, 92.045, 59.89, 41.9025, 173.3575, 35.49, 32.9425, 19.155,
               149.61, 6.7275, 31.135, 33.775, 37.0125, 23.1475, 18.5325, 20.2625, 75.765, 40.6175, 51.0825, 30.195,
               19.0125, 31.025, 3.5675, 18.28, 41.89, 58.4125, 50.7225, 82.5725, 76.52, 44.5875, 46.5325, 27.625,
               25.185, 37.67, 40.985, 46.4575, 58.4075, 46.905, 60.8, 174.7125, 422.11, 400.2225, 337.8525, 61.5575,
               175.9625, 89.3175, 460.6075, 89.6, 102.8875, 358.375, 604.1375, 85.0125, 254.7775, 263.3975, 90.515,
               393.5325, 227.645, 47.9375, 54.1175, 43.8825, 52.2575, 41.5825, 37.85, 41.7225, 41.8025, 37.415, 40.37,
               30.4875]
    h2_day1 = [47.5825, 49.3325, 49.75, 52.75, 51.6675, 40.5, 70.8325, 50.0, 54.75, 40.5825, 55.5, 50.0, 55.4175,
               49.1675, 47.1675, 65.0825, 45.3325, 48.9175, 43.9175, 62.4725, 51.7025, 142.19, 193.4975, 193.08,
               200.6625, 185.8325, 195.2125, 194.735, 207.58, 119.1525, 321.435, 157.9375, 80.34, 92.5325, 261.155,
               155.045, 160.8925, 133.6875, 110.2625, 117.8875, 310.4825, 290.185, 276.3125, 257.9375, 281.3975,
               242.385, 239.93, 452.595, 562.67, 309.045, 191.865, 232.83, 223.435, 254.645, 509.5425, 115.81, 142.59,
               315.4, 167.545, 255.7225, 253.385, 220.4775, 218.15, 230.7025, 224.36, 209.34, 229.9775, 101.035, 66.62,
               57.0, 62.78, 54.115, 69.7575, 73.5825, 89.5275, 116.99, 104.64, 70.45, 87.975, 138.07, 188.49, 211.765,
               201.7925, 220.94, 315.1275, 259.3325, 256.4075, 170.9675, 88.9575, 125.7775, 119.9775, 80.605, 102.3825,
               48.51, 45.0375, 59.8125]
    h2_day2 = [61.0825, 395.8325, 62.0825, 51.3325, 72.9175, 54.5825, 67.25, 47.6675, 66.1675, 72.5825, 52.1675, 66.25,
               51.0, 66.1675, 66.3325, 47.0, 73.6675, 54.9175, 67.4175, 63.8325, 46.73, 79.665, 188.3275, 206.5625,
               184.2325, 207.6525, 213.2575, 327.725, 236.115, 281.7, 455.32, 264.4725, 188.12, 118.9375, 152.19,
               151.035, 104.3875, 99.0175, 86.2825, 111.87, 90.465, 111.1025, 311.94, 498.0125, 392.84, 267.2675,
               753.3475, 442.7325, 447.7525, 366.375, 471.7425, 227.1175, 223.5475, 229.7, 195.4675, 204.695, 201.1275,
               117.82, 62.8375, 70.52, 76.395, 66.1625, 77.03, 59.5375, 71.365, 72.965, 83.6425, 68.1475, 83.1275,
               214.0975, 227.06, 202.725, 203.92, 254.2825, 216.3625, 230.1725, 154.45, 71.6075, 69.5875, 84.99,
               87.7275, 63.6075, 73.7575, 62.9425, 92.5975, 50.0775, 63.5875, 59.3025, 94.3925, 85.8575, 228.8325,
               592.4625, 64.9525, 58.83, 237.4425, 68.2825]
    h2_day3 = [46.5825, 70.5, 55.5825, 73.25, 58.1675, 58.8325, 72.5, 60.5825, 74.3325, 50.8325, 74.5, 57.5, 75.0825,
               56.75, 67.0, 72.25, 57.5, 77.9175, 52.9175, 80.9175, 58.1675, 161.9175, 187.0825, 269.25, 224.9175,
               194.8325, 205.4175, 185.0825, 249.3325, 188.7175, 65.8675, 58.7575, 63.5, 70.245, 60.8325, 75.5, 52.995,
               76.25, 59.4175, 64.9175, 50.8275, 51.5825, 58.7375, 69.56, 72.625, 51.3275, 195.9, 184.57, 199.7925,
               180.9225, 192.535, 145.8625, 54.97, 66.67, 53.8625, 69.9275, 53.3025, 121.69, 52.0875, 69.4125, 78.6325,
               417.9675, 532.45, 378.53, 369.9225, 79.665, 97.6675, 119.1875, 224.0825, 342.2225, 705.95, 225.4175,
               293.0, 367.1675, 496.5825, 266.1675, 212.0, 220.0825, 75.75, 112.4175, 86.0, 95.6675, 122.8325, 90.4175,
               92.6675, 90.1675, 81.4175, 49.4175, 68.9175, 52.0825, 64.6675, 168.3325, 145.0, 103.0825, 54.0825, 54.75]
    h3_day1 = [44.9975, 50.8675, 34.235, 38.0425, 41.8525, 40.3425, 62.5175, 48.535, 28.9875, 35.705, 50.655, 38.94,
               35.585, 50.7725, 30.235, 27.1425, 46.06, 33.96, 22.775, 40.6725, 62.87, 42.3525, 44.475, 58.7825, 63.32,
               62.3925, 84.4175, 66.5825, 71.5, 79.5825, 131.5825, 46.415, 109.9975, 86.8425, 216.5175, 87.9325,
               219.825, 106.6375, 159.865, 133.5075, 80.4425, 159.7725, 93.5875, 110.3725, 130.8475, 143.1875, 143.49,
               169.48, 227.7525, 125.62, 224.9275, 297.365, 287.0575, 254.2825, 227.3575, 254.2075, 221.545, 371.2575,
               303.79, 503.315, 601.5075, 394.8075, 161.4375, 140.6675, 149.26, 110.1725, 277.6175, 297.665, 155.1925,
               167.3725, 346.0875, 258.075, 235.18, 127.19, 122.61, 112.4925, 117.235, 133.435, 103.65, 122.115,
               141.155, 118.11, 117.145, 120.685, 104.32, 104.4775, 136.0125, 116.1025, 64.945, 92.77, 87.8925, 126.32,
               43.345, 52.3225, 42.475, 49.1025]
    h3_day2 = [41.6275, 45.6925, 54.6375, 33.74, 62.6925, 29.0075, 33.5525, 55.8175, 35.0425, 55.1825, 50.665, 37.6225,
               42.0475, 33.765, 41.8575, 66.595, 41.155, 48.37, 45.2375, 51.075, 35.355, 40.6225, 35.63, 61.8875,
               54.2375, 54.555, 85.7325, 76.825, 80.1875, 72.2875, 85.2725, 134.28, 120.785, 207.3475, 243.88, 171.8975,
               209.565, 220.085, 322.0975, 234.9825, 223.95, 250.8, 244.35, 241.85, 337.9425, 164.1725, 252.4475,
               264.735, 246.9175, 239.195, 485.655, 231.145, 244.7325, 140.6425, 325.3075, 225.1075, 114.5375, 258.465,
               255.7375, 369.795, 302.5225, 339.1625, 308.955, 322.6725, 385.665, 315.02, 172.555, 137.0925, 399.4675,
               334.3575, 212.7425, 104.215, 89.1075, 108.98, 100.9025, 95.33, 100.6475, 87.835, 433.8625, 385.4325,
               114.0725, 110.7975, 105.32, 94.385, 247.5275, 47.4475, 36.555, 54.365, 101.325, 45.5, 34.1825, 27.3675,
               44.17, 30.45, 46.3075, 42.3725]
    h3_day3 = [53.06, 30.79, 34.1875, 50.385, 41.87, 37.5775, 51.265, 43.755, 43.5175, 61.3975, 41.86, 34.7, 50.68,
               38.6675, 33.165, 53.1175, 48.1525, 41.5775, 66.4225, 50.7775, 50.565, 32.765, 63.685, 43.3275, 114.2525,
               60.185, 53.775, 33.9, 60.5025, 64.9, 86.2125, 297.585, 124.9925, 575.7775, 287.945, 215.9075, 128.4575,
               135.03, 114.05, 661.615, 427.28, 400.32, 224.4875, 227.6, 223.26, 228.1375, 250.4125, 253.57, 213.6725,
               212.75, 224.63, 226.415, 251.335, 196.1275, 244.62, 191.5825, 237.985, 272.2825, 223.4575, 215.7475,
               210.07, 200.8475, 210.45, 204.5025, 192.23, 177.8475, 76.2, 101.2125, 81.865, 37.3, 41.3525, 45.175,
               35.22, 74.7225, 109.0875, 130.4625, 304.9525, 244.1625, 138.9425, 250.245, 269.8375, 273.805, 103.99,
               35.8275, 60.2075, 51.3125, 34.155, 42.63, 33.42, 38.6175, 47.355, 35.865, 33.565, 51.23, 51.6075,
               29.6325]
    h4_day1 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
               0.0, 0.0, 0.0, 0.0, 0.0, 3.0825, 45.8825, 151.49, 139.575, 190.9825, 238.81, 287.815, 330.2425, 420.1375,
               395.8275, 421.6175, 455.7, 484.255, 493.9025, 503.7675, 500.1925, 523.1825, 542.5125, 540.675, 519.9975,
               520.5075, 517.9225, 521.765, 516.26, 498.1425, 495.745, 520.1375, 509.29, 435.2075, 459.205, 433.8,
               407.79, 385.28, 348.0675, 304.31, 227.685, 101.28, 131.46, 79.735, 29.0775, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
               0.0, 0.0, 0.0, 0.0, 0.0, 454.4175, 752.7025, 0.0, 0.0, 0.0, 9.5825, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
               0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    h4_day2 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
               0.0, 0.0, 0.0, 0.0, 0.0, 82.175, 139.53, 196.0075, 225.4075, 274.1725, 305.2925, 335.7425, 368.82,
               424.9125, 468.3425, 499.945, 508.6425, 532.16, 561.1425, 584.865, 586.1725, 609.3075, 620.7825, 626.2575,
               616.4975, 623.26, 664.8225, 550.585, 586.095, 669.345, 637.45, 648.275, 601.7175, 591.525, 567.1425,
               549.2325, 546.055, 503.9575, 471.285, 449.82, 418.2025, 382.0375, 560.1025, 0.0, 86.3175, 54.5825, 6.825,
               0.0, 0.0, 16.1775, 10.425, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
               0.0, 0.0, 0.0, 0.0, 0.0, 45.33, 0.0, 0.0, 45.2525]
    h4_day3 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
               0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 451.96, 1256.02, 0.0, 74.8325, 209.3425, 229.905, 291.2075, 323.475,
               355.105, 382.1525, 436.385, 449.29, 443.3075, 457.3975, 479.165, 416.74, 505.4925, 494.21, 483.56,
               483.3875, 448.215, 446.0375, 430.6275, 458.295, 455.5475, 474.5825, 498.6, 408.2225, 436.7825, 394.4875,
               349.85, 345.67, 266.6675, 235.835, 234.2825, 132.8625, 96.1, 1.435, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
               0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
               0.0, 0.0, 0.0]

    # Summary of the different days to a total Load Profile for each Household (to make it a bit easier to submit):
    fixed_load1 = h1_day1 + h1_day2 + h1_day3
    fixed_load2 = h2_day1 + h2_day2 + h2_day3
    fixed_load3 = h3_day1 + h3_day2 + h3_day3
    fixed_load4 = h4_day1 + h4_day2 + h4_day3

    # Fixed Load data of the different Households (Unit: Wh; LIST LENGTH: H)
    # REMARK on the format: This is a list that contains lists filled with Fixed Load data. In an hourly resolution
    # there should be (T/tau) + 1 many entries to interpolate enough values to cover each time period t (analogue to the
    # procedure for Global Radiation and Ambient Temperature data). Otherwise, if T many values are available, as in
    # this case, this data record is just adopted.
    # SUBMISSION:
    es_load_ind = [fixed_load1]

    # SL RELATED DATA
    # List of total amounts of energy that are scheduled to run/charge certain SLs in T (Unit: Wh; LIST LENGTH: J).
    # The same SLs are assumed for each household.
    car_total = 30000  # It is assumed that the electric car is charged to a range of 300km.
    wash_total = 500  # Estimated average energy consumption: 0.5kWh.
    dry_total = 1700  # Estimated average energy consumption: 1.7kWh.
    dish_total = 550  # Estimated average energy consumption: 0.55kWh.
    # SUBMISSION:
    ls_sltotal = [wash_total, dry_total, dish_total, car_total]

    # List of time periods t when it is not possible to run/charge certain SLs (No unit; LIST LENGTH: J)
    # Remark: As discussed in the thesis, it is assumed that individual households (except H4) can only charge the
    # electric car between 7 pm and 7 am. Accordingly, N1 contains the corresponding indices t for the respective
    # quarter hours in which charging is not permitted. All other SLs can be used at any time.
    # EXCEPTION: For H4 it has been assumed that the car can always be charged (thus a special case (see thesis) was
    # also tested); the list has to be adapted accordingly if H4 is involved.
    N1 = list(range(28, 76, 1)) + list(range(124, 172, 1)) + list(range(220, 268, 1))
    N2 = []  # no times are forbidden
    # SUBMISSION:
    Ns = [N2, N2, N2, N1]

    # List of runtimes for each SL (Number of time periods required; No unit; LIST LENGTH: J).
    # Remark: If there is no limit (for example for electric cars) choose value of T - 2 (explained in the thesis)
    runtime_car = 286  # The UC for the electric car is assumed to be unlimited as far as possible as explained
    runtime_wash = 15  # Average time for a single cycle: 3:45h
    runtime_dry = 14  # Average time for a single cycle: 3.5 h
    runtime_dish = 18  # Average time for a single cycle: 4:30h
    # SUBMISSION:
    runtimes_sl = [runtime_wash, runtime_dry, runtime_dish, runtime_car]

    # List of maximum energy to run the different SLs in any time period t (Unit: Wh; LIST LENGTH: J).
    # As discussed in the thesis, the following simplification is made: If a SL is cycle-constrained, then the limit
    # for the energy that can be scheduled per time unit t for this SL corresponds to the quotient
    # (total energy for SL j) / (number of periods required for once cycle of SL j).
    # A non-cycle-constrained SL, such as an electric car, is limited in particular by its maximum power consumption
    # (or by the maximum power to be made available).Depending on the available data (and its units), it is important to
    # check that the units are compatible (in this case Wh!) and that the temporal resolution is also taken into
    # account. For example, the power consumption of an electric car is usually limited by the wallbox (11kW), i.e.
    # the car can consume a maximum of 2750 watt hours of energy in a quarter of an hour!
    e_max_car = 2750  # limited by 11kW Wallbox and/or its own maximum power consumption
    e_max_wash = wash_total / runtime_wash
    e_max_dryer = dry_total / runtime_dry
    e_max_dishwasher = dish_total / runtime_dish
    # SUBMISSION:
    es_slmax = [e_max_wash, e_max_dryer, e_max_dishwasher, e_max_car]

    # BATTERY RELATED DATA
    # List of self-discharging rates for the batteries of each Household (No unit; LIST LENGTH: H).
    default_eta_bats = 0.00002
    # SUBMISSION:
    etas_bats = [default_eta_bats]

    # List of charging rates for the batteries of each Household (No unit; LIST LENGTH: H).
    default_eta_batc = 0.964
    # SUBMISSION:
    etas_batc = [default_eta_batc]

    # List of discharging rates for the batteries of each Household (No unit; LIST LENGTH: H).
    # IMPORTANT: This value must not be equal to zero because there would be issues caused by division by zero
    # in Constraint 2!
    default_eta_batd = 0.964
    # SUBMISSION:
    etas_batd = [default_eta_batd]

    # Battery Capacities for each Household (Unit: Wh; LIST LENGTH: H).
    cs_bat13 = 16380  # Capacity of the batteries of H1 and H3
    cs_bat24 = 9830  # Capacity of the batteries of H2 and H4
    # SUBMISSION:
    cs_bat = [cs_bat13]

    # Initial SOC of the battery of each household (Unit: Wh; LIST LENGTH: H).
    default_soc_initial = 0
    # SUBMISSION:
    socs_initial = [default_soc_initial]

    # List of maximum energy that can be charged into the different batteries in any time period t (Unit: Wh;
    # LIST LENGTH: H).
    # Remark: The nominal power of the battery is divided by the factor tau. This way, the power unit W can be converted
    # to the energy unit Wh, taking into account the desired temporal resolution.
    ps_batcmax1 = 12800
    ps_batcmax2 = 7680
    # SUBMISSION:
    es_batcmax = [ps_batcmax1 / tau]

    # List of maximum energy that can be discharged from the different batteries in any time period t (Unit: Wh;
    # LIST LENGTH: H).
    # Analogue procedure as above.
    ps_batdmax1 = 12800
    ps_batdmax2 = 7680
    # SUBMISSION:
    es_batdmax = [ps_batdmax1 / tau]

    model = SelfSufficiencyModel(T=T, tau=tau, H=H, J=J, legend_households=legend_households,
                                 legend_sls=legend_sls, mip_gap=mip_gap, time_limit=time_limit,
                                 start_date=start_date, periods=periods, resolution=resolution, es_stc=es_stc,
                                 ts_noct=ts_noct, ns=ns, cs_temp=cs_temp, etas_pr=etas_pr, etas_inv=etas_inv,
                                 gr_progs_h=gr_progs_h, temp_progs_h=temp_progs_h, Ns=Ns, es_load_ind=es_load_ind,
                                 ls_sltotal=ls_sltotal, runtimes_sl=runtimes_sl, es_slmax=es_slmax, etas_bats=etas_bats,
                                 etas_batc=etas_batc, etas_batd=etas_batd, cs_bat=cs_bat, socs_initial=socs_initial,
                                 es_batcmax=es_batcmax, es_batdmax=es_batdmax)

    # Run the methods of class SelfSufficiencyModel
    model.get_estimated_pv_data()
    model.get_total_load_profile()
    model.solve_self_sufficiency_model()
    model.get_profiles()
    model.get_datasets()
