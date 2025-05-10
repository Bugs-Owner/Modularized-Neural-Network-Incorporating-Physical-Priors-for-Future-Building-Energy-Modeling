import matplotlib.pyplot as plt

class Battery:
    """
    Battery module for building-to-grid simulation.

    Attributes:
        capacity (float): Nominal capacity of the battery in kWh
        max_charge_rate (float): Maximum charge rate in kW
        max_discharge_rate (float): Maximum discharge rate in kW
        efficiency_charge (float): Charging efficiency (0-1)
        efficiency_discharge (float): Discharging efficiency (0-1)
        soc_min (float): Minimum state of charge (0-1)
        soc_max (float): Maximum state of charge (0-1)
        soc (float): Current state of charge (0-1)
    """

    def __init__(self, capacity, max_charge_rate=0.25, max_discharge_rate=0.25,
                 efficiency_charge=0.95, efficiency_discharge=0.95,
                 soc_min=0.1, soc_max=0.9, initial_soc=0.5, resolution=15):
        """
        Initialize the battery with given parameters.

        Args:
            capacity (float): Nominal capacity of the battery in kWh
            max_charge_rate (float, optional): Maximum charge rate in kW. If None, defaults to 0.25C
            max_discharge_rate (float, optional): Maximum discharge rate in kW. If None, defaults to 0.25C
            efficiency_charge (float, optional): Charging efficiency (0-1). Defaults to 0.95
            efficiency_discharge (float, optional): Discharging efficiency (0-1). Defaults to 0.95
            soc_min (float, optional): Minimum state of charge (0-1). Defaults to 0.1
            soc_max (float, optional): Maximum state of charge (0-1). Defaults to 0.9
            initial_soc (float, optional): Initial state of charge (0-1). Defaults to 0.5
            resolution (int, optional): Time resolution. Defaults to 15 minutes)
        """
        # Battery Initialize
        self.capacity = capacity
        self.max_charge_rate = max_charge_rate
        self.max_discharge_rate = max_discharge_rate
        self.max_charge_power = max_charge_rate*capacity
        self.max_discharge_power = max_discharge_rate*capacity
        self.efficiency_charge = efficiency_charge
        self.efficiency_discharge = efficiency_discharge

        # SOC limits and initial state
        self.soc_min = max(0.0, min(soc_min, 1.0))  # [0-1]
        self.soc_max = max(self.soc_min, min(soc_max, 1.0))  # [min-1]
        self.soc = initial_soc

        # Time resolution
        self.resolution = resolution/60 # Convert from minute to hour

        # Power tracking
        self.power_tracking = []  # Actual power tracking

        # SOC tracking
        self.soc_tracking = []  # SOC tracking

    def simulate(self, power_series):
        """
        Simulate battery operation.

        Args:
            power_series (np.array, with shape N*1): positive for charging, negative for discharging in kW

        Returns:
            dict: operation data
        """
        self.power_series = power_series
        for t in range(power_series.shape[0]):
            self.soc_tracking.append(self.soc)
            # During operation stage, the real charging/discharging power is constrainted
            power = power_series[t]
            if power > 0:  # Charging
                # feasible charing power constrained by max_charge_power
                actual_power = min(power, self.max_charge_power)
                # power to energy
                energy_to_battery = actual_power * self.resolution * self.efficiency_charge
                # feasible charing power constrained by max soc
                actual_energy_to_battery = min(energy_to_battery, (self.soc_max-self.soc) * self.capacity)
                # update soc
                self.soc += actual_energy_to_battery/self.capacity
                self.power_tracking.append(actual_energy_to_battery/self.resolution)
            elif power == 0:  # Idle
                #TODO: Add storage power loss
                self.soc += 0
                self.power_tracking.append(0)
            else:
                # feasible charing power constrained by max_charge_power
                actual_power = max(power, -self.max_discharge_power)
                # power to energy
                energy_from_battery = actual_power * self.resolution / self.efficiency_discharge
                # feasible charing power constrained by max soc
                actual_energy_from_battery = max(energy_from_battery, (self.soc_min - self.soc) * self.capacity)
                # update soc
                self.soc += actual_energy_from_battery / self.capacity
                self.power_tracking.append(actual_energy_from_battery/self.resolution)

    def show(self):
        fig, axes = plt.subplots(
            nrows=2,
            figsize=(2, 1.8),
            dpi=500,
            constrained_layout=True,
            gridspec_kw={'height_ratios': [1, 3]}
        )

        axes[0].plot(self.soc_tracking, linewidth=1, color="black")
        axes[0].set_ylabel("SoC", fontsize=7)
        axes[0].tick_params(axis='both', which='both', labelsize=7)
        axes[0].tick_params(labelbottom=False)
        axes[0].grid(True, linestyle="--", alpha=0.4)
        axes[0].set_ylim(-0.05,1.05)

        axes[1].plot(self.power_tracking, linewidth=1, color="red", label="Actual")
        axes[1].plot(self.power_series, linewidth=1, color="blue", label="Signal")
        axes[1].set_ylabel("Power(kW)", fontsize=7)
        axes[1].set_xlabel(f"Timestep ({int(self.resolution*60)} minutes)", fontsize=7)
        axes[1].tick_params(axis='both', which='both', labelsize=7)
        axes[1].legend(loc='center', bbox_to_anchor=(0.5, 1.06), ncol=2, fontsize=7, frameon=False)
        axes[1].grid(True, linestyle="--", alpha=0.4)
        axes[1].margins(x=0)

        plt.show()