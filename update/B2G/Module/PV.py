import matplotlib.pyplot as plt

class PV:
    """
    PV module for building-to-grid simulation.

    Attributes:
        capacity (float): Nominal capacity of the PV in kW
        solar (float): Solar radiation in kW
        tamb (float): Ambient air temperature in C
    """

    def __init__(self, capacity, solar, tamb):
        self.capacity = capacity
        self.solar = solar
        self.tamb = tamb

    def generate(self):
        K = -3.7 / 1000
        Sol_ref = 1000
        Temp_ref = 25
        PV_gen = (solar / Sol_ref) * (1 + K * (tamb + (0.0256 * solar) - Temp_ref))

    def show(self):
        fig, ax = plt.subplots(
            nrows=1,
            figsize=(2, 1),
            dpi=500,
            constrained_layout=True
        )

        ax.plot(PV_gen, linewidth=1, color="green")
        ax.set_ylabel("PV(kW)", fontsize=7)
        ax.tick_params(axis='both', which='both', labelsize=7)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.set_xlabel(f"Timestep ({int(self.resolution*60)} minutes)", fontsize=7)
        ax.tick_params(axis='both', which='both', labelsize=7)
        ax.margins(x=0)

        plt.show()