from Module.Battery import Battery
import numpy as np

batt = Battery(
        capacity=20,  # 100 kWh battery
        max_charge_rate=0.3,  # 0.3 C
        max_discharge_rate=0.3,  # 0.3 C
        efficiency_charge=0.95,
        efficiency_discharge=0.95,
        soc_min=0.1,  # 10% minimum SOC
        soc_max=0.9,  # 90% maximum SOC
        initial_soc=0.5,  # Starting at 50% SOC
        resolution=15, # time resolution
    )

# Random a operation schedule
power_profile = np.zeros(96)
power_profile[:8*4] = 2
power_profile[8*4:12*4] = -4
power_profile[12*4:16*4] = 6
power_profile[16*4:22*4] = -8
power_profile[22*4:24*4] = 4

batt.simulate(power_profile)
batt.show()