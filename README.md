# 0D1S3O - On Progress
## Abstract
This work addresses the problem of the economic feasibility of space mining, analyzing the transition from the initial focus on precious metals towards the exploitation of water on asteroids for in-situ markets. An optimization model is proposed that integrates R* search algorithms with economic and technical criteria to design commercially viable prospecting routes. The objective is to identify optimal sequences of asteroids that maximize the return on investment, considering mission costs, mining operation costs, and the market value of water in space. The results demonstrate that multi-objective economic optimization surpasses purely technical approaches, identifying routes that reach the break-even point within timeframes compatible with private investment.

## Problem Statement
Asteroid mining has captured the scientific and entrepreneurial imagination for decades, evolving significantly in its strategic focus. Initially, pioneers like Kargel (1996) [4] and Elvis (2014) [5] visualized asteroids as sources of precious metals, particularly platinum group metals (PGMs), arguing that their high market value (∼$70,000/kg) would justify the enormous costs of space missions. Kargel [4] calculated that a single metallic asteroid could contain up to 7,500 tons of platinum, representing a potential value of trillions of dollars.

However, more recent techno-economic analyses, particularly the work of Hein et al. (2019) in Acta Astronautica, have demonstrated that this approach faces nearly insurmountable economic obstacles in the short and medium term. Platinum mining requires processing rates orders of magnitude higher due to its low concentration (∼10⁻⁵), and the injection of even small quantities into the terrestrial market would cause a drastic price drop, eliminating profitability.

The new paradigm focuses on the mining of volatiles, especially water, for its use in space. This approach is significantly more viable because:

* It creates an in-situ market where water is sold as propellant for satellites and spacecraft, competing with the high cost of launch from Earth (∼$20,000/kg)

* It is technically less demanding, as the concentration of water in C-type asteroids is much higher (5-10%)

* It serves as a catalyst for a sustainable space economy

The central technical problem lies in the optimization of multi-asteroid prospecting routes. Foundational research such as Olympio (2011) [3] demonstrated that visiting multiple asteroids in a single mission reduces costs and increases returns. Building on this foundation, Yang et al. (2015) developed advanced methods using gravity assists and optimization algorithms like PSO to design low delta-V routes.

However, a critical gap persists: these models are optimized for minimum technical cost (delta-V), but do not comprehensively consider commercial economic feasibility, which must include mining operation costs, the in-situ business model, and potential revenue.

### Hypothesis:
The optimization of multi-asteroid prospecting routes using the R* search algorithm, which integrates low-cost transfers with gravity assists and an in-situ market economic model, will allow for the identification of commercially viable missions that overcome the limitations of purely technical optimization models.

### Objectives:

#### General:
To design and evaluate an optimization model for asteroid prospecting routes that, through the combination of the R* algorithm, low-cost orbital transfers, and an in-situ market economic model, demonstrates the commercial feasibility of a space-based prospecting and water supply service.

#### Specific:
* Develop an Economic Evaluation Model for the In-Situ Market
    * Integrate mining and transportation costs to orbital markets (Lunar Gateway Station, cis-lunar orbits)

    * Create an evaluation function that maximizes Return on Investment (ROI) based on space-based sale prices

* Implement the R* Search Algorithm for Multi-Asteroid Route Optimization

    * Adapt R* to explore asteroid sequences with economic criteria

    * Use the framework of orbital transfers and gravity assists for delta-V calculation

* Validate Against Traditional Mission Planning Approaches

    * Compare the performance of R* against the PSO of Yang[2] et al.

    * Use a database of known NEOs and criteria established in the literature (delta-V, spectral type, water content).

* Identify Commercially Viable Routes

    * Apply the integrated model to find routes with an achievable break-even point

    * Perform a sensitivity analysis of critical parameters in economic feasibilit

## Methodology

### Mechanic Orbital Specific Functions

#### Lambert Transfer

#### Find Gravity Assist Opportunities

#### Calcute Return Delta V

### R* Algorithm adapted for Economy Orbital Optimization
```
f(s) = g(s) + h(s) = -ROI(s) + h(s)
```

* g(s) = -ROI(s): Real acumulated cost
* h(s): Heurístic that estimates ROI maximum potencial of no visited asteroids.

### Main Algorithm

#### R* ASTEROID OPTIMIZATION

#### SELECT RANDOM INTERMEDIATE GOAL

#### A STAR TO RANDOM GOAL

### Integrated Economic Model
#### ROI Calculus
```
ROI(s) = (Itotal(s) - Ctotal(s)) / (Cdev + Claunch)
```
* Incomes: Itotal(s) = s.mwait * Pw
* Total Costs: Ctotal(s) = Cprop(s) + Cmin(s) + Cops(s) + Creturn(s)
* Propulsion Cost: Cprop(s) = Cfuel*m0*(1- e^(-s.deltaVused/(Iused*g0)))
* Mining Cost: Cmin(s) = tmining*costhour*|s.seq|
* Operations Cost: Cops(s) = s.tcurrent*costday
* Return Cost: Creturn(s) = deltaVreturn*Cfuel*mwet

## References:

[1].	Hein, A. M., et al. "A techno-economic analysis of asteroid mining." Acta Astronautica (2019)

[2].	Yang, H., et al. "Low-cost transfer between asteroids with distant orbits using multiple gravity assists." Advances in Space Research (2015)

[3].	Olympio, J.T. "Optimal control problem for low-thrust multiple asteroid tour missions." Journal of Guidance, Control, and Dynamics (2011)

[4].	Kargel, J.S. "Market value of asteroidal precious metals in an age of diminishing terrestrial resources." Engineering, Construction, and Operations in Space V (1996)

[5].	Elvis, M. "How many ore-bearing asteroids?" Planetary and Space Science (2014)

[6] Izzo, D. "Revisiting Lambert's Problem." Celestial Mechanics and Dynamical Astronomy, 121(1), 2014.

[7] Likhachev, M., & Stentz, A. "R* Search." Proceedings of the National Conference on Artificial Intelligence, 2005.
