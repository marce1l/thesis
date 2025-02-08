# thesis

## Pseudo-random test maneuver generation for robustness tests

### Abstract

Power steering is a well-known and widely used automotive component. It
enables driving and defines the steering feel of automobiles. Power steering is a safety
critical system, therefore testing is an important step during the development cycle.
Software defects are made unintentionally during development, so it is important to test
the system thoroughly before the start of production.

Checking the robustness of a system is a form of testing. This can be achieved by
injecting faults during different test maneuvers. Making a robustness test maneuver from
car simulation or car driving measurements demands significant resources, therefore
these test maneuvers are available in a limited quantity only. For this reason, the system
can only be tested with the same test maneuvers. Testing with more test maneuvers could
mean finding more defects in the system which we would not have found because of the
number of test maneuvers or which we would have only found later in the development
cycle.

For increasing the testable operating points and the effectiveness of the robustness
test, I made a pseudo-random test maneuver generator algorithm. The algorithm makes
steering wheel angle and vehicle speed reference signals from randomly generated noise,
with considering the attributes of driving a car. With the help of this algorithm, we can
create an arbitrary number of reproducible, realistic test maneuvers quickly in the
operating point range determined by the test engineer. I examined the operation of the
completed algorithm in an automotive test environment.

With the use of the algorithm an arbitrary number of operating point ranges can
be tested. Because of the diversity of the generated test maneuvers, more operating points
can be tested at the specified operating point range, than with the two other test maneuver
creating methods. Thus, by using the algorithm we can increase the effectiveness of the
robustness tests and find defects in the system faster during development.

### Example

```python
import random
from Perlin.robustness_generator import UserParameters, RobustnessGenerator, random_ranges, RobustnessVisualizer

# generate n count of random vehicle speed ranges between min_vsp and max_vsp
ranges = random_ranges(min_vsp=25, max_vsp=70, count=6)
user_params = UserParameters(endpos=450, test_length_sec=4*60, vsp_ranges=ranges)

rg = RobustnessGenerator(user_params)
Time, vsp, stwa = rg.generate()

# saving maneuver parameters
rg.save_params(savedir="", fname="params.json")

# display maneuver with matplotlib
rv = RobustnessVisualizer(Time, vsp, stwa, rg.get_params())
rv.maneuver_plot()
```

![Maneuver plot](/examples/min25_max70_count6_240sec.png) Maneuver plot

#### Generated parameters

```json
{
  "seed": 1738971964,
  "test_length_sec": 240,
  "global_asp_limit": 700,
  "vsp_ranges": [
    {
      "low_lim": 33.86,
      "high_lim": 47.16,
      "percent": "1/8"
    },
    {
      "low_lim": 44.75,
      "high_lim": 62.04,
      "percent": "9/40"
    },
    {
      "low_lim": 26.08,
      "high_lim": 63.14,
      "percent": "1/10"
    },
    {
      "low_lim": 29.04,
      "high_lim": 60.42,
      "percent": "9/40"
    },
    {
      "low_lim": 34.54,
      "high_lim": 42.98,
      "percent": "3/20"
    },
    {
      "low_lim": 41.49,
      "high_lim": 61.27,
      "percent": "7/40"
    }
  ],
  "endpos": 450,
  "octave": 5,
  "min_stwa": 5,
  "endpos_till_vsp": 15,
  "perlin_length": 10000,
  "acc_ramp_rate": 10.0,
  "deacc_ramp_rate": 30.0
}
```

#### Regenerating maneuver from the parameters

```python
from Perlin.robustness_generator import RobustnessGenerator_from_json, RobustnessVisualizer

# regenerate maneuver from parameters
rg = RobustnessGenerator_from_json("params.json")
Time, vsp, stwa = rg.generate()

# display maneuver with matplotlib
rv = RobustnessVisualizer(Time, vsp, stwa, rg.get_params())
rv.maneuver_plot()
```
