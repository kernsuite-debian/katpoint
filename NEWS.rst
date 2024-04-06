History
=======

0.10.1 (2023-10-10)
-------------------
* Fix deprecated NumPy type aliases and configparser attributes (#81)
* Flake8 cleanup (#82)

0.10 (2021-04-20)
-----------------
* Handle out-of-range projection inputs (#67, #68)
* Fix the sign of the NIAO term in the delay model (#75)
* Allow fixed parameters during pointing model fitting (#73)
* Handle katpoint v1 description strings in v0 (#76)
* Improve Timestamp arithmetic and comparison operators (#57)
* Update pointing and other documentation (#59, #65, #70)
* Minor improvements to requirements, testing (#62 - #64, #69, #71, #72, #74)
* Mark Python 2 support as deprecated (#61)

0.9 (2019-10-02)
----------------
* Add Antenna.array_reference_antenna utility function (#51)
* Vectorise Target.uvw (#49)
* Improve precision of flux model description string (#52)
* Produce documentation on readthedocs.org (#48)
* Add script that converts PSRCAT database into Catalogue (#16)

0.8 (2019-02-12)
----------------
* Improve UVW coordinates by using local and not global North (#46)
* Allow different target with same name in Catalogue (#44)
* Add support for polarisation in flux density models (#38)
* Fix tab completion in Catalogue (#39)
* More Python 3 and flake8 cleanup (#43)
* The GitHub repository is now public as well

0.7 (2017-08-01)
----------------
* Support Python 3 (#36)
* Improve DelayCorrection, adding description string and offset (#37)

0.6.1 (2017-07-20)
------------------
* Resolve issue with ska-sa/katdal#85 - SensorData rework (#34)

0.6 (2016-09-16)
----------------
* Initial release of katpoint
