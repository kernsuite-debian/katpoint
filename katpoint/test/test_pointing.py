################################################################################
# Copyright (c) 2009-2021, National Research Foundation (SARAO)
#
# Licensed under the BSD 3-Clause License (the "License"); you may not use
# this file except in compliance with the License. You may obtain a copy
# of the License at
#
#   https://opensource.org/licenses/BSD-3-Clause
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
################################################################################

"""Tests for the pointing module."""
from __future__ import print_function, division, absolute_import

import sys
import warnings

if sys.version_info < (3,):
    import unittest2 as unittest
else:
    import unittest

import numpy as np

import katpoint


def assert_angles_almost_equal(x, y, **kwargs):
    def primary_angle(x):
        return x - np.round(x / (2.0 * np.pi)) * 2.0 * np.pi
    np.testing.assert_almost_equal(primary_angle(x - y), np.zeros(np.shape(x)), **kwargs)


class TestPointingModel(unittest.TestCase):
    """Test pointing model."""
    def setUp(self):
        az_range = katpoint.deg2rad(np.arange(-185.0, 275.0, 5.0))
        el_range = katpoint.deg2rad(np.arange(0.0, 86.0, 1.0))
        mesh_az, mesh_el = np.meshgrid(az_range, el_range)
        self.az = mesh_az.ravel()
        self.el = mesh_el.ravel()
        # Generate random parameter values with this spread
        self.param_stdev = katpoint.deg2rad(20. / 60.)
        self.num_params = len(katpoint.PointingModel())

    def test_pointing_model_load_save(self):
        """Test construction / load / save of pointing model."""
        params = katpoint.deg2rad(np.random.randn(self.num_params + 1))
        pm = katpoint.PointingModel(params[:-1])
        print('%r %s' % (pm, pm))
        pm2 = katpoint.PointingModel(params[:-2])
        self.assertEqual(pm2.values()[-1], 0.0, 'Unspecified pointing model params not zeroed')
        pm3 = katpoint.PointingModel(params)
        self.assertEqual(pm3.values()[-1], params[-2], 'Superfluous pointing model params not handled correctly')
        pm4 = katpoint.PointingModel(pm.description)
        self.assertEqual(pm4.description, pm.description, 'Saving pointing model to string and loading it again failed')
        self.assertEqual(pm4, pm, 'Pointing models should be equal')
        self.assertNotEqual(pm2, pm, 'Pointing models should be inequal')
        np.testing.assert_almost_equal(pm4.values(), pm.values(), decimal=6)
        try:
            self.assertEqual(hash(pm4), hash(pm), 'Pointing model hashes not equal')
        except TypeError:
            self.fail('PointingModel object not hashable')

    def test_pointing_closure(self):
        """Test closure between pointing correction and its reverse operation."""
        # Generate random pointing model
        params = self.param_stdev * np.random.randn(self.num_params)
        pm = katpoint.PointingModel(params)
        # Test closure on (az, el) grid
        pointed_az, pointed_el = pm.apply(self.az, self.el)
        az, el = pm.reverse(pointed_az, pointed_el)
        assert_angles_almost_equal(az, self.az, decimal=6, err_msg='Azimuth closure error for params=%s' % (params,))
        assert_angles_almost_equal(el, self.el, decimal=7, err_msg='Elevation closure error for params=%s' % (params,))

    def test_pointing_fit(self):
        """Test fitting of pointing model."""
        # Generate random pointing model and corresponding offsets on (az, el) grid
        params = self.param_stdev * np.random.randn(self.num_params)
        params[1] = params[9] = 0.0
        pm = katpoint.PointingModel(params.copy())
        delta_az, delta_el = pm.offset(self.az, self.el)
        # All parameters are enabled
        enabled_params = (np.arange(self.num_params) + 1).tolist()
        # Don't fit anything, but keep existing model
        fitted_params, sigma_params = pm.fit(self.az, self.el, delta_az, delta_el,
                                             enabled_params=[], keep_disabled_params=True)
        np.testing.assert_equal(fitted_params, params)
        np.testing.assert_equal(sigma_params, np.zeros(self.num_params))
        with self.assertWarns(FutureWarning):
            # Don't fit anything, and zero the model (deprecated)
            fitted_params, _ = pm.fit(self.az, self.el, delta_az, delta_el, enabled_params=[])
        np.testing.assert_equal(fitted_params, np.zeros(self.num_params))
        # Clear model explicitly and fit all parameters
        pm.set()
        fitted_params, _ = pm.fit(self.az, self.el, delta_az, delta_el,
                                  enabled_params=enabled_params, keep_disabled_params=True)
        np.testing.assert_almost_equal(fitted_params, params, decimal=9)
        np.testing.assert_equal(fitted_params, pm.values())
        # Don't clear model and refit all parameters - same result
        fitted_params, _ = pm.fit(self.az, self.el, delta_az, delta_el,
                                  enabled_params=enabled_params, keep_disabled_params=True)
        np.testing.assert_almost_equal(fitted_params, params, decimal=9)
        # Fit some different parameters and keep the rest
        pm = katpoint.PointingModel(params.copy())
        fitted_params, _ = pm.fit(self.az, self.el, delta_az + 0.001, delta_el,
                                  enabled_params=[1, 2, 3], keep_disabled_params=True)
        self.assertRaises(AssertionError, np.testing.assert_equal, fitted_params[:3], params[:3])
        np.testing.assert_equal(fitted_params[3:], params[3:])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Fit some different parameters and zero the rest
            fitted_params, _ = pm.fit(self.az, self.el, delta_az + 0.001, delta_el,
                                      enabled_params=[1, 2, 3], keep_disabled_params=False)
        self.assertRaises(AssertionError, np.testing.assert_equal, fitted_params[:3], params[:3])
        np.testing.assert_equal(fitted_params[3:], np.zeros(self.num_params - 3))
