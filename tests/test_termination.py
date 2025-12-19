"""Test termination conditions."""

# Copyright (c) 2025 Alliance for Energy Innovation, LLC

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

__authors__ = ["Weslley S. Pereira"]

import numpy as np
import pytest

from soogo.termination import (
    UnsuccessfulImprovement,
    RobustCondition,
    IterateNTimes,
)
from soogo import OptimizeResult


class TestUnsuccessfulImprovement:
    """Test suite for UnsuccessfulImprovement termination condition."""

    def test_initialization(self):
        """Test default initialization."""
        condition = UnsuccessfulImprovement()
        assert condition.threshold == 0.001
        assert condition.value_range == 0.0
        assert condition.lowest_value == float("inf")
        assert not condition.is_met()

    def test_custom_threshold(self):
        """Test initialization with custom threshold."""
        condition = UnsuccessfulImprovement(threshold=0.01)
        assert condition.threshold == 0.01

    def test_update_with_improvement(self):
        """Test when there is significant improvement."""
        condition = UnsuccessfulImprovement(threshold=0.001)

        # First evaluation
        out1 = OptimizeResult(
            nfev=5,
            fx=np.array([10.0]),
            fsample=np.array([10.0, 11.0, 12.0, 13.0, 14.0]),
            nobj=1,
        )
        condition.update(out1)
        # After first update, value_range should be 4.0 (14-10)
        # lowest_value should be 10.0
        assert condition.value_range == 4.0
        assert condition.lowest_value == 10.0
        # First time, improvement from inf to 10 is large, so not met
        assert not condition.is_met()

        # Second evaluation with significant improvement
        out2 = OptimizeResult(
            nfev=10,
            fx=np.array([5.0]),
            fsample=np.array(
                [10.0, 11.0, 12.0, 13.0, 14.0, 5.0, 6.0, 7.0, 8.0, 9.0]
            ),
            nobj=1,
        )
        condition.update(out2)
        # value_range should now be 9.0 (14-5)
        # improvement from 10 to 5 is 5, which is > 0.001 * 9 = 0.009
        assert condition.value_range == 9.0
        assert condition.lowest_value == 5.0
        assert not condition.is_met()

    def test_update_without_improvement(self):
        """Test when improvement is below threshold."""
        condition = UnsuccessfulImprovement(threshold=0.1)

        # Setup initial state
        out1 = OptimizeResult(
            nfev=3,
            fx=np.array([10.0]),
            fsample=np.array([10.0, 12.0, 14.0]),
            nobj=1,
        )
        condition.update(out1)
        assert condition.value_range == 4.0
        assert condition.lowest_value == 10.0

        # Small improvement that's below threshold
        out2 = OptimizeResult(
            nfev=5,
            fx=np.array([9.9]),
            fsample=np.array([10.0, 12.0, 14.0, 9.9, 11.0]),
            nobj=1,
        )
        condition.update(out2)
        # improvement is 10.0 - 9.9 = 0.1
        # threshold * value_range = 0.1 * 4.0 = 0.4
        # 0.1 <= 0.4, so condition should be met
        assert condition.is_met()

    def test_reset_clears_state(self):
        """Test reset clears internal state."""
        condition = UnsuccessfulImprovement()

        # Setup some state
        out = OptimizeResult(
            nfev=3,
            fx=np.array([5.0]),
            fsample=np.array([5.0, 10.0, 15.0]),
            nobj=1,
        )
        condition.update(out)
        assert condition.value_range > 0
        assert condition.lowest_value < float("inf")

        # Reset without keeping data knowledge
        condition.reset(keep_data_knowledge=False)
        assert condition.value_range == 0.0
        assert condition.lowest_value == float("inf")
        assert not condition.is_met()

    def test_reset_keeps_data_knowledge(self):
        """Test reset with keep_data_knowledge=True."""
        condition = UnsuccessfulImprovement()

        # Setup some state
        out = OptimizeResult(
            nfev=3,
            fx=np.array([5.0]),
            fsample=np.array([5.0, 10.0, 15.0]),
            nobj=1,
        )
        condition.update(out)
        old_value_range = condition.value_range
        old_lowest = condition.lowest_value

        # Reset keeping data knowledge
        condition.reset(keep_data_knowledge=True)
        assert condition.value_range == old_value_range
        assert condition.lowest_value == old_lowest
        assert not condition.is_met()

    def test_with_surrogate_model(self):
        """Test update with surrogate model."""
        from tests.acquisition.utils import MockSurrogateModel

        condition = UnsuccessfulImprovement()

        # Create mock surrogate with some data
        X_train = np.array([[0.5, 0.5], [0.3, 0.7]])
        Y_train = np.array([3.0, 20.0])
        model = MockSurrogateModel(X_train, Y_train)

        out = OptimizeResult(
            nfev=2,
            fx=np.array([10.0]),
            fsample=np.array([10.0, 12.0]),
            nobj=1,
        )
        condition.update(out, model)

        # value_range should consider model.Y as well
        # max from fsample is 12, max from Y_train is 20
        # min from fsample is 10, min from Y_train is 3
        # so value_range should be 20 - 3 = 17
        assert condition.value_range == 17.0
        assert condition.lowest_value == 3.0

    def test_multiple_objectives_raises_error(self):
        """Should raise error for multiple objectives."""
        condition = UnsuccessfulImprovement()

        out = OptimizeResult(
            nfev=2,
            fx=np.array([[5.0, 6.0]]),
            fsample=np.array([[5.0, 6.0], [7.0, 8.0]]),
            nobj=2,
        )

        with pytest.raises(AssertionError, match="single objective"):
            condition.update(out)

    def test_zero_evaluations(self):
        """Test update with zero evaluations does nothing."""
        condition = UnsuccessfulImprovement()

        out = OptimizeResult(nfev=0, fx=None, fsample=None, nobj=1)
        condition.update(out)

        # State should remain unchanged
        assert condition.value_range == 0.0
        assert condition.lowest_value == float("inf")
        assert not condition.is_met()


class TestRobustCondition:
    """Test suite for RobustCondition wrapper."""

    def test_initialization(self):
        """Test wrapping another condition."""
        inner = IterateNTimes(5)
        condition = RobustCondition(inner, period=10)

        assert condition.termination is inner
        assert condition.history.maxlen == 10
        assert len(condition.history) == 0
        assert not condition.is_met()

    def test_not_met_until_period_fills(self):
        """Should not be met until history fills with period evaluations."""
        inner = IterateNTimes(1)  # Always met after 1 iteration
        condition = RobustCondition(inner, period=3)

        out = OptimizeResult(
            nfev=1, fx=np.array([1.0]), fsample=np.array([1.0]), nobj=1
        )

        # First update - history has 1 True
        condition.update(out, None)
        assert len(condition.history) == 1
        assert not condition.is_met()  # Need 3 values

        # Second update - history has 2 Trues
        condition.update(out, None)
        assert len(condition.history) == 2
        assert not condition.is_met()  # Need 3 values

    def test_all_true_in_period(self):
        """Should be met when all values in period are True."""
        inner = IterateNTimes(1)
        condition = RobustCondition(inner, period=3)

        out = OptimizeResult(
            nfev=1, fx=np.array([1.0]), fsample=np.array([1.0]), nobj=1
        )

        # Fill history with True values
        for _ in range(3):
            condition.update(out, None)

        assert len(condition.history) == 3
        assert all(condition.history)
        assert condition.is_met()

    def test_mixed_history(self):
        """Should not be met with mixed True/False in history."""
        inner = IterateNTimes(2)  # Met after 2 iterations
        condition = RobustCondition(inner, period=3)

        out = OptimizeResult(
            nfev=1, fx=np.array([1.0]), fsample=np.array([1.0]), nobj=1
        )

        # First update: inner not met (count=1)
        condition.update(out, None)
        assert not condition.history[-1]

        # Second update: inner is met (count=2)
        condition.update(out, None)
        assert not condition.history[-1]

        # Third update: inner still met (count=3)
        condition.update(out, None)
        assert len(condition.history) == 3
        assert not all(condition.history)  # Has one False
        assert not condition.is_met()

    def test_reset_clears_history(self):
        """Reset should clear history and delegate to inner condition."""
        inner = IterateNTimes(5)
        condition = RobustCondition(inner, period=3)

        out = OptimizeResult(
            nfev=1, fx=np.array([1.0]), fsample=np.array([1.0]), nobj=1
        )

        # Add some history
        for _ in range(3):
            condition.update(out, None)
        assert len(condition.history) == 3
        assert inner.iterationCount == 3

        # Reset
        condition.reset()
        assert len(condition.history) == 0
        assert inner.iterationCount == 0

    def test_history_overflow(self):
        """Test that history deque properly removes old values."""
        inner = IterateNTimes(1)
        condition = RobustCondition(inner, period=3)

        out = OptimizeResult(
            nfev=1, fx=np.array([1.0]), fsample=np.array([1.0]), nobj=1
        )

        # Fill beyond period
        for _ in range(5):
            condition.update(out, None)

        # Should only keep last 3
        assert len(condition.history) == 3
        assert all(condition.history)


class TestIterateNTimes:
    """Test suite for IterateNTimes termination condition."""

    def test_initialization(self):
        """Test initialization with different values."""
        condition = IterateNTimes()
        assert condition.nTimes == 1
        assert condition.iterationCount == 0
        assert not condition.is_met()

        condition2 = IterateNTimes(10)
        assert condition2.nTimes == 10
        assert condition2.iterationCount == 0

    def test_is_met_after_n_iterations(self):
        """Should be met exactly after n iterations."""
        condition = IterateNTimes(5)

        for i in range(4):
            condition.update()
            assert not condition.is_met()
            assert condition.iterationCount == i + 1

        condition.update()
        assert condition.is_met()
        assert condition.iterationCount == 5

    def test_stays_met_after_n_iterations(self):
        """Should stay met after reaching n iterations."""
        condition = IterateNTimes(2)

        condition.update()
        condition.update()
        assert condition.is_met()

        # Continue updating
        condition.update()
        assert condition.is_met()
        assert condition.iterationCount == 3

    def test_reset(self):
        """Reset should restart counter."""
        condition = IterateNTimes(3)

        condition.update()
        condition.update()
        assert condition.iterationCount == 2
        assert not condition.is_met()

        condition.reset()
        assert condition.iterationCount == 0
        assert not condition.is_met()

    def test_update_with_arguments(self):
        """Update should work with any arguments (they're ignored)."""
        condition = IterateNTimes(2)

        out = OptimizeResult(
            nfev=1, fx=np.array([1.0]), fsample=np.array([1.0]), nobj=1
        )

        condition.update(out, None)
        assert condition.iterationCount == 1

        condition.update(out, model=None)
        assert condition.iterationCount == 2
        assert condition.is_met()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
