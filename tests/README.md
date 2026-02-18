# Tests and Limitations

This folder contains pytest-based tests for Tasks 1–3 and the RK4 integrator.

## Works well (passing tests)
- Task 1: robot follows spline centerline inside corridor when width and gains are properly tuned.
- Task 2: bidirectional swarms avoid collisions when Rsafe, repulsion gain, and lane offset are sufficient.
- Task 3: optical flow correctly detects opposite pedestrian directions in the top vs bottom halves for the synthetic video.
- Numerics: RK4 shows improved accuracy when dt is decreased.

## Does not work / limitations (expected failures)
- Task 1: if corridor width is too narrow relative to robot/controller aggressiveness, border violations occur.
  This is shown with an xfail test: `test_task1_fails_when_corridor_too_narrow`.

Recommended mitigation: increase corridor width, increase damping (kd), reduce kp, or reduce dt.
