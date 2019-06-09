#!/usr/bin/env python3

import numpy as np

class Embodiment:
    def __init__(self, relative_joint_frames, parent_ids, relative_link_frames=None, normalize_to=100):
        assert parent_ids[0] is None, "First element of learner_parent_ids must me 'None'!"