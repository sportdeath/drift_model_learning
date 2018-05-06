import numpy as np
import tfdeploy as tfd

class DriftDynamics:
    def __init__(self):
        NUMPY_MODEL = "model/model.pkl"

        self.model = tfd.Model(NUMPY_MODEL)

        self.states_, self.controls_, self.dstates_ = self.model.get("states", "controls")

    def f(self, states, controls):
        dstates_.eval({states_: states, controls_: controls})

    def A(self):


