import tensorflow as tf

class TimeStepper:

    def __init__(self, f):
        self.f = f

    @staticmethod
    def increment_controls(i, control_batch, control_check_batch, name="increment_controls"):
        with tf.variable_scope(name):
            control_batch = tf.concat((control_batch[:,1:],tf.expand_dims(control_check_batch[:,i], axis=1)),axis=1)
            i = i + 1

        return i, control_batch

    def integrate(i, h, state_batch, control_batch, control_check_batch, training, reuse, name):
        pass


class ForwardEuler(TimeStepper):

    def integrate(self, i, h, state_batch, control_batch, control_check_batch, training, reuse, name="time_stepping"):
        with tf.variable_scope(name):
            dstate_batch = self.f(h, state_batch, control_batch, training, reuse)
            i, control_batch = self.increment_controls(i, control_batch, control_check_batch)

            state_batch = state_batch + h * dstate_batch

        return i, state_batch, control_batch

class RungeKutta(TimeStepper):

    def integrate(self, i, h, state_batch, control_batch, control_check_batch, training, reuse, name="time_stepping"):
        with tf.variable_scope(name):
            k1 = self.f(h, state_batch, control_batch, training, reuse)

            i, control_batch = self.increment_controls(i, control_batch, control_check_batch)
            k2 = self.f(h,
                state_batch + k1 * h,
                control_batch,
                training, True)
            k3 = self.f(h,
                state_batch + k2 * h,
                control_batch,
                training, True)

            i, control_batch = self.increment_controls(i, control_batch, control_check_batch)
            k4 = self.f(h,
                state_batch + k3 * 2 * h,
                control_batch,
                training, True)

            state_batch = state_batch + (h/3.) * (k1 + 2*k2 + 2*k3 + k4)

        return i, state_batch, control_batch

if __name__ == "__main__":
    import numpy as np
    import params
    import plotting

    # f is the equation of a circle
    def f(h, state_batch, control_batch, training, reuse):
        dx = -state_batch[:,:,params.Y_IND]
        dy = state_batch[:,:,params.X_IND]
        dtheta = np.reshape(1., (1, 1))
        z = np.zeros((1, 1))
        state_batch = tf.stack((dx, dy, dtheta, z, z), axis=2)
        return state_batch

    # Define the time steppers
    fe = ForwardEuler(f)
    rk = RungeKutta(f)

    # Set up the integrations
    time_step = 0.1
    state_batch_ph = tf.placeholder(tf.float32, (1, 1, params.STATES))
    _, state_batch_next_fe, _ = fe.integrate(0, time_step, state_batch_ph, np.zeros((1, 2, 1)), np.zeros((1, 2, 1)), False, False)
    _, state_batch_next_rk, _ = rk.integrate(0, time_step, state_batch_ph, np.zeros((1, 2, 1)), np.zeros((1, 2, 1)), False, False)
    
    # Define initial conditions
    state_batch = np.zeros((1, 1, params.STATES))
    state_batch[:,:,params.X_IND] = 1
    state_batch[:,:,params.Y_IND] = 0
    state_batch[:,:,params.THETA_IND] = np.pi/2.

    feed_dict = {}
    states_fe = [state_batch]
    states_rk = [state_batch]
    with tf.Session() as sess:
        for i in range(200):
            # Step with Forward Euler
            feed_dict[state_batch_ph] = states_fe[-1]
            states_fe.append(sess.run(state_batch_next_fe, feed_dict=feed_dict))

            # Step with Runge-Kutta
            feed_dict[state_batch_ph] = states_rk[-1]
            states_rk.append(sess.run(state_batch_next_rk, feed_dict=feed_dict))

    # Plot the results
    states_fe = np.concatenate(states_fe, axis=1)
    states_rk = np.concatenate(states_rk, axis=1)
    plotting.plot_states([states_fe, states_rk])
