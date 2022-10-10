### Implement Weight Decay Scheduling for Exploration as well as network training

class Schedule(object):
    def value(self, t):
        """Value of the schedule at time t"""
        raise NotImplementedError()


def linear_interpolation(l, r, alpha):
    return l + alpha * (r - l)

class BaseScheduler(object):
    def __init__(self, max_eps, opt_str):
        self.parse_opts(opt_str)
        self.prev_loss = self.loss = self.max_eps = self.epoch_length = float("nan")
        self.eps = 0.0
        self.max_eps = max_eps
        self.is_training = True
        self.epoch = 0
        self.batch = 0

    def __repr__(self):
        return '<BaseScheduler: eps {}, max_eps {}>'.format(self.eps, self.max_eps)

    def parse_opts(self, s):
        opts = s.split(',')
        self.params = {}
        for o in opts:
            if o.strip():
                key, val = o.split('=')
                self.params[key] = val

    def get_max_eps(self):
        return self.max_eps

    def get_eps(self):
        return self.eps

    def reached_max_eps(self):
        return abs(self.eps - self.max_eps) < 1e-3

    def step_batch(self, verbose=False):
        if self.is_training:
            self.batch += 1
        return

    def step_epoch(self, verbose=False):
        if self.is_training:
            self.epoch += 1
        return

    def update_loss(self, new_loss):
        self.prev_loss = self.loss
        self.loss = new_loss

    def train(self):
        self.is_training = True
        
    def eval(self):
        self.is_training = False
    
    # Set how many batches in an epoch
    def set_epoch_length(self, epoch_length):
        self.epoch_length = epoch_length

class LinearScheduler(BaseScheduler):

    def __init__(self, max_eps, opt_str):
        super(LinearScheduler, self).__init__(max_eps, opt_str)
        self.schedule_start = int(self.params['start'])
        self.schedule_length = int(self.params['length'])
        self.epoch_start_eps = self.epoch_end_eps = 0

    def __repr__(self):
        return '<LinearScheduler: start_eps {:.3f}, end_eps {:.3f}>'.format(
            self.epoch_start_eps, self.epoch_end_eps)

    def step_epoch(self, verbose = True):
        self.epoch += 1
        self.batch = 0
        if self.epoch < self.schedule_start:
            self.epoch_start_eps = 0
            self.epoch_end_eps = 0
        else:
            eps_epoch = self.epoch - self.schedule_start
            if self.schedule_length == 0:
                self.epoch_start_eps = self.epoch_end_eps = self.max_eps
            else:
                eps_epoch_step = self.max_eps / self.schedule_length
                self.epoch_start_eps = min(eps_epoch * eps_epoch_step, self.max_eps)
                self.epoch_end_eps = min((eps_epoch + 1) * eps_epoch_step, self.max_eps)
        self.eps = self.epoch_start_eps
#         if verbose:
#             logger.info("Epoch {:3d} eps start {:7.5f} end {:7.5f}".format(self.epoch, self.epoch_start_eps, self.epoch_end_eps))

    def step_batch(self):
        if self.is_training:
            self.batch += 1
            eps_batch_step = (self.epoch_end_eps - self.epoch_start_eps) / self.epoch_length
            self.eps = self.epoch_start_eps + eps_batch_step * (self.batch - 1)
            if self.batch > self.epoch_length:
                logger.warning('Warning: we expect {} batches in this epoch but this is batch {}'.format(self.epoch_length, self.batch))
                self.eps = self.epoch_end_eps

class PiecewiseSchedule(object):
    def __init__(self, endpoints, interpolation=linear_interpolation, outside_value=None):
        """Piecewise schedule.
        outside_value: 
            if the value is requested outside of all the intervals sepecified in
            `endpoints` this value is returned. If None then AssertionError is
            raised when outside value is requested.
        """
        idxes = [e[0] for e in endpoints]
        assert idxes == sorted(idxes)
        self._interpolation = interpolation
        self._outside_value = outside_value
        self._endpoints      = endpoints

    def value(self, t):
        """See Schedule.value"""
        for (l_t, l), (r_t, r) in zip(self._endpoints[:-1], self._endpoints[1:]):
            if l_t <= t and t < r_t:
                alpha = float(t - l_t) / (r_t - l_t)
                return self._interpolation(l, r, alpha)

        # t does not belong to any of the pieces, so doom.
        assert self._outside_value is not None
        return self._outside_value

### Exponentially decaying schedule
class ExponentialSchedule(object):
    def __init__(self,initial_p=1.0, final_p=0.01,decay=0.995):
        self.decay = decay
        self.initial_p = initial_p
        self.final_p = final_p
        
    def value(self,t):
        return max(self.final_p, (self.decay**t)*self.initial_p)


def optimizer_schedule(num_timesteps):
    lr_schedule = PiecewiseSchedule(
        [
            (0, 1e-1),
            (num_timesteps / 40, 1e-1),
            (num_timesteps / 8, 5e-2),
        ],
        outside_value=5e-2,
    )
    return lr_schedule