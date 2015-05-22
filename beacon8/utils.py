import theano as _th

def create_param_state_as(other, initial_value=0):
    return _th.shared(other.get_value()*0 + initial_value,
        broadcastable=other.broadcastable,
        name='state_for_' + str(other.name)
    )
