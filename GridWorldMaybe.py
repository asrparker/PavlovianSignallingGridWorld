import learn
import time
import datetime

import numpy as np

from dynamic_plotterm import *

# control variables
index = 4
indexLast = 4
x = 0
indexIncrease = True

threshold = 400

c = 0
clast = 0

upticks = 0
dwnticks = 0
fup = 0
fdwn = 0

upC = 0
downC = 0
timeSince_upC = 0
timeSince_downC = 0
upTime = 0
downTime = 0

direction_size = 32
world_size = 3 * direction_size

upper_contact = 31
lower_contact = 0

# learning variables
gamma = 0.9
lamb = 0.9
alpha_W = 0.2  # *(1-lamb)
alpha_omega = 0.01

feature = 0

# set state once to get vector size
S = learn.manage_state(world_size)
S.update_state(0)
state_size = len(S.state_vector)

agent_up = learn.GTDGVF(gam=gamma, lam=lamb, alpha_W=alpha_W, alpha_omega=alpha_omega, size=state_size)
agent_down = learn.GTDGVF(gam=gamma, lam=lamb, alpha_W=alpha_W, alpha_omega=alpha_omega, size=state_size)

observe = DynamicPlot(window_x=50, title='Offline Pavlovian Control Testing', xlabel='Time Step', ylabel='Value',
                      num_plots=3)
observe.add_line('Command', 0)
observe.add_line('Prediction Up', 1)
observe.add_line('Prediction Down', 1)
observe.add_line('Learning Error Up', 2)
observe.add_line('Learning Error Down', 2)

start = time.time()
print(datetime.datetime.now())
try:
    while True:
        # get command
        indexLast = index

        if (indexIncrease is True and agent_up.predict > threshold) or index == upper_contact:
            indexIncrease = False
        elif (indexIncrease is False and agent_down.predict > threshold) or index == lower_contact:
            indexIncrease = True

        if indexIncrease is True:
            index = index + 1
        elif indexIncrease is False:
            index = index - 1

        if index == upper_contact or index == lower_contact:
            c = 1
        else:
            c = 0

        if index == upper_contact:
            upC += 1
            timeSince_upC = time.time() - upTime
            upTime = time.time()
            print(index, upC, timeSince_upC)
            print(datetime.datetime.now())
        elif index == lower_contact:
            downC += 1
            timeSince_downC = time.time() - downTime
            downTime = time.time()
            print(index, downC, timeSince_downC)
            print(datetime.datetime.now())

        # find active feature
        feature = 100
        # move index to a higher part of the space for moving down or being still
        if indexIncrease is True:
            feature = index
        elif indexIncrease is False:
            feature = index + direction_size
        else:
            feature = index + (2 * direction_size)

        if indexIncrease is True:
            agent_up.update(c, S.state_vector, rho=1)
            agent_down.update(c, S.state_vector, rho=0)
        elif indexIncrease is False:
            agent_up.update(c, S.state_vector, rho=0)
            agent_down.update(c, S.state_vector, rho=1)
        else:
            agent_up.update(c, S.state_vector, rho=0)
            agent_down.update(c, S.state_vector, rho=0)

        # if agent_up.predict >= 0.6:
        #     if fup == 0:
        #         fup = time.time() - start
        #         print(time.time() - start)
        #     upticks += 1
        # if agent_down.predict <= 0.6:
        #     if fdwn == 0:
        #         fdwn = time.time() - start
        #         print(time.time() - start)
        #     dwnticks += 1

        observe.update(x, [index, agent_up.predict, agent_down.predict, agent_up.delta, agent_down.delta])
        time.sleep(.000001)

        x += 1

except KeyboardInterrupt:
    print "interrupt received, stopping..."
    print "Total time:", (time.time() - start)
    print upticks, "up triggers"
    print dwnticks, "down triggers"
    print "First up trigger time:", fup
    print "First down trigger time:", fdwn
